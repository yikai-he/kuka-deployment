"""GPU regression preflight and reproducibility helpers.

This module does not construct or alter the perception pipeline.  It validates the
external environment used by the opt-in integration test and records provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import yaml

from .sources import RecordedSequenceSource


@dataclass(frozen=True)
class RegressionExpectations:
    frame_count: int = 429
    width: int = 640
    height: int = 480
    cube_size_m: float = 0.040
    cube_size_tolerance_m: float = 0.001
    register_iterations: int = 5
    track_iterations: int = 2
    gpu_name_contains: str = "T4"
    foundationpose_commit: str = "a1b694b83e633c2cb6115b9063d940a687759392"


def _check(report: dict[str, Any], name: str, passed: bool, detail: Any) -> None:
    report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})


def _file_record(path: Path, *, digest: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    record: dict[str, Any] = {"path": str(resolved), "size_bytes": resolved.stat().st_size}
    if digest:
        value = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        record["sha256"] = value.hexdigest()
    return record


def run_gpu_preflight(config_path: str | Path, dataset: str | Path,
                      expectations: RegressionExpectations = RegressionExpectations()) -> dict[str, Any]:
    """Return a complete check report; callers decide whether failed checks are fatal."""
    config_path, dataset = Path(config_path).resolve(), Path(dataset).resolve()
    report: dict[str, Any] = {
        "passed": False, "expectations": asdict(expectations), "checks": [], "deviations": [],
    }
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        _check(report, "cuda_available", cuda, {
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count() if cuda else 0,
        })
        device_name = torch.cuda.get_device_name(0) if cuda else None
        _check(report, "nvidia_gpu", cuda and bool(device_name), device_name)
        if device_name and expectations.gpu_name_contains.lower() not in device_name.lower():
            report["deviations"].append(
                f"GPU is {device_name!r}; validated baseline used {expectations.gpu_name_contains!r}")
    except Exception as exc:
        _check(report, "cuda_available", False, repr(exc))
        _check(report, "nvidia_gpu", False, "PyTorch CUDA inspection failed")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        _check(report, "nvidia_smi", bool(result.stdout.strip()), result.stdout.strip().splitlines())
    except Exception as exc:
        _check(report, "nvidia_smi", False, repr(exc))

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        _check(report, "configuration", isinstance(config, dict), str(config_path))
    except Exception as exc:
        _check(report, "configuration", False, repr(exc))
        report["passed"] = False
        return report

    models = config.get("models", {})
    for field, expected in (("register_iterations", expectations.register_iterations),
                            ("track_iterations", expectations.track_iterations)):
        actual = models.get(field)
        _check(report, field, actual == expected, {"actual": actual, "expected": expected})
    for name, kind in (("yolo_weights", "file"), ("foundationpose_repository", "dir"),
                       ("foundationpose_weights", "dir"), ("mesh", "file")):
        path = Path(str(models.get(name, ""))).expanduser()
        exists = path.is_file() if kind == "file" else path.is_dir()
        _check(report, name, exists, str(path.resolve()) if path.exists() else str(path))

    weights = Path(str(models.get("foundationpose_weights", ""))).expanduser()
    weight_files = sorted(weights.rglob("*.pth")) if weights.is_dir() else []
    _check(report, "foundationpose_weight_files", bool(weight_files),
           [str(path.resolve()) for path in weight_files])

    repository = Path(str(models.get("foundationpose_repository", ""))).expanduser()
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        matches = commit == expectations.foundationpose_commit
        _check(report, "foundationpose_commit", matches, {
            "actual": commit, "expected": expectations.foundationpose_commit,
        })
        if not matches:
            report["deviations"].append("FoundationPose commit differs from validated baseline")
    except Exception as exc:
        _check(report, "foundationpose_commit", False, repr(exc))

    try:
        source = RecordedSequenceSource(
            dataset,
            min_depth_m=float(config.get("source", {}).get("min_depth_m", 0.001)),
            max_depth_m=float(config.get("source", {}).get("max_depth_m", 3.0)),
        )
        _check(report, "rgb_depth_pairing", True, {"first": source.ids[0], "last": source.ids[-1]})
        _check(report, "frame_count", len(source.ids) == expectations.frame_count, len(source.ids))
        K = source.K
        intrinsics_ok = (K.shape == (3, 3) and np.isfinite(K).all()
                         and K[0, 0] > 0 and K[1, 1] > 0 and np.isclose(K[2, 2], 1))
        _check(report, "camera_intrinsics", intrinsics_ok, K.tolist())
        first = next(iter(source))
        resolution = [int(first.rgb.shape[1]), int(first.rgb.shape[0])]
        _check(report, "resolution", resolution == [expectations.width, expectations.height], resolution)
        _check(report, "rgbd_contract", True, {
            "rgb_dtype": str(first.rgb.dtype), "depth_dtype": str(first.depth_m.dtype),
            "frame_id": first.frame_id,
        })
    except Exception as exc:
        _check(report, "dataset", False, repr(exc))

    try:
        import trimesh
        mesh = trimesh.load(str(Path(models["mesh"]).expanduser().resolve()), force="mesh")
        extents = np.asarray(mesh.extents, dtype=np.float64)
        expected = np.full(3, expectations.cube_size_m)
        scale_ok = extents.shape == (3,) and np.allclose(
            extents, expected, atol=expectations.cube_size_tolerance_m, rtol=0)
        _check(report, "cad_mesh_scale", scale_ok, {
            "extents_m": extents.tolist(), "expected_extents_m": expected.tolist(),
        })
    except Exception as exc:
        _check(report, "cad_mesh_scale", False, repr(exc))

    report["passed"] = all(item["passed"] for item in report["checks"])
    return report


def reproducibility_manifest(config_path: str | Path, dataset: str | Path,
                             estimator_dtypes: dict[str, Any],
                             preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = config["models"]
    packages = {}
    for name in ("numpy", "opencv-python", "PyYAML", "torch", "torchvision", "trimesh",
                 "nvdiffrast", "ultralytics"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    assets = {
        "configuration": _file_record(config_path),
        "mesh": _file_record(Path(models["mesh"])),
        "yolo_weights": _file_record(Path(models["yolo_weights"])),
        "foundationpose_weights": [
            _file_record(path) for path in sorted(Path(models["foundationpose_weights"]).rglob("*.pth"))
        ],
    }
    fp_repository = Path(models["foundationpose_repository"]).resolve()
    try:
        fp_commit = subprocess.run(
            ["git", "-C", str(fp_repository), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except Exception:
        fp_commit = None
    return {
        "schema_version": 1,
        "platform": {"python": platform.python_version(), "system": platform.platform()},
        "packages": packages,
        "dataset_path": str(Path(dataset).resolve()),
        "foundationpose_repository": str(fp_repository),
        "foundationpose_commit": fp_commit,
        "effective_config": config,
        "assets": assets,
        "foundationpose_dtype_diagnostics": estimator_dtypes,
        "preflight": preflight,
    }


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
