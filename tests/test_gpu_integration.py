"""Opt-in real-GPU regression for task1/episode_000.

Enable only with KUKA_GPU_REGRESSION=1.  This test intentionally uses the public
replay CLI so it cannot grow a second tracking/recovery implementation.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from kuka_deployment.perception.regression import run_gpu_preflight, write_json
from kuka_deployment.perception.transforms import validate_transform
from scripts.compare_perception_runs import compare_runs, load_run, render_markdown
from scripts.replay_perception import main as replay_main


pytestmark = [
    pytest.mark.gpu_integration,
    pytest.mark.skipif(os.environ.get("KUKA_GPU_REGRESSION") != "1",
                       reason="set KUKA_GPU_REGRESSION=1 to run real GPU regression"),
]


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} must be set for the GPU regression")
    return Path(value).expanduser().resolve()


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def test_task1_episode_000_gpu_regression(monkeypatch):
    dataset = _required_path("KUKA_GPU_DATASET")
    config = _required_path("KUKA_GPU_CONFIG")
    baseline = _required_path("KUKA_GPU_BASELINE")
    output = _required_path("KUKA_GPU_OUTPUT")
    if output.exists() and any(output.iterdir()):
        pytest.fail(f"KUKA_GPU_OUTPUT must be a fresh empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    preflight = run_gpu_preflight(config, dataset)
    write_json(output / "preflight.json", preflight)
    failures = [item for item in preflight["checks"] if not item["passed"]]
    assert preflight["passed"], f"GPU preflight failed; see {output / 'preflight.json'}: {failures}"

    monkeypatch.setattr(
        "sys.argv",
        ["replay_perception.py", "--input", str(dataset), "--config", str(config),
         "--output", str(output), "--save-reproducibility-manifest"],
    )
    assert replay_main() == 0

    poses = np.load(output / "poses.npy", allow_pickle=False)
    with (output / "poses.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert poses.shape == (429, 4, 4)
    assert len(rows) == 429
    for row, pose in zip(rows, poses, strict=True):
        if _is_true(row["valid"]):
            validate_transform(pose)
        else:
            assert np.isnan(pose).all(), f"invalid frame {row['frame_number']} retained a pose"

    manifest_path = output / "reproducibility_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dtypes = manifest["foundationpose_dtype_diagnostics"]
    assert dtypes["model_pts_dtype"] == "float32"
    assert dtypes["model_normals_dtype"] == "float32"
    assert dtypes["diameter_dtype"] == "float32"
    assert dtypes["register_calls"] >= 1
    assert dtypes["track_calls"] >= 1
    assert {item["depth"] for item in dtypes["register_input_dtypes"]} == {"float32"}
    assert {item["depth"] for item in dtypes["track_input_dtypes"]} == {"float32"}
    assert set(dtypes["output_dtypes"]) == {"float32"}

    baseline_kind = os.environ.get("KUKA_GPU_BASELINE_KIND", "legacy")
    translation_tolerance = float(os.environ.get("KUKA_GPU_TRANSLATION_TOLERANCE_M", "0.005"))
    rotation_tolerance = float(os.environ.get("KUKA_GPU_ROTATION_TOLERANCE_DEG", "2.0"))
    comparison = compare_runs(
        load_run(baseline, baseline_kind), load_run(output, "modular"),
        translation_tolerance_m=translation_tolerance,
        rotation_tolerance_deg=rotation_tolerance,
    )
    write_json(output / "baseline_comparison.json", comparison)
    (output / "baseline_comparison.md").write_text(
        render_markdown(comparison), encoding="utf-8")

    deviations = list(preflight["deviations"])
    frames = comparison["frame_correspondence"]
    validity = comparison["validity"]
    translation = comparison["translation_error_m"]
    rotation = comparison["cube_symmetry_rotation_error_deg"]
    if frames["legacy_count"] != 429 or frames["modular_count"] != 429 or frames["matched_count"] != 429:
        deviations.append(f"frame correspondence differs: {frames}")
    if validity["agreement_rate"] != 1.0:
        deviations.append(f"validity differs: {validity}")
    if translation["count"] and translation["max"] > translation_tolerance:
        deviations.append(f"translation max {translation['max']} exceeds {translation_tolerance} m")
    if rotation["count"] and rotation["max"] > rotation_tolerance:
        deviations.append(f"rotation max {rotation['max']} exceeds {rotation_tolerance} deg")
    registration = comparison["tracking_and_recovery"]["first_successful_registration"]
    if registration["legacy"] != registration["modular"]:
        deviations.append(f"first successful registration differs: {registration}")
    events = comparison["tracking_and_recovery"]
    for field in ("mode_agreement", "recovery_attempt_agreement", "recovery_success_agreement"):
        agreement = events[field]
        if agreement["comparable"] and agreement["agreements"] != agreement["comparable"]:
            deviations.append(f"{field} differs: {agreement}")
    if events["lost_intervals"]["legacy"] != events["lost_intervals"]["modular"]:
        deviations.append(f"LOST intervals differ: {events['lost_intervals']}")

    manifest["preflight"] = preflight
    manifest["baseline"] = {
        "path": str(baseline), "kind": baseline_kind,
        "translation_tolerance_m": translation_tolerance,
        "rotation_tolerance_deg": rotation_tolerance,
    }
    manifest["deviations_from_validated_baseline"] = deviations
    write_json(manifest_path, manifest)
    if os.environ.get("KUKA_GPU_FAIL_ON_DEVIATION", "1") == "1":
        assert not deviations, f"baseline deviations recorded in {manifest_path}: {deviations}"
