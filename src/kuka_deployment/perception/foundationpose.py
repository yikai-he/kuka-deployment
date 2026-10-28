"""Lazy adapter around an unmodified NVLabs FoundationPose checkout."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .interfaces import RGBDFrame


class PoseEstimator(Protocol):
    def register(self, frame: RGBDFrame, mask: np.ndarray) -> np.ndarray: ...
    def track(self, frame: RGBDFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class FoundationPoseConfig:
    repository: Path
    mesh: Path
    weights: Path
    debug_dir: Path
    register_iterations: int = 5
    track_iterations: int = 2


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class FoundationPoseEstimator:
    """Construct expensive GPU models once, then expose register/track methods.

    FoundationPose returns ``T_camera_object``; this convention is preserved.
    Imports occur only when this class is instantiated, so CPU tests can collect.
    """

    def __init__(self, config: FoundationPoseConfig) -> None:
        self.config = config
        self.dtype_diagnostics: dict[str, object] = {
            "register_calls": 0,
            "track_calls": 0,
            "register_input_dtypes": [],
            "track_input_dtypes": [],
            "output_dtypes": [],
        }
        for name, path in (
            ("repository", config.repository),
            ("mesh", config.mesh),
            ("weights", config.weights),
        ):
            if not path.exists():
                raise FileNotFoundError(f"FoundationPose {name} path not found: {path}")
        if config.register_iterations <= 0 or config.track_iterations <= 0:
            raise ValueError("FoundationPose iteration counts must be positive")
        try:
            import nvdiffrast.torch as dr
            import trimesh
        except ImportError as exc:  # pragma: no cover - GPU environment only
            raise RuntimeError(
                "FoundationPose GPU dependencies are unavailable"
            ) from exc

        repository = str(config.repository.resolve())
        if repository not in sys.path:
            sys.path.insert(0, repository)
        # Upstream resolves its model configs/weights relative to its checkout.
        os.environ["FOUNDATIONPOSE_WEIGHT_DIR"] = str(config.weights.resolve())
        try:
            with _working_directory(config.repository):
                from estimater import (
                    FoundationPose,
                    PoseRefinePredictor,
                    ScorePredictor,
                )

                mesh = trimesh.load(str(config.mesh.resolve()), force="mesh")
                # Trimesh deliberately stores geometry as float64, including values
                # assigned from float32 arrays.  Keep that native representation for
                # the mesh object while passing explicit float32 model arrays across
                # the FoundationPose boundary.
                model_pts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
                model_normals = np.ascontiguousarray(
                    mesh.vertex_normals, dtype=np.float32
                )
                scorer, refiner = ScorePredictor(), PoseRefinePredictor()
                glctx = dr.RasterizeCudaContext()
                self._estimator = FoundationPose(
                    model_pts=model_pts,
                    model_normals=model_normals,
                    mesh=mesh,
                    scorer=scorer,
                    refiner=refiner,
                    debug_dir=str(config.debug_dir),
                    debug=0,
                    glctx=glctx,
                )
                self._estimator.diameter = np.float32(self._estimator.diameter)
                self.dtype_diagnostics.update(
                    {
                        "mesh_vertices_dtype": str(mesh.vertices.dtype),
                        "mesh_vertex_normals_dtype": str(mesh.vertex_normals.dtype),
                        "model_pts_dtype": str(model_pts.dtype),
                        "model_normals_dtype": str(model_normals.dtype),
                        "diameter_dtype": str(
                            np.asarray(self._estimator.diameter).dtype
                        ),
                    }
                )
        except Exception as exc:
            raise RuntimeError(f"FoundationPose initialization failed: {exc}") from exc

    def register(self, frame: RGBDFrame, mask: np.ndarray) -> np.ndarray:
        try:
            self.dtype_diagnostics["register_calls"] = (
                int(self.dtype_diagnostics["register_calls"]) + 1
            )
            self.dtype_diagnostics["register_input_dtypes"].append(
                {
                    "K": str(frame.K.dtype),
                    "rgb": str(frame.rgb.dtype),
                    "depth": str(frame.depth_m.dtype),
                    "mask": str(np.asarray(mask).dtype),
                }
            )
            pose = self._estimator.register(
                K=frame.K,
                rgb=frame.rgb,
                depth=frame.depth_m,
                ob_mask=mask,
                iteration=self.config.register_iterations,
            )
            result = np.asarray(pose, dtype=np.float32).reshape(4, 4)
            self.dtype_diagnostics["output_dtypes"].append(str(result.dtype))
            return result
        except Exception as exc:
            raise RuntimeError(f"FoundationPose registration failed: {exc}") from exc

    def track(self, frame: RGBDFrame) -> np.ndarray:
        try:
            self.dtype_diagnostics["track_calls"] = (
                int(self.dtype_diagnostics["track_calls"]) + 1
            )
            self.dtype_diagnostics["track_input_dtypes"].append(
                {
                    "K": str(frame.K.dtype),
                    "rgb": str(frame.rgb.dtype),
                    "depth": str(frame.depth_m.dtype),
                }
            )
            pose = self._estimator.track_one(
                rgb=frame.rgb,
                depth=frame.depth_m,
                K=frame.K,
                iteration=self.config.track_iterations,
            )
            result = np.asarray(pose, dtype=np.float32).reshape(4, 4)
            self.dtype_diagnostics["output_dtypes"].append(str(result.dtype))
            return result
        except Exception as exc:
            raise RuntimeError(f"FoundationPose tracking failed: {exc}") from exc
