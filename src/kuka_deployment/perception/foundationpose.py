"""Lazy adapter around an unmodified NVLabs FoundationPose checkout."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Iterator, Protocol

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
        for name, path in (("repository", config.repository), ("mesh", config.mesh),
                           ("weights", config.weights)):
            if not path.exists():
                raise FileNotFoundError(f"FoundationPose {name} path not found: {path}")
        if config.register_iterations <= 0 or config.track_iterations <= 0:
            raise ValueError("FoundationPose iteration counts must be positive")
        try:
            import trimesh
            import nvdiffrast.torch as dr
        except ImportError as exc:  # pragma: no cover - GPU environment only
            raise RuntimeError("FoundationPose GPU dependencies are unavailable") from exc

        repository = str(config.repository.resolve())
        if repository not in sys.path:
            sys.path.insert(0, repository)
        # Upstream resolves its model configs/weights relative to its checkout.
        os.environ["FOUNDATIONPOSE_WEIGHT_DIR"] = str(config.weights.resolve())
        try:
            with _working_directory(config.repository):
                from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
                mesh = trimesh.load(str(config.mesh.resolve()), force="mesh")
                mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32)
                mesh.vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
                scorer, refiner = ScorePredictor(), PoseRefinePredictor()
                glctx = dr.RasterizeCudaContext()
                self._estimator = FoundationPose(
                    model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                    scorer=scorer, refiner=refiner, debug_dir=str(config.debug_dir),
                    debug=0, glctx=glctx,
                )
                self._estimator.diameter = np.float32(self._estimator.diameter)
        except Exception as exc:
            raise RuntimeError(f"FoundationPose initialization failed: {exc}") from exc

    def register(self, frame: RGBDFrame, mask: np.ndarray) -> np.ndarray:
        try:
            pose = self._estimator.register(K=frame.K, rgb=frame.rgb, depth=frame.depth_m,
                                            ob_mask=mask, iteration=self.config.register_iterations)
            return np.asarray(pose, dtype=np.float32).reshape(4, 4)
        except Exception as exc:
            raise RuntimeError(f"FoundationPose registration failed: {exc}") from exc

    def track(self, frame: RGBDFrame) -> np.ndarray:
        try:
            pose = self._estimator.track_one(rgb=frame.rgb, depth=frame.depth_m, K=frame.K,
                                             iteration=self.config.track_iterations)
            return np.asarray(pose, dtype=np.float32).reshape(4, 4)
        except Exception as exc:
            raise RuntimeError(f"FoundationPose tracking failed: {exc}") from exc
