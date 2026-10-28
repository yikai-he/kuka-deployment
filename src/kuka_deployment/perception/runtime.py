"""Shared session construction, timing, and output records for perception CLIs."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .foundationpose import FoundationPoseConfig, FoundationPoseEstimator
from .interfaces import ObjectPoseEstimate, RGBDFrame
from .segmenter import MaskValidationConfig, RedCubeCandidateConfig, YoloSegmenter
from .tracker import (
    PerceptionPipeline,
    PipelineConfig,
    PoseValidationConfig,
    RecoveryConfig,
)


def build_pipeline(
    config: Mapping[str, Any], output_dir: Path, *, initial_mask: np.ndarray | None = None
) -> PerceptionPipeline:
    """Build model instances once for one offline or live processing session."""
    models, validation = config["models"], config["validation"]
    segmentation = config["segmentation"]
    recovery = config["recovery"]
    candidate_values = dict(config["red_cube"])
    candidate_values.update(
        {
            key: recovery[key]
            for key in (
                "recovery_hint_weight",
                "recovery_hint_max_px",
                "lost_hint_expand_px",
                "hint_fallback_lost_streak",
            )
        }
    )
    candidate_values["class_id"] = int(segmentation["class_id"])
    segmenter = YoloSegmenter(
        models["yolo_weights"],
        confidence=float(segmentation["confidence"]),
        class_id=int(segmentation["class_id"]),
        device=str(segmentation.get("device", "cuda:0")),
        iou=float(segmentation["iou"]),
        image_size=int(segmentation["image_size"]),
        max_detections=int(segmentation["max_detections"]),
        candidate_config=RedCubeCandidateConfig(**candidate_values),
    )
    estimator = FoundationPoseEstimator(
        FoundationPoseConfig(
            repository=Path(models["foundationpose_repository"]),
            mesh=Path(models["mesh"]),
            weights=Path(models["foundationpose_weights"]),
            debug_dir=output_dir / "debug",
            register_iterations=int(models["register_iterations"]),
            track_iterations=int(models["track_iterations"]),
        )
    )
    return PerceptionPipeline(
        str(config["object_id"]),
        segmenter,
        estimator,
        PipelineConfig(
            mask=MaskValidationConfig(**validation["mask"]),
            pose=PoseValidationConfig(**validation["pose"]),
            recovery=RecoveryConfig(
                enabled=bool(recovery["enabled"]),
                cooldown_frames=int(recovery["cooldown_frames"]),
            ),
        ),
        initial_mask=initial_mask,
    )


def synchronize_cuda_if_available() -> None:
    """Synchronize CUDA lazily, without making torch a core import dependency."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def process_frame_timed(
    pipeline: PerceptionPipeline,
    frame: RGBDFrame,
    *,
    synchronize: Callable[[], None] | None = None,
) -> tuple[ObjectPoseEstimate, float, float | None]:
    """Process one frame and return inference and measurable end-to-end latency."""
    if synchronize is not None:
        synchronize()
    started_ns = time.monotonic_ns()
    result = pipeline.process_frame(frame)
    if synchronize is not None:
        synchronize()
    finished_ns = time.monotonic_ns()
    inference_ms = (finished_ns - started_ns) / 1_000_000
    end_to_end_ms = (
        None
        if frame.acquisition_monotonic_ns is None
        else (finished_ns - frame.acquisition_monotonic_ns) / 1_000_000
    )
    return result, inference_ms, end_to_end_ms


def result_record(
    frame: RGBDFrame,
    result: ObjectPoseEstimate,
    pipeline: PerceptionPipeline,
    inference_ms: float,
    *,
    end_to_end_ms: float | None = None,
) -> dict[str, Any]:
    """Create the common serializable pose/status record used by both modes."""
    pose = result.T_frame_object
    record: dict[str, Any] = {
        "frame_number": frame.frame_number,
        "timestamp_ns": result.timestamp_ns,
        "frame_id": result.frame_id,
        "object_id": result.object_id,
        "coordinate_convention": "T_camera_object (object coordinates to frame_id)",
        "valid": result.valid,
        "status": result.status.value,
        "detector_confidence": result.detector_confidence,
        "failure_reason": result.failure_reason,
        "inference_ms": inference_ms,
        "end_to_end_ms": end_to_end_ms,
        "T_camera_object": None if pose is None else np.asarray(pose).tolist(),
        "failure_stage": pipeline.last_diagnostics.get("failure_stage"),
        "recovery_attempted": pipeline.last_diagnostics.get(
            "recovery_attempted", False
        ),
        "recovery_success": pipeline.last_diagnostics.get("recovery_success", False),
        "tracker_state": pipeline.last_diagnostics.get("state"),
        "lost_streak": pipeline.last_diagnostics.get("lost_streak", 0),
        "cooldown_remaining": pipeline.last_diagnostics.get("cooldown_remaining", 0),
        "depth_check_status": pipeline.last_diagnostics.get("depth_check_status"),
    }
    return record
