#!/usr/bin/env python3
"""Replay a recorded RGB-D sequence through YOLO-Seg + FoundationPose."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_deployment.perception.foundationpose import FoundationPoseConfig, FoundationPoseEstimator
from kuka_deployment.perception.segmenter import (
    MaskValidationConfig, RedCubeCandidateConfig, YoloSegmenter,
)
from kuka_deployment.perception.sources import RecordedSequenceSource
from kuka_deployment.perception.regression import reproducibility_manifest, write_json
from kuka_deployment.perception.tracker import (
    PerceptionPipeline, PipelineConfig, PoseValidationConfig, RecoveryConfig,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--save-visualization", action="store_true")
    parser.add_argument("--save-reproducibility-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    models, validation = config["models"], config["validation"]
    args.output.mkdir(parents=True, exist_ok=True)

    source_config = config["source"]
    source = RecordedSequenceSource(
        args.input, min_depth_m=float(source_config.get("min_depth_m", 0.001)),
        max_depth_m=float(source_config["max_depth_m"]),
    )
    segmentation_config = config["segmentation"]
    recovery_config = config["recovery"]
    candidate_values = dict(config["red_cube"])
    candidate_values.update({key: recovery_config[key] for key in (
        "recovery_hint_weight", "recovery_hint_max_px", "lost_hint_expand_px",
        "hint_fallback_lost_streak")})
    candidate_values["class_id"] = int(segmentation_config["class_id"])
    segmenter = YoloSegmenter(
        models["yolo_weights"], confidence=float(segmentation_config["confidence"]),
        class_id=int(segmentation_config["class_id"]),
        device=str(segmentation_config.get("device", "cuda:0")),
        iou=float(segmentation_config["iou"]), image_size=int(segmentation_config["image_size"]),
        max_detections=int(segmentation_config["max_detections"]),
        candidate_config=RedCubeCandidateConfig(**candidate_values),
    )
    estimator = FoundationPoseEstimator(FoundationPoseConfig(
        repository=Path(models["foundationpose_repository"]), mesh=Path(models["mesh"]),
        weights=Path(models["foundationpose_weights"]), debug_dir=args.output / "debug",
        register_iterations=int(models["register_iterations"]),
        track_iterations=int(models["track_iterations"]),
    ))
    initial_mask = None
    if recovery_config.get("allow_initial_mask_fallback", False):
        import cv2
        template = recovery_config.get("initial_mask")
        if not template:
            raise ValueError("initial_mask is required when initial-mask fallback is enabled")
        initial_path = Path(str(template).format(frame_id=source.ids[0]))
        if not initial_path.is_absolute():
            initial_path = args.input / initial_path
        raw_mask = cv2.imread(str(initial_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(f"initial fallback mask not found: {initial_path}")
        initial_mask = raw_mask > 0
    pipeline = PerceptionPipeline(
        config["object_id"], segmenter, estimator,
        PipelineConfig(mask=MaskValidationConfig(**validation["mask"]),
                       pose=PoseValidationConfig(**validation["pose"]),
                       recovery=RecoveryConfig(enabled=bool(recovery_config["enabled"]),
                                               cooldown_frames=int(recovery_config["cooldown_frames"]))),
        initial_mask=initial_mask,
    )

    rows, poses, timings = [], [], []
    writer = None
    output_config = config.get("outputs", {})
    save_visualization = args.save_visualization or bool(output_config.get("save_visualization", False))
    save_pose_files = bool(output_config.get("save_pose_files", False))
    if save_pose_files:
        (args.output / "poses").mkdir(exist_ok=True)
    if save_visualization:
        import cv2
    try:
        for frame in source:
            started = time.perf_counter()
            result = pipeline.process_frame(frame)
            elapsed_ms = (time.perf_counter() - started) * 1000
            pose = result.T_frame_object if result.valid else np.full((4, 4), np.nan, np.float32)
            poses.append(pose)
            timings.append(elapsed_ms)
            row = {
                "frame_number": frame.frame_number, "timestamp_ns": result.timestamp_ns,
                "frame_id": result.frame_id, "object_id": result.object_id,
                "valid": result.valid, "status": result.status.value,
                "detector_confidence": result.detector_confidence,
                "failure_reason": result.failure_reason, "inference_ms": elapsed_ms,
                "failure_stage": pipeline.last_diagnostics.get("failure_stage"),
                "recovery_attempted": pipeline.last_diagnostics.get("recovery_attempted", False),
                "recovery_success": pipeline.last_diagnostics.get("recovery_success", False),
                "tracker_state": pipeline.last_diagnostics.get("state"),
                "lost_streak": pipeline.last_diagnostics.get("lost_streak", 0),
                "cooldown_remaining": pipeline.last_diagnostics.get("cooldown_remaining", 0),
                "depth_check_status": pipeline.last_diagnostics.get("depth_check_status"),
                "step_translation_m": pipeline.last_diagnostics.get("step_translation_m"),
                "center_depth_m": pipeline.last_diagnostics.get("center_depth_m"),
                "center_depth_residual_m": pipeline.last_diagnostics.get("center_depth_residual_m"),
                "register_mask_center_error_m": pipeline.last_diagnostics.get("register_mask_center_error_m"),
                "candidate_count": pipeline.last_diagnostics.get("candidate_count"),
                "selected_mask_pixels": pipeline.last_diagnostics.get("mask_pixels"),
                "selected_size_m": pipeline.last_diagnostics.get("equiv_side_m"),
                "selected_hint_distance_px": pipeline.last_diagnostics.get("hint_distance_px"),
                "fallback_without_hint": pipeline.last_diagnostics.get("fallback_without_hint", False),
            }
            for r in range(4):
                for c in range(4):
                    row[f"T{r}{c}"] = float(pose[r, c])
            rows.append(row)
            if save_pose_files:
                np.savetxt(args.output / "poses" / f"{frame.frame_number:06d}.txt", pose, fmt="%.8f")
            print(f"frame={frame.frame_number} status={result.status.value} "
                  f"valid={result.valid} time_ms={elapsed_ms:.1f}")
            if save_visualization:
                image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                color = (0, 180, 0) if result.valid else (0, 0, 255)
                cv2.putText(image, result.status.value, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2, cv2.LINE_AA)
                if writer is None:
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(str(args.output / "tracking.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                                             (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("could not open visualization video writer")
                writer.write(image)
    finally:
        if writer is not None:
            writer.release()

    if not rows:
        raise RuntimeError("source yielded no frames")
    np.save(args.output / "poses.npy", np.stack(poses))
    with (args.output / "poses.csv").open("w", newline="", encoding="utf-8") as stream:
        output = csv.DictWriter(stream, fieldnames=list(rows[0]))
        output.writeheader()
        output.writerows(rows)
    with (args.output / "metrics.json").open("w", encoding="utf-8") as stream:
        valid_count = sum(row["valid"] for row in rows)
        first_registration = next((row["frame_number"] for row in rows if row["status"] in
                                   {"registered", "reregistered"} and row["valid"]), None)
        json.dump({"frames": len(rows), "valid_frames": valid_count,
                   "invalid_frames": len(rows) - valid_count,
                   "valid_pose_ratio": valid_count / len(rows),
                   "lost_frames": sum(row["tracker_state"] == "lost" for row in rows),
                   "recovery_attempts": sum(bool(row["recovery_attempted"]) for row in rows),
                   "recovery_successes": sum(bool(row["recovery_success"]) for row in rows),
                   "first_successful_registration": first_registration,
                   "mean_inference_ms": float(np.mean(timings)),
                   "events": {status: sum(row["status"] == status for row in rows)
                              for status in sorted({row["status"] for row in rows})}}, stream, indent=2)
    with (args.output / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    if args.save_reproducibility_manifest:
        write_json(
            args.output / "reproducibility_manifest.json",
            reproducibility_manifest(args.config, args.input, estimator.dtype_diagnostics),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
