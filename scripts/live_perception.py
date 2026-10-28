#!/usr/bin/env python3
"""Run YOLO-Seg + FoundationPose on a pluggable live RGB-D camera backend."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_deployment.perception.interfaces import RGBDFrame
from kuka_deployment.perception.runtime import (
    build_pipeline,
    process_frame_timed,
    result_record,
    synchronize_cuda_if_available,
)
from kuka_deployment.perception.sources import CameraBackend, LiveFrameSource


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--backend",
        required=True,
        metavar="MODULE:FACTORY",
        help="lazy camera factory accepting the live.camera config mapping",
    )
    parser.add_argument("--backend-config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--save-visualization", action="store_true")
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args(argv)


def load_backend(
    specification: str, options: Mapping[str, Any]
) -> CameraBackend:
    """Load an optional camera plugin only after command-line parsing."""
    module_name, separator, factory_name = specification.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("backend must use MODULE:FACTORY syntax")
    module = importlib.import_module(module_name)
    factory: Callable[[Mapping[str, Any]], CameraBackend] = getattr(
        module, factory_name
    )
    backend = factory(options)
    for method in ("start", "read", "stop"):
        if not callable(getattr(backend, method, None)):
            raise TypeError(f"camera backend has no callable {method}()")
    return backend


def _backend_options(config: Mapping[str, Any], path: Path | None) -> dict[str, Any]:
    options = dict(config.get("live", {}).get("camera", {}))
    if path is not None:
        with path.open(encoding="utf-8") as stream:
            override = yaml.safe_load(stream) or {}
        if not isinstance(override, dict):
            raise ValueError("backend config must contain a YAML mapping")
        options.update(override)
    return options


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    args.output.mkdir(parents=True, exist_ok=True)
    backend = load_backend(args.backend, _backend_options(config, args.backend_config))
    source_options = config.get("live", {}).get("source", {})
    source = LiveFrameSource(
        backend,
        read_timeout_s=float(source_options.get("read_timeout_s", 0.25)),
        retry_delay_s=float(source_options.get("retry_delay_s", 0.01)),
        startup_timeout_s=float(source_options.get("startup_timeout_s", 5.0)),
        shutdown_timeout_s=float(source_options.get("shutdown_timeout_s", 2.0)),
    )
    pipeline = build_pipeline(config, args.output)
    result_path = args.output / "live_results.jsonl"
    counts: Counter[str] = Counter()
    inference_times: list[float] = []
    end_to_end_times: list[float] = []
    writer = None
    interrupted = False
    fatal_error = None
    if args.save_visualization:
        import cv2
    try:
        with result_path.open("w", encoding="utf-8") as output:
            for frame in source:
                result, inference_ms, end_to_end_ms = process_frame_timed(
                    pipeline,
                    frame,
                    synchronize=synchronize_cuda_if_available,
                )
                record = result_record(
                    frame,
                    result,
                    pipeline,
                    inference_ms,
                    end_to_end_ms=end_to_end_ms,
                )
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
                counts[result.status.value] += 1
                counts["valid"] += int(result.valid)
                counts["recovery_attempts"] += int(record["recovery_attempted"])
                counts["recovery_successes"] += int(record["recovery_success"])
                inference_times.append(inference_ms)
                if end_to_end_ms is not None:
                    end_to_end_times.append(end_to_end_ms)
                print(
                    f"frame={frame.frame_number} status={result.status.value} "
                    f"valid={result.valid} inference_ms={inference_ms:.1f}"
                )
                if args.save_visualization:
                    image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                    color = (0, 180, 0) if result.valid else (0, 0, 255)
                    cv2.putText(
                        image,
                        result.status.value,
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    if writer is None:
                        height, width = image.shape[:2]
                        writer = cv2.VideoWriter(
                            str(args.output / "live_tracking.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            10.0,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError("could not open visualization video writer")
                    writer.write(image)
                if args.max_frames is not None and len(inference_times) >= args.max_frames:
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; closing camera", file=sys.stderr)
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        print(f"live perception failed: {fatal_error}", file=sys.stderr)
    finally:
        source.close()
        if writer is not None:
            writer.release()

    processed = len(inference_times)
    metrics = {
        "processed_frames": processed,
        "valid_frames": counts["valid"],
        "invalid_frames": processed - counts["valid"],
        "dropped_camera_frames": source.dropped_frames,
        "recovery_attempts": counts["recovery_attempts"],
        "recovery_successes": counts["recovery_successes"],
        "mean_inference_ms": (
            None if not inference_times else float(np.mean(inference_times))
        ),
        "mean_end_to_end_ms": (
            None if not end_to_end_times else float(np.mean(end_to_end_times))
        ),
        "events": {
            key: value
            for key, value in sorted(counts.items())
            if key not in {"valid", "recovery_attempts", "recovery_successes"}
        },
        "interrupted": interrupted,
        "fatal_error": fatal_error,
    }
    with (args.output / "live_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    with (args.output / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return 1 if fatal_error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
