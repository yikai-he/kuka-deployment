#!/usr/bin/env python3
"""Render qualitative CAD overlays from an existing offline replay result."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kuka_deployment.perception.visualization import box_corners, draw_cad_overlay


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="recorded episode")
    parser.add_argument("--results", required=True, type=Path, help="replay output")
    parser.add_argument("--mesh", required=True, type=Path, help="CAD mesh in metres")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--frames", default="", help="comma-separated source frame numbers to save"
    )
    parser.add_argument("--video", action="store_true", help="write overlay.mp4")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--use-distortion",
        action="store_true",
        help="apply camera.json D as k1,k2,p1,p2,k3; omit for rectified RGB",
    )
    parser.add_argument("--axis-length-m", type=float, default=0.03)
    return parser.parse_args(argv)


def parse_frame_numbers(value: str) -> set[int]:
    if not value.strip():
        return set()
    try:
        result = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError("--frames must be comma-separated integers") from exc
    if any(number < 0 for number in result):
        raise ValueError("frame numbers must be non-negative")
    return result


def _load_results(results: Path) -> list[tuple[int, np.ndarray]]:
    poses = np.load(results / "poses.npy")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses.npy must have shape (N,4,4)")
    with (results / "poses.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(poses):
        raise ValueError("poses.csv and poses.npy frame counts differ")
    numbers = [int(row["frame_number"]) for row in rows]
    if len(numbers) != len(set(numbers)):
        raise ValueError("poses.csv contains duplicate frame numbers")
    return list(zip(numbers, poses, strict=True))


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    selected = parse_frame_numbers(args.frames)
    if not selected and not args.video:
        raise ValueError("request --frames, --video, or both")
    if args.fps <= 0 or args.axis_length_m <= 0:
        raise ValueError("--fps and --axis-length-m must be positive")

    camera = json.loads((args.input / "camera.json").read_text(encoding="utf-8"))
    K = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    distortion = None
    if args.use_distortion:
        if "D" not in camera:
            raise ValueError("--use-distortion requested but camera.json has no D")
        distortion = np.asarray(camera["D"], dtype=np.float64)

    import cv2
    import trimesh

    mesh = trimesh.load(str(args.mesh), force="mesh")
    corners = box_corners(mesh.bounds)
    records = _load_results(args.results)
    available = {number for number, _pose in records}
    missing = sorted(selected - available)
    if missing:
        raise ValueError(f"requested frames are absent from results: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    frame_output = args.output / "frames"
    if selected:
        frame_output.mkdir(exist_ok=True)
    writer = None
    try:
        for frame_number, pose in records:
            rgb_path = args.input / "rgb" / f"{frame_number:06d}.png"
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"could not read RGB frame: {rgb_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            valid = np.isfinite(pose).all()
            overlay = (
                draw_cad_overlay(
                    rgb,
                    pose,
                    K,
                    corners,
                    distortion=distortion,
                    axis_length_m=args.axis_length_m,
                )
                if valid
                else rgb.copy()
            )
            label = f"frame={frame_number:06d} pose={'valid' if valid else 'invalid'}"
            bgr_overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr_overlay,
                label,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255) if valid else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            if frame_number in selected:
                path = frame_output / f"{frame_number:06d}.png"
                if not cv2.imwrite(str(path), bgr_overlay):
                    raise RuntimeError(f"could not write overlay frame: {path}")
            if args.video:
                if writer is None:
                    height, width = bgr_overlay.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.output / "overlay.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("could not open overlay video writer")
                writer.write(bgr_overlay)
    finally:
        if writer is not None:
            writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
