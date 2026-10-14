"""Offline RGB-D sources producing the same :class:`RGBDFrame` contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from .interfaces import RGBDFrame
from .rgbd import bgr_to_rgb, depth_to_metres


class RGBDSource(Protocol):
    def __iter__(self) -> Iterator[RGBDFrame]: ...


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("OpenCV is required to read image/video sources") from exc
    return cv2


def _load_camera(path: Path) -> tuple[np.ndarray, float, str, dict[str, int]]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if "K" not in data:
        raise ValueError(f"{path} has no K")
    K = np.asarray(data["K"], dtype=np.float32).reshape(3, 3)
    scale = float(data.get("depth_scale", 0.001))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("camera depth_scale must be finite and positive")
    frame_id = str(data.get("frame_id", "camera_color_optical_frame"))
    raw_timestamps = data.get("timestamps_ns", {})
    if isinstance(raw_timestamps, list):
        timestamps = {str(i): int(value) for i, value in enumerate(raw_timestamps)}
    elif isinstance(raw_timestamps, dict):
        timestamps = {str(key): int(value) for key, value in raw_timestamps.items()}
    else:
        raise ValueError("timestamps_ns must be a list or object when supplied")
    return K, scale, frame_id, timestamps


def _numeric_pngs(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {path.stem: path for path in directory.glob("*.png") if path.stem.isdigit()}
    if not result:
        raise ValueError(f"no numeric PNG frames in {directory}")
    return result


class RecordedSequenceSource:
    """Read ``rgb/<id>.png``, ``depth/<id>.png`` and ``camera.json``.

    RGB/depth ID sets must match exactly: missing frames are errors, not omissions.
    """

    def __init__(self, root: str | Path, *, min_depth_m: float = 0.001,
                 max_depth_m: float = 3.0) -> None:
        self.root = Path(root)
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.K, self.scale, self.frame_id, self.timestamps = _load_camera(self.root / "camera.json")
        self.rgb_paths = _numeric_pngs(self.root / "rgb")
        self.depth_paths = _numeric_pngs(self.root / "depth")
        if self.rgb_paths.keys() != self.depth_paths.keys():
            missing_depth = sorted(self.rgb_paths.keys() - self.depth_paths.keys())
            missing_rgb = sorted(self.depth_paths.keys() - self.rgb_paths.keys())
            raise ValueError(f"unpaired frames: missing_depth={missing_depth}, missing_rgb={missing_rgb}")
        self.ids = sorted(self.rgb_paths, key=int)

    def __iter__(self) -> Iterator[RGBDFrame]:
        cv2 = _cv2()
        for sequence_number, identifier in enumerate(self.ids):
            bgr = cv2.imread(str(self.rgb_paths[identifier]), cv2.IMREAD_COLOR)
            raw_depth = cv2.imread(str(self.depth_paths[identifier]), cv2.IMREAD_UNCHANGED)
            if bgr is None or raw_depth is None:
                raise ValueError(f"could not decode frame {identifier}")
            yield RGBDFrame(
                rgb=bgr_to_rgb(bgr),
                depth_m=depth_to_metres(raw_depth, self.scale, min_depth_m=self.min_depth_m,
                                        max_depth_m=self.max_depth_m),
                K=self.K.copy(), frame_number=int(identifier),
                timestamp_ns=self.timestamps.get(identifier), frame_id=self.frame_id,
            )


class VideoDepthSource:
    """Read RGB video using an explicit JSON frame-to-depth correspondence.

    Manifest format: ``[{"video_frame": 0, "depth": "000010.png",
    "timestamp_ns": null}, ...]``. Video presentation times are recorded separately.
    """

    def __init__(self, video: str | Path, depth_dir: str | Path, camera_json: str | Path,
                 correspondence_json: str | Path, *, min_depth_m: float = 0.001,
                 max_depth_m: float = 3.0) -> None:
        self.video, self.depth_dir = Path(video), Path(depth_dir)
        self.K, self.scale, self.frame_id, _ = _load_camera(Path(camera_json))
        self.min_depth_m, self.max_depth_m = min_depth_m, max_depth_m
        with Path(correspondence_json).open(encoding="utf-8") as stream:
            self.mapping = json.load(stream)
        if not isinstance(self.mapping, list) or not self.mapping:
            raise ValueError("correspondence must be a non-empty JSON list")
        video_ids = [entry.get("video_frame") for entry in self.mapping]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in video_ids):
            raise ValueError("every correspondence needs a non-negative integer video_frame")
        if video_ids != sorted(video_ids) or len(video_ids) != len(set(video_ids)):
            raise ValueError("video_frame values must be unique and ascending")
        for entry in self.mapping:
            if not isinstance(entry.get("depth"), str):
                raise ValueError("every correspondence needs an explicit depth filename")
            if not (self.depth_dir / entry["depth"]).is_file():
                raise FileNotFoundError(self.depth_dir / entry["depth"])

    def __iter__(self) -> Iterator[RGBDFrame]:
        cv2 = _cv2()
        capture = cv2.VideoCapture(str(self.video))
        if not capture.isOpened():
            raise FileNotFoundError(f"could not open video {self.video}")
        try:
            current = -1
            for entry in self.mapping:
                target = entry["video_frame"]
                bgr = None
                while current < target:
                    ok, bgr = capture.read()
                    current += 1
                    if not ok:
                        raise ValueError(f"video ended before mapped frame {target}")
                raw = cv2.imread(str(self.depth_dir / entry["depth"]), cv2.IMREAD_UNCHANGED)
                if raw is None:
                    raise ValueError(f"could not decode depth {entry['depth']}")
                timestamp = entry.get("timestamp_ns")
                presentation_ns = round(float(capture.get(cv2.CAP_PROP_POS_MSEC)) * 1_000_000)
                yield RGBDFrame(
                    rgb=bgr_to_rgb(bgr),
                    depth_m=depth_to_metres(raw, self.scale, min_depth_m=self.min_depth_m,
                                            max_depth_m=self.max_depth_m),
                    K=self.K.copy(), frame_number=target,
                    timestamp_ns=None if timestamp is None else int(timestamp),
                    presentation_timestamp_ns=presentation_ns, frame_id=self.frame_id,
                )
        finally:
            capture.release()


class RGBOnlyVideoSource:
    def __init__(self, *_args, **_kwargs) -> None:
        raise ValueError("RGB-only video has no metric depth and cannot drive FoundationPose")
