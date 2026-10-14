import json
from pathlib import Path

import numpy as np
import pytest

from kuka_deployment.perception import sources
from kuka_deployment.perception.sources import RecordedSequenceSource, VideoDepthSource


class FakeCapture:
    def __init__(self, _path):
        self.index = 0
    def isOpened(self): return True
    def read(self):
        image = np.full((2, 2, 3), self.index, np.uint8)
        self.index += 1
        return True, image
    def get(self, _prop): return self.index * 100.0
    def release(self): pass


class FakeCV2:
    IMREAD_COLOR = 1
    IMREAD_UNCHANGED = -1
    CAP_PROP_POS_MSEC = 0
    VideoCapture = FakeCapture
    @staticmethod
    def imread(path, mode):
        return np.full((2, 2, 3), 7, np.uint8) if mode == 1 else np.full((2, 2), 1000, np.uint16)


def make_sequence(tmp_path: Path):
    (tmp_path / "rgb").mkdir(); (tmp_path / "depth").mkdir()
    (tmp_path / "camera.json").write_text(json.dumps({
        "K": [[100, 0, 1], [0, 100, 1], [0, 0, 1]], "depth_scale": 0.001,
        "frame_id": "camera", "timestamps_ns": {"000000": 123},
    }))
    (tmp_path / "rgb/000000.png").touch(); (tmp_path / "depth/000000.png").touch()


def test_recorded_source_pairs_ids_and_preserves_metadata(tmp_path, monkeypatch):
    make_sequence(tmp_path)
    monkeypatch.setattr(sources, "_cv2", lambda: FakeCV2)
    frames = list(RecordedSequenceSource(tmp_path))
    assert len(frames) == 1 and frames[0].timestamp_ns == 123
    assert frames[0].frame_id == "camera" and frames[0].depth_m[0, 0] == pytest.approx(1)
    np.testing.assert_array_equal(frames[0].rgb[0, 0], [7, 7, 7])


def test_recorded_source_rejects_missing_pair(tmp_path):
    make_sequence(tmp_path)
    (tmp_path / "rgb/000001.png").touch()
    with pytest.raises(ValueError, match="unpaired"):
        RecordedSequenceSource(tmp_path)


def test_video_depth_uses_explicit_mapping_and_separates_presentation_time(tmp_path, monkeypatch):
    camera = tmp_path / "camera.json"
    camera.write_text(json.dumps({"K": np.eye(3).tolist(), "depth_scale": 0.001}))
    depth = tmp_path / "depth"; depth.mkdir(); (depth / "d.png").touch()
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps([{"video_frame": 2, "depth": "d.png", "timestamp_ns": None}]))
    video = tmp_path / "input.mp4"; video.touch()
    monkeypatch.setattr(sources, "_cv2", lambda: FakeCV2)
    result = list(VideoDepthSource(video, depth, camera, mapping))[0]
    assert result.frame_number == 2 and result.timestamp_ns is None
    assert result.presentation_timestamp_ns == 300_000_000


def test_video_depth_rejects_missing_depth(tmp_path):
    camera = tmp_path / "camera.json"
    camera.write_text(json.dumps({"K": np.eye(3).tolist(), "depth_scale": 0.001}))
    depth = tmp_path / "depth"; depth.mkdir()
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps([{"video_frame": 0, "depth": "missing.png"}]))
    with pytest.raises(FileNotFoundError):
        VideoDepthSource(tmp_path / "x.mp4", depth, camera, mapping)
