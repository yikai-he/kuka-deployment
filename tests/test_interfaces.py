import numpy as np
import pytest

from kuka_deployment.perception.interfaces import ObjectPoseEstimate, RGBDFrame, TrackingStatus


def frame(**changes):
    values = dict(rgb=np.zeros((2, 3, 3), np.uint8), depth_m=np.ones((2, 3), np.float32),
                  K=np.eye(3, dtype=np.float32), frame_number=0, timestamp_ns=None, frame_id="camera")
    values.update(changes)
    return RGBDFrame(**values)


def test_valid_frame_allows_unavailable_capture_timestamp():
    assert frame().timestamp_ns is None


@pytest.mark.parametrize("changes", [
    {"rgb": np.zeros((2, 3, 3), np.float32)},
    {"depth_m": np.ones((2, 3), np.float64)},
    {"depth_m": np.ones((3, 2), np.float32)},
    {"K": np.diag([0.0, 1.0, 1.0]).astype(np.float32)},
    {"timestamp_ns": -1},
    {"acquisition_monotonic_ns": -1},
])
def test_invalid_frame_inputs_are_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        frame(**changes)


def test_invalid_pose_cannot_carry_stale_transform():
    with pytest.raises(ValueError):
        ObjectPoseEstimate("cube", None, "camera", np.eye(4), False,
                           TrackingStatus.TRACKING_FAILED, 2, failure_reason="lost")


def test_valid_pose_requires_se3():
    bad = np.eye(4); bad[0, 0] = 2
    with pytest.raises(ValueError):
        ObjectPoseEstimate("cube", 1, "camera", bad, True, TrackingStatus.TRACKED, 2)
