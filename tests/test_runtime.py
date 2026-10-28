import numpy as np

from kuka_deployment.perception.interfaces import (
    ObjectPoseEstimate,
    RGBDFrame,
    TrackingStatus,
)
from kuka_deployment.perception.runtime import process_frame_timed, result_record


def frame() -> RGBDFrame:
    return RGBDFrame(
        rgb=np.zeros((2, 2, 3), np.uint8),
        depth_m=np.ones((2, 2), np.float32),
        K=np.eye(3, dtype=np.float32),
        frame_number=8,
        timestamp_ns=123,
        frame_id="camera",
    )


class Pipeline:
    last_diagnostics = {
        "state": "lost",
        "recovery_attempted": True,
        "recovery_success": False,
    }

    def process_frame(self, value):
        return ObjectPoseEstimate(
            "cube",
            value.timestamp_ns,
            value.frame_id,
            None,
            False,
            TrackingStatus.TRACKING_FAILED,
            value.frame_number,
            failure_reason="recovery failed",
        )


def test_timed_processing_synchronizes_before_and_after_inference():
    calls = []
    result, elapsed_ms, end_to_end_ms = process_frame_timed(
        Pipeline(), frame(), synchronize=lambda: calls.append("sync")
    )
    assert result.timestamp_ns == 123
    assert elapsed_ms >= 0
    assert end_to_end_ms is None
    assert calls == ["sync", "sync"]


def test_result_record_declares_pose_convention_and_never_serializes_stale_pose():
    value = frame()
    pipeline = Pipeline()
    result = pipeline.process_frame(value)
    record = result_record(value, result, pipeline, 1.25)
    assert record["T_camera_object"] is None
    assert record["coordinate_convention"].startswith("T_camera_object")
    assert record["recovery_attempted"] is True
    assert record["valid"] is False
