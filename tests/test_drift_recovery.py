"""CPU-only tests for the opt-in image-plane drift recovery extension."""

import numpy as np

from kuka_deployment.perception.interfaces import RGBDFrame, TrackingStatus
from kuka_deployment.perception.segmenter import Segmentation
from kuka_deployment.perception.tracker import (
    DriftCheckConfig,
    PerceptionPipeline,
    PipelineConfig,
    PoseValidationConfig,
    RecoveryConfig,
    TrackerState,
    project_object_point_to_pixel,
)


def make_frame(number: int) -> RGBDFrame:
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], np.float32)
    return RGBDFrame(
        np.zeros((100, 100, 3), np.uint8),
        np.ones((100, 100), np.float32),
        K,
        number,
        number,
        "camera",
    )


def pose() -> np.ndarray:
    value = np.eye(4, dtype=np.float32)
    value[2, 3] = 1.0
    return value


def detection(center_u: int, confidence: float = 0.9) -> Segmentation:
    mask = np.zeros((100, 100), bool)
    mask[50, center_u] = True
    return Segmentation(mask, confidence, {"v7_candidate_validated": True})


class Segmenter:
    def __init__(self, values):
        self.values = list(values)
        self.last_candidates = []
        self.hints = []

    def detect_with_context(self, _frame, *, hint_uv=None, lost_streak=0):
        self.hints.append((hint_uv, lost_streak))
        return self.values.pop(0)


class Estimator:
    def __init__(self, registrations, tracks):
        self.registrations = list(registrations)
        self.tracks = list(tracks)
        self.register_calls = 0

    def register(self, _frame, _mask):
        self.register_calls += 1
        return self.registrations.pop(0)

    def track(self, _frame):
        return self.tracks.pop(0)


def pipeline(detections, tracks, *, threshold=10.0, consecutive=2, recovery=True):
    segmenter = Segmenter(detections)
    estimator = Estimator([pose(), pose()], tracks)
    config = PipelineConfig(
        pose=PoseValidationConfig(
            max_translation_step_m=0.25, center_depth_residual_m=0
        ),
        recovery=RecoveryConfig(enabled=recovery),
        drift_check=DriftCheckConfig(
            enabled=True,
            interval_frames=1,
            max_center_error_px=threshold,
            min_detection_confidence=0.5,
            consecutive_failures=consecutive,
        ),
    )
    return PerceptionPipeline("cube", segmenter, estimator, config), estimator


def test_projects_native_cad_center_not_pose_origin():
    value = pose()
    value[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    pixel = project_object_point_to_pixel(
        np.array([0.1, 0.0, 0.0]), value, make_frame(0).K
    )
    np.testing.assert_allclose(pixel, (50.0, 60.0))


def test_pixel_threshold_is_strict_and_boundary_does_not_fail():
    tracker, estimator = pipeline(
        [detection(50), detection(60)], [pose()], threshold=10, consecutive=1
    )
    tracker.process_frame(make_frame(0))
    result = tracker.process_frame(make_frame(1))
    assert result.status is TrackingStatus.TRACKED
    assert tracker.last_diagnostics["drift_pixel_error"] == 10.0
    assert estimator.register_calls == 1


def test_consecutive_failures_trigger_same_frame_mask_recovery():
    tracker, estimator = pipeline(
        [detection(50), detection(70), detection(70)], [pose(), pose()], consecutive=2
    )
    tracker.process_frame(make_frame(0))
    first = tracker.process_frame(make_frame(1))
    second = tracker.process_frame(make_frame(2))
    assert first.status is TrackingStatus.TRACKED
    assert second.valid and second.status is TrackingStatus.REREGISTERED
    assert estimator.register_calls == 2
    assert tracker.last_diagnostics["failure_stage"] is None
    assert tracker.last_diagnostics["recovery_attempted"] is True
    assert tracker.last_diagnostics["drift_recovery_outcome"] == "succeeded"


def test_inconclusive_detection_breaks_hysteresis_without_false_positive():
    tracker, estimator = pipeline(
        [detection(50), detection(70), None, detection(70)],
        [pose(), pose(), pose()],
        consecutive=2,
    )
    tracker.process_frame(make_frame(0))
    tracker.process_frame(make_frame(1))
    middle = tracker.process_frame(make_frame(2))
    last = tracker.process_frame(make_frame(3))
    assert middle.valid and middle.status is TrackingStatus.TRACKED
    assert last.valid and last.status is TrackingStatus.TRACKED
    assert tracker.last_diagnostics["drift_failure_streak"] == 1
    assert estimator.register_calls == 1


def test_low_confidence_detection_is_inconclusive():
    tracker, estimator = pipeline(
        [detection(50), detection(90, 0.49)], [pose()], consecutive=1
    )
    tracker.process_frame(make_frame(0))
    result = tracker.process_frame(make_frame(1))
    assert result.valid and result.status is TrackingStatus.TRACKED
    assert tracker.last_diagnostics["drift_pixel_error"] is None
    assert (
        tracker.last_diagnostics["drift_reason"]
        == "detection confidence below threshold"
    )
    assert estimator.register_calls == 1


def test_check_interval_skips_yolo_between_scheduled_frames():
    tracker, estimator = pipeline(
        [detection(50), detection(90)], [pose(), pose()], consecutive=1
    )
    tracker.config = PipelineConfig(
        pose=tracker.config.pose,
        recovery=tracker.config.recovery,
        drift_check=DriftCheckConfig(
            enabled=True,
            interval_frames=2,
            max_center_error_px=10,
            consecutive_failures=1,
        ),
    )
    tracker.process_frame(make_frame(0))
    skipped = tracker.process_frame(make_frame(1))
    recovered = tracker.process_frame(make_frame(2))
    assert skipped.status is TrackingStatus.TRACKED
    assert recovered.status is TrackingStatus.REREGISTERED
    assert estimator.register_calls == 2


def test_confirmed_drift_with_recovery_disabled_never_emits_rejected_pose():
    tracker, _ = pipeline(
        [detection(50), detection(70)], [pose()], consecutive=1, recovery=False
    )
    tracker.process_frame(make_frame(0))
    result = tracker.process_frame(make_frame(1))
    assert not result.valid and result.T_frame_object is None
    assert result.status is TrackingStatus.TRACKING_FAILED
    assert tracker.state is TrackerState.LOST
    assert tracker.last_diagnostics["failure_stage"] == "drift_validation"
    assert tracker.last_diagnostics["drift_recovery_outcome"] == "disabled"


def test_failed_drift_recovery_does_not_emit_rejected_or_stale_pose():
    tracker, estimator = pipeline(
        [detection(50), detection(70)], [pose()], consecutive=1
    )
    invalid = pose()
    invalid[0, 0] = 2.0
    estimator.registrations[1] = invalid
    tracker.process_frame(make_frame(0))
    result = tracker.process_frame(make_frame(1))
    assert not result.valid and result.T_frame_object is None
    assert tracker.state is TrackerState.LOST
    assert tracker.last_diagnostics["drift_recovery_outcome"] == "failed"
