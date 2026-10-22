"""CPU-only regression scenarios for the legacy-v7 recovery behavior."""

import numpy as np

from kuka_deployment.perception.interfaces import RGBDFrame, TrackingStatus
from kuka_deployment.perception.segmenter import (
    RedCubeCandidateConfig, Segmentation, select_red_cube_candidate,
)
from kuka_deployment.perception.tracker import (
    DepthCheckStatus, PerceptionPipeline, PipelineConfig, PoseValidationConfig,
    RecoveryConfig, TrackerState, detect_tracking_failure,
)


def frame(number=0, *, depth=True, size=10, K=None):
    depth_m = np.ones((size, size), np.float32) if depth else np.zeros((size, size), np.float32)
    intrinsics = np.eye(3, dtype=np.float32) if K is None else K
    return RGBDFrame(np.zeros((size, size, 3), np.uint8), depth_m, intrinsics,
                     number, number, "camera")


def pose(x=0.0, z=1.0):
    result = np.eye(4, dtype=np.float32)
    result[:3, 3] = [x, 0, z]
    return result


class QueueSegmenter:
    def __init__(self, detections):
        self.detections = list(detections)
        self.contexts = []
        self.last_candidates = []

    def detect(self, _frame):
        return self.detections.pop(0) if self.detections else None

    def detect_with_context(self, frame, *, hint_uv=None, lost_streak=0):
        self.contexts.append((hint_uv, lost_streak))
        return self.detect(frame)


class QueueEstimator:
    def __init__(self, registrations=None, tracks=None):
        self.registrations = list(registrations or [])
        self.tracks = list(tracks or [])
        self.register_calls = 0
        self.track_calls = 0

    def register(self, _frame, _mask):
        self.register_calls += 1
        value = self.registrations.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def track(self, _frame):
        self.track_calls += 1
        value = self.tracks.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


MASK = np.ones((10, 10), bool)
DETECTION = Segmentation(MASK, 0.8)


def pipeline(detections, registrations, tracks=(), *, pose_config=None,
             recovery=None, initial_mask=None):
    detector = QueueSegmenter(detections)
    estimator = QueueEstimator(registrations, tracks)
    value = PerceptionPipeline(
        "cube", detector, estimator,
        PipelineConfig(pose=pose_config or PoseValidationConfig(),
                       recovery=recovery or RecoveryConfig()),
        initial_mask=initial_mask,
    )
    return value, detector, estimator


def test_01_initial_yolo_registration_succeeds():
    tracker, _, estimator = pipeline([DETECTION], [pose()])
    result = tracker.process_frame(frame())
    assert result.valid and result.status is TrackingStatus.REGISTERED
    assert estimator.register_calls == 1 and estimator.track_calls == 0


def test_02_initial_registration_failure_enters_lost():
    tracker, _, estimator = pipeline([DETECTION], [RuntimeError("register failed")])
    result = tracker.process_frame(frame())
    assert not result.valid and tracker.state is TrackerState.LOST
    assert estimator.track_calls == 0


def test_03_initial_mask_fallback_succeeds():
    # Match v7: a YOLO candidate may exist but its registration can fail, after
    # which the explicitly enabled recorded mask is tried once.
    tracker, _, estimator = pipeline([DETECTION], [RuntimeError("YOLO register failed"), pose()],
                                     initial_mask=MASK)
    result = tracker.process_frame(frame())
    assert result.valid and result.status is TrackingStatus.REGISTERED
    assert tracker.last_diagnostics["initial_mask_fallback"] is True
    assert estimator.register_calls == 2


def test_04_normal_tracking_succeeds():
    tracker, _, estimator = pipeline([DETECTION], [pose()], [pose(0.01)])
    tracker.process_frame(frame(0))
    result = tracker.process_frame(frame(1))
    assert result.valid and result.status is TrackingStatus.TRACKED
    assert estimator.track_calls == 1


def test_05_translation_jump_is_detected():
    config = PoseValidationConfig(max_translation_step_m=0.02, center_depth_residual_m=0)
    details = {}
    failed, reason = detect_tracking_failure(pose(0.1), pose(), config,
                                             frame=frame(), diagnostics=details)
    assert failed and "translation jump" in reason
    assert details["step_translation_m"] > 0.02


def test_06_missing_center_depth_is_explicit_and_policy_controlled():
    fail_details, allow_details = {}, {}
    fail_config = PoseValidationConfig(center_depth_residual_m=0.06, depth_unavailable_policy="fail")
    allow_config = PoseValidationConfig(center_depth_residual_m=0.06, depth_unavailable_policy="allow")
    failed, _ = detect_tracking_failure(pose(), pose(), fail_config,
                                        frame=frame(depth=False), diagnostics=fail_details)
    allowed, _ = detect_tracking_failure(pose(), pose(), allow_config,
                                         frame=frame(depth=False), diagnostics=allow_details)
    assert failed and not allowed
    assert fail_details["depth_check_status"] == DepthCheckStatus.UNAVAILABLE.value
    assert allow_details["depth_check_status"] == DepthCheckStatus.UNAVAILABLE.value


def test_07_structurally_invalid_se3_is_rejected():
    bad = pose(); bad[0, 0] = 2
    tracker, _, _ = pipeline([DETECTION], [bad])
    result = tracker.process_frame(frame())
    assert not result.valid and result.status is TrackingStatus.REGISTRATION_FAILED


def test_08_multiple_candidates_use_v7_score_not_highest_confidence():
    K = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]], np.float32)
    item = frame(size=30, K=K)
    masks = np.zeros((3, 30, 30), np.float32)
    masks[0, 2:10, 2:10] = 1       # 8 cm, confidence 0.99
    masks[1, 15:19, 15:19] = 1     # target 4 cm, confidence 0.80
    masks[2, 20:24, 2:6] = 1       # wrong class
    selected, candidates = select_red_cube_candidate(
        masks, np.array([0.99, 0.80, 1.0]), np.array([0, 0, 4]), item,
        RedCubeCandidateConfig(min_component_area=1),
    )
    assert selected is not None and selected.metadata["prediction_index"] == 1
    assert len(candidates) == 2


def test_09_recovery_succeeds_after_tracking_failure():
    config = PoseValidationConfig(max_translation_step_m=0.02, center_depth_residual_m=0)
    tracker, _, estimator = pipeline([DETECTION, DETECTION], [pose(), pose(0.01)],
                                     [pose(0.2)], pose_config=config)
    tracker.process_frame(frame(0))
    recovered = tracker.process_frame(frame(1))
    assert recovered.valid and recovered.status is TrackingStatus.REREGISTERED
    assert tracker.last_diagnostics["recovery_attempted"] is True
    assert tracker.last_diagnostics["recovery_success"] is True
    assert estimator.register_calls == 2


def test_10_failed_recovery_enters_lost():
    config = PoseValidationConfig(max_translation_step_m=0.02, center_depth_residual_m=0)
    tracker, _, _ = pipeline([DETECTION, None], [pose()], [pose(0.2)], pose_config=config)
    tracker.process_frame(frame(0))
    result = tracker.process_frame(frame(1))
    assert not result.valid and tracker.state is TrackerState.LOST


def test_11_lost_state_recovers_on_later_frame_without_tracking_first():
    tracker, _, estimator = pipeline([None, DETECTION], [pose()])
    assert not tracker.process_frame(frame(0)).valid
    result = tracker.process_frame(frame(1))
    assert result.valid and result.status is TrackingStatus.REREGISTERED
    assert estimator.track_calls == 0


def test_12_recovery_disabled_never_tracks_before_registration():
    recovery = RecoveryConfig(enabled=False)
    tracker, _, estimator = pipeline([DETECTION], [RuntimeError("no registration")],
                                     [pose()], recovery=recovery)
    tracker.process_frame(frame(0))
    second = tracker.process_frame(frame(1))
    assert not second.valid and tracker.state is TrackerState.LOST
    assert estimator.track_calls == 0


def test_13_stale_pose_is_never_marked_valid():
    config = PoseValidationConfig(max_translation_step_m=0.02, center_depth_residual_m=0)
    tracker, _, _ = pipeline([DETECTION, None], [pose()], [pose(0.2)], pose_config=config)
    accepted = tracker.process_frame(frame(0))
    failed = tracker.process_frame(frame(1))
    assert accepted.valid
    assert not failed.valid and failed.T_frame_object is None


def test_14_recovery_cooldown_suppresses_only_translation_jump():
    config = PoseValidationConfig(max_translation_step_m=0.02, center_depth_residual_m=0)
    recovery = RecoveryConfig(cooldown_frames=1)
    tracker, _, estimator = pipeline([DETECTION], [pose()], [pose(0.2)],
                                     pose_config=config, recovery=recovery)
    tracker.process_frame(frame(0))
    result = tracker.process_frame(frame(1))
    assert result.status is TrackingStatus.TRACKED and estimator.register_calls == 1
    assert tracker.cooldown_remaining == 0


def test_15_lost_hint_expands_then_falls_back_without_hint():
    K = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]], np.float32)
    item = frame(size=200, K=K)
    masks = np.zeros((1, 200, 200), np.float32)
    masks[0, 100:104, 100:104] = 1
    config = RedCubeCandidateConfig(min_component_area=1, recovery_hint_max_px=10,
                                    lost_hint_expand_px=5, hint_fallback_lost_streak=3)
    selected_1, candidates_1 = select_red_cube_candidate(
        masks, np.array([0.9]), np.array([0]), item, config, hint_uv=(0, 0), lost_streak=1)
    selected_3, _ = select_red_cube_candidate(
        masks, np.array([0.9]), np.array([0]), item, config, hint_uv=(0, 0), lost_streak=3)
    assert selected_1 is None and candidates_1[0]["allowed_hint_px"] == 15
    assert selected_3 is not None and selected_3.metadata["fallback_without_hint"] is True


def test_16_recovery_disabled_distinguishes_failure_from_lost_skip():
    config = PoseValidationConfig(max_translation_step_m=0.02, center_depth_residual_m=0)
    tracker, _, estimator = pipeline(
        [DETECTION], [pose()], [pose(0.0228023)], pose_config=config,
        recovery=RecoveryConfig(enabled=False),
    )
    tracker.process_frame(frame(0))

    failed = tracker.process_frame(frame(328))
    assert not failed.valid and failed.status is TrackingStatus.TRACKING_FAILED
    assert failed.failure_reason == "translation jump exceeded configured threshold"
    assert tracker.last_diagnostics["failure_stage"] == "tracking_validation"
    assert tracker.last_diagnostics["step_translation_m"] == np.float32(0.0228023)
    assert tracker.state is TrackerState.LOST
    assert estimator.track_calls == 1

    skipped = tracker.process_frame(frame(329))
    assert not skipped.valid and skipped.status is TrackingStatus.TRACKING_FAILED
    assert skipped.failure_reason == "recovery disabled while tracker is LOST"
    assert tracker.last_diagnostics["failure_stage"] == "lost_recovery_disabled"
    assert tracker.last_diagnostics["recovery_attempted"] is False
    assert tracker.lost_streak == 2
    assert estimator.track_calls == 1
