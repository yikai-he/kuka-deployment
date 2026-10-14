import numpy as np

from kuka_deployment.perception.interfaces import RGBDFrame, TrackingStatus
from kuka_deployment.perception.segmenter import Segmentation
from kuka_deployment.perception.tracker import PerceptionPipeline, TrackerState


def frame(number):
    return RGBDFrame(np.zeros((10, 10, 3), np.uint8), np.ones((10, 10), np.float32),
                     np.eye(3, dtype=np.float32), number, number, "camera")


def pose(x=0):
    value = np.eye(4, dtype=np.float32); value[:3, 3] = [x, 0, 1]
    return value


class Segmenter:
    def __init__(self): self.calls = 0; self.fail = False
    def detect(self, _frame):
        self.calls += 1
        return None if self.fail else Segmentation(np.ones((10, 10), bool), 0.8)


class Estimator:
    def __init__(self): self.register_calls = 0; self.tracks = []
    def register(self, _frame, _mask):
        self.register_calls += 1
        return pose(0.01 * self.register_calls)
    def track(self, _frame): return self.tracks.pop(0)


def test_initial_registration_then_tracking():
    detector, estimator = Segmenter(), Estimator()
    pipeline = PerceptionPipeline("cube", detector, estimator)
    first = pipeline.process_frame(frame(0))
    estimator.tracks.append(pose(0.02))
    second = pipeline.process_frame(frame(1))
    assert first.status is TrackingStatus.REGISTERED
    assert second.status is TrackingStatus.TRACKED
    assert pipeline.state is TrackerState.TRACKING and detector.calls == 1


def test_tracking_jump_triggers_redetection_and_reregistration():
    detector, estimator = Segmenter(), Estimator()
    pipeline = PerceptionPipeline("cube", detector, estimator)
    pipeline.process_frame(frame(0))
    estimator.tracks.append(pose(1.0))
    recovered = pipeline.process_frame(frame(1))
    assert recovered.status is TrackingStatus.REREGISTERED and recovered.valid
    assert detector.calls == 2 and estimator.register_calls == 2


def test_failed_recovery_returns_no_stale_pose_and_enters_lost():
    detector, estimator = Segmenter(), Estimator()
    pipeline = PerceptionPipeline("cube", detector, estimator)
    pipeline.process_frame(frame(0))
    detector.fail = True
    estimator.tracks.append(pose(1.0))
    failed = pipeline.process_frame(frame(1))
    assert failed.status is TrackingStatus.TRACKING_FAILED
    assert not failed.valid and failed.T_frame_object is None
    assert pipeline.state is TrackerState.LOST


def test_lost_pipeline_attempts_registration_again():
    detector, estimator = Segmenter(), Estimator()
    pipeline = PerceptionPipeline("cube", detector, estimator)
    detector.fail = True
    assert not pipeline.process_frame(frame(0)).valid
    detector.fail = False
    recovered = pipeline.process_frame(frame(1))
    assert recovered.status is TrackingStatus.REREGISTERED
