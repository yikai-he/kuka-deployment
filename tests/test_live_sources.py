import threading
import numpy as np
import pytest

from kuka_deployment.perception.interfaces import RGBDFrame
from kuka_deployment.perception.sources import LiveFrameSource


def frame(number: int, *, timestamp_ns: int | None = None) -> RGBDFrame:
    return RGBDFrame(
        rgb=np.zeros((2, 3, 3), np.uint8),
        depth_m=np.ones((2, 3), np.float32),
        K=np.eye(3, dtype=np.float32),
        frame_number=number,
        timestamp_ns=timestamp_ns,
        frame_id="camera_optical_frame",
    )


class SequenceBackend:
    def __init__(self, values):
        self.values = iter(values)
        self.started = False
        self.stop_calls = 0

    def start(self):
        self.started = True

    def read(self, _timeout_s):
        return next(self.values)

    def stop(self):
        self.stop_calls += 1


def test_live_source_keeps_latest_frame_and_preserves_capture_metadata():
    backend = SequenceBackend([frame(0, timestamp_ns=10), frame(1, timestamp_ns=20)])
    source = LiveFrameSource(backend)
    frames = list(source)
    assert backend.started
    assert frames[-1].frame_number == 1
    assert frames[-1].timestamp_ns == 20
    assert frames[-1].acquisition_monotonic_ns is not None
    assert source.dropped_frames + len(frames) == 2
    assert backend.stop_calls == 1


def test_live_source_retries_temporary_missing_frame_without_busy_loop():
    backend = SequenceBackend([None, frame(4), None])
    source = LiveFrameSource(backend, retry_delay_s=0.001)
    iterator = iter(source)
    assert next(iterator).frame_number == 4
    source.close()
    assert backend.stop_calls == 1


def test_live_source_propagates_invalid_camera_output_and_stops_backend():
    backend = SequenceBackend(["not a frame"])
    source = LiveFrameSource(backend)
    with pytest.raises(RuntimeError, match="camera acquisition failed"):
        list(source)
    assert backend.stop_calls == 1


class BlockingBackend:
    def __init__(self):
        self.stopped = threading.Event()

    def start(self):
        pass

    def read(self, _timeout_s):
        self.stopped.wait()
        raise StopIteration

    def stop(self):
        self.stopped.set()


def test_live_source_close_unblocks_acquisition_and_is_idempotent():
    source = LiveFrameSource(BlockingBackend())
    source.start()
    source.close()
    source.close()


def test_live_source_validates_timeouts():
    with pytest.raises(ValueError, match="read_timeout_s"):
        LiveFrameSource(SequenceBackend([]), read_timeout_s=0)


class FailingStartBackend(SequenceBackend):
    def start(self):
        raise OSError("camera missing")


def test_live_source_reports_camera_initialization_failure():
    backend = FailingStartBackend([])
    source = LiveFrameSource(backend)
    with pytest.raises(RuntimeError, match="initialization failed"):
        source.start()
    assert backend.stop_calls == 1
