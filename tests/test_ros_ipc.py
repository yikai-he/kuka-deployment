import ast
import importlib.util
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kuka_deployment.perception.interfaces import RGBDFrame
from kuka_deployment.perception.ros_ipc import (
    LatestFrameServer,
    UnixSocketCameraBackend,
    decode_frame,
    encode_frame,
)


ROOT = Path(__file__).resolve().parents[1]


def frame(number=3):
    return RGBDFrame(
        rgb=np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        depth_m=np.array([[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]], np.float32),
        K=np.array([[100, 0, 1], [0, 101, 0.5], [0, 0, 1]], np.float32),
        frame_number=number,
        timestamp_ns=123456,
        frame_id="cam_1_color_optical_frame",
        acquisition_monotonic_ns=time.monotonic_ns(),
    )


def test_wire_round_trip_preserves_rgbd_contract():
    sequence, restored = decode_frame(encode_frame(frame(), 7))
    assert sequence == 7
    np.testing.assert_array_equal(restored.rgb, frame().rgb)
    np.testing.assert_array_equal(restored.depth_m, frame().depth_m)
    np.testing.assert_array_equal(restored.K, frame().K)
    assert restored.frame_id == "cam_1_color_optical_frame"
    assert restored.timestamp_ns == 123456


def test_decoder_rejects_truncated_packet():
    with pytest.raises(ValueError, match="truncated"):
        decode_frame(b"bad")


def test_latest_server_replaces_unconsumed_frame():
    server = LatestFrameServer("/unused/test.sock", request_timeout_s=0.02)
    server.publish(frame(1))
    server.publish(frame(2))
    sequence, latest = decode_frame(server._latest)
    assert sequence == 1 and latest.frame_number == 2
    assert server.published_frames == 2 and server.replaced_frames == 1


class ScriptedSocket:
    def __init__(self, packets):
        self.packets = list(packets)
        self.response = bytearray()

    def settimeout(self, _timeout):
        pass

    def sendall(self, _request):
        packet = self.packets.pop(0)
        self.response.extend(struct.pack("!Q", len(packet)) + packet)

    def recv(self, size):
        chunk = self.response[:size]
        del self.response[:size]
        return bytes(chunk)

    def shutdown(self, _how):
        pass

    def close(self):
        pass


def test_ipc_backend_decodes_increasing_frames_and_empty_response():
    backend = UnixSocketCameraBackend({})
    backend._socket = ScriptedSocket([encode_frame(frame(2), 1), b"", encode_frame(frame(4), 2)])
    assert backend.read(0.1).frame_number == 2
    assert backend.read(0.1) is None
    assert backend.read(0.1).frame_number == 4


def _bridge_module():
    path = ROOT / "scripts" / "ros_rgbd_bridge.py"
    specification = importlib.util.spec_from_file_location("ros_rgbd_bridge_test", path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


class ArrayBridge:
    def imgmsg_to_cv2(self, message, desired_encoding):
        assert desired_encoding == "passthrough"
        return message.array


def message(array, encoding, frame_id="cam_1_color_optical_frame", stamp_ns=20):
    return SimpleNamespace(
        array=array,
        encoding=encoding,
        header=SimpleNamespace(
            frame_id=frame_id,
            stamp=SimpleNamespace(secs=stamp_ns // 1_000_000_000, nsecs=stamp_ns % 1_000_000_000),
        ),
    )


def options():
    return SimpleNamespace(depth_scale=0.001, min_depth_m=0.1, max_depth_m=3.0)


def test_ros_like_conversion_handles_bgr_uint16_scale_and_invalid_depth():
    module = _bridge_module()
    bgr = np.array([[[1, 2, 3], [4, 5, 6]]], np.uint8)
    raw_depth = np.array([[0, 1500]], np.uint16)
    rgb_msg = message(bgr, "bgr8")
    depth_msg = message(raw_depth, "16UC1")
    info = message(None, None)
    info.width, info.height = 2, 1
    info.K = [100, 0, 0.5, 0, 100, 0.5, 0, 0, 1]
    converted = module.convert_ros_messages(
        rgb_msg, depth_msg, info, ArrayBridge(), options(), 8
    )
    np.testing.assert_array_equal(converted.rgb, [[[3, 2, 1], [6, 5, 4]]])
    np.testing.assert_allclose(converted.depth_m, [[0.0, 1.5]])
    assert converted.frame_number == 8
    assert converted.timestamp_ns == 20


def test_ros_like_conversion_rejects_unaligned_or_nonoptical_frames():
    module = _bridge_module()
    rgb = message(np.zeros((2, 2, 3), np.uint8), "rgb8", "camera_frame")
    depth = message(np.ones((2, 2), np.uint16), "16UC1", "other_frame")
    info = message(None, None, "camera_frame")
    info.width, info.height = 2, 2
    info.K = [100, 0, 1, 0, 100, 1, 0, 0, 1]
    with pytest.raises(ValueError, match="frame_id"):
        module.convert_ros_messages(rgb, depth, info, ArrayBridge(), options(), 0)


def test_ros_bridge_and_ipc_modules_parse_as_python38():
    for path in (
        ROOT / "scripts" / "ros_rgbd_bridge.py",
        ROOT / "src" / "kuka_deployment" / "perception" / "ros_ipc.py",
    ):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
