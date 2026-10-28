"""ROS-independent latest-frame IPC for separate ROS and inference Pythons.

This module deliberately uses Python 3.8 syntax so a ROS Noetic interpreter can
load it. The protocol is local Unix stream sockets, not pickle or ROS messages.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_LENGTH = struct.Struct("!Q")
_HEADER_LENGTH = struct.Struct("!I")
_REQUEST = struct.Struct("!qd")
_MAX_PACKET_BYTES = 256 * 1024 * 1024


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("IPC peer closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def encode_frame(frame: Any, sequence: int) -> bytes:
    """Encode one validated RGBDFrame without pickle or compression."""
    rgb = np.ascontiguousarray(frame.rgb, dtype=np.uint8)
    depth = np.ascontiguousarray(frame.depth_m, dtype=np.float32)
    header = {
        "schema_version": 1,
        "sequence": int(sequence),
        "frame_number": int(frame.frame_number),
        "timestamp_ns": frame.timestamp_ns,
        "frame_id": frame.frame_id,
        "acquisition_monotonic_ns": frame.acquisition_monotonic_ns,
        "height": int(rgb.shape[0]),
        "width": int(rgb.shape[1]),
        "K": np.asarray(frame.K, dtype=np.float32).reshape(-1).tolist(),
        "rgb_bytes": int(rgb.nbytes),
        "depth_bytes": int(depth.nbytes),
    }
    metadata = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _HEADER_LENGTH.pack(len(metadata)) + metadata + rgb.tobytes() + depth.tobytes()


def decode_frame(packet: bytes) -> Tuple[int, Any]:
    """Decode and validate one IPC packet as the common RGBDFrame contract."""
    if len(packet) < _HEADER_LENGTH.size:
        raise ValueError("truncated IPC frame header")
    header_size = _HEADER_LENGTH.unpack_from(packet)[0]
    metadata_end = _HEADER_LENGTH.size + header_size
    if metadata_end > len(packet):
        raise ValueError("truncated IPC metadata")
    try:
        header = json.loads(packet[_HEADER_LENGTH.size : metadata_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid IPC metadata") from exc
    if header.get("schema_version") != 1:
        raise ValueError("unsupported IPC schema version")
    height, width = int(header["height"]), int(header["width"])
    expected_rgb = height * width * 3
    expected_depth = height * width * 4
    if header.get("rgb_bytes") != expected_rgb or header.get("depth_bytes") != expected_depth:
        raise ValueError("IPC byte counts do not match declared dimensions")
    if len(packet) != metadata_end + expected_rgb + expected_depth:
        raise ValueError("IPC packet length does not match metadata")
    rgb_end = metadata_end + expected_rgb
    rgb = np.frombuffer(packet[metadata_end:rgb_end], np.uint8).reshape(height, width, 3)
    depth = np.frombuffer(packet[rgb_end:], "<f4").reshape(height, width)
    # Imported only by the Python 3.11 inference process. The ROS Python 3.8
    # bridge uses encode/server only and never parses the 3.10+ core package.
    from .interfaces import RGBDFrame

    frame = RGBDFrame(
        rgb=rgb.copy(),
        depth_m=depth.astype(np.float32, copy=True),
        K=np.asarray(header["K"], dtype=np.float32).reshape(3, 3),
        frame_number=int(header["frame_number"]),
        timestamp_ns=header.get("timestamp_ns"),
        frame_id=str(header["frame_id"]),
        acquisition_monotonic_ns=header.get("acquisition_monotonic_ns"),
    )
    return int(header["sequence"]), frame


class LatestFrameServer:
    """Serve only the latest published frame to local Unix-socket clients."""

    def __init__(self, socket_path: str, request_timeout_s: float = 0.25) -> None:
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        self.socket_path = Path(socket_path)
        self.request_timeout_s = float(request_timeout_s)
        self._condition = threading.Condition()
        self._latest: Optional[bytes] = None
        self._sequence = -1
        self._last_delivered = -1
        self._stop = threading.Event()
        self._listener: Optional[socket.socket] = None
        self._connection: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self.published_frames = 0
        self.replaced_frames = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.settimeout(0.1)
                probe.connect(str(self.socket_path))
            except OSError:
                self.socket_path.unlink()
            else:
                raise RuntimeError("IPC socket is already in use: %s" % self.socket_path)
            finally:
                probe.close()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.socket_path))
        os.chmod(str(self.socket_path), 0o600)
        listener.listen(1)
        listener.settimeout(0.2)
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, name="rgbd-ipc", daemon=True)
        self._thread.start()

    def publish(self, frame: Any) -> None:
        with self._condition:
            self._sequence += 1
            if self._latest is not None and self._sequence > self._last_delivered:
                self.replaced_frames += 1
            self._latest = encode_frame(frame, self._sequence)
            self.published_frames += 1
            self._condition.notify_all()

    def _serve(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                connection, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._connection = connection
            try:
                self._serve_client(connection)
            except (ConnectionError, OSError, struct.error):
                pass
            finally:
                connection.close()
                self._connection = None

    def _serve_client(self, connection: socket.socket) -> None:
        while not self._stop.is_set():
            last_sequence, client_timeout_s = _REQUEST.unpack(
                _receive_exact(connection, _REQUEST.size)
            )
            wait_s = min(self.request_timeout_s, max(client_timeout_s * 0.8, 0.001))
            deadline = time.monotonic() + wait_s
            with self._condition:
                while self._sequence <= last_sequence and not self._stop.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                packet = self._latest if self._sequence > last_sequence else None
            if packet is None:
                connection.sendall(_LENGTH.pack(0))
            else:
                self._last_delivered = max(self._last_delivered, self._sequence)
                connection.sendall(_LENGTH.pack(len(packet)) + packet)

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._listener is not None:
            self._listener.close()
        if self._connection is not None:
            try:
                self._connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._connection.close()
        if self._thread is not None:
            self._thread.join(2.0)
        if self.socket_path.exists():
            self.socket_path.unlink()


class UnixSocketCameraBackend:
    """Inference-side backend for :class:`LiveFrameSource`."""

    def __init__(self, options: Dict[str, Any]) -> None:
        self.socket_path = str(options.get("socket_path", "/tmp/kuka_rgbd.sock"))
        self.connect_timeout_s = float(options.get("connect_timeout_s", 5.0))
        self.max_packet_bytes = int(options.get("max_packet_bytes", _MAX_PACKET_BYTES))
        if self.connect_timeout_s <= 0 or self.max_packet_bytes <= 0:
            raise ValueError("IPC timeouts and packet limits must be positive")
        self._socket: Optional[socket.socket] = None
        self._last_sequence = -1

    def start(self) -> None:
        deadline = time.monotonic() + self.connect_timeout_s
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(min(self.connect_timeout_s, 0.25))
        while True:
            try:
                connection.connect(self.socket_path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    connection.close()
                    raise TimeoutError("timed out waiting for RGB-D IPC bridge")
                time.sleep(0.05)
        connection.settimeout(None)
        self._socket = connection

    def read(self, timeout_s: float) -> Optional[Any]:
        if self._socket is None:
            raise RuntimeError("IPC backend has not been started")
        self._socket.settimeout(timeout_s)
        try:
            self._socket.sendall(_REQUEST.pack(self._last_sequence, timeout_s))
            packet_size = _LENGTH.unpack(_receive_exact(self._socket, _LENGTH.size))[0]
            if packet_size == 0:
                return None
            if packet_size > self.max_packet_bytes:
                raise ValueError("IPC frame exceeds configured packet limit")
            sequence, frame = decode_frame(_receive_exact(self._socket, packet_size))
        except socket.timeout:
            return None
        finally:
            if self._socket is not None:
                self._socket.settimeout(None)
        if sequence <= self._last_sequence:
            raise ValueError("IPC frame sequence did not increase")
        self._last_sequence = sequence
        return frame

    def stop(self) -> None:
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None


def create_camera(options: Dict[str, Any]) -> UnixSocketCameraBackend:
    """Factory used by ``scripts/live_perception.py --backend``."""
    return UnixSocketCameraBackend(dict(options))
