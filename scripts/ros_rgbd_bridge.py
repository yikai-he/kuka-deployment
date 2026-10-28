#!/usr/bin/env python3
"""Read-only ROS1 RGB-D subscriber bridge to local latest-frame IPC."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
# Import this Python-3.8-compatible module directly, without importing the
# Python-3.11 inference package or its type-annotation syntax under ROS Noetic.
sys.path.insert(0, str(PROJECT / "src" / "kuka_deployment" / "perception"))

from ros_ipc import LatestFrameServer


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-topic", default="/cam_1/color/image_raw")
    parser.add_argument(
        "--depth-topic", default="/cam_1/aligned_depth_to_color/image_raw"
    )
    parser.add_argument("--camera-info-topic", default="/cam_1/color/camera_info")
    parser.add_argument("--socket", default="/tmp/kuka_rgbd.sock")
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--sync-queue-size", type=int, default=10)
    parser.add_argument("--sync-slop-s", type=float, default=0.03)
    parser.add_argument("--request-timeout-s", type=float, default=0.25)
    return parser.parse_args(argv)


def validate_options(args):
    if not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
        raise ValueError("--depth-scale must be finite and positive")
    if not 0 <= args.min_depth_m < args.max_depth_m:
        raise ValueError("depth limits must satisfy 0 <= min < max")
    if args.sync_queue_size <= 0 or args.sync_slop_s < 0:
        raise ValueError("synchronizer queue must be positive and slop non-negative")


def convert_ros_messages(rgb_msg, depth_msg, info_msg, bridge, args, frame_number):
    """Validate and convert synchronized ROS-like messages for IPC publication."""
    if rgb_msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError("unsupported RGB encoding: %s" % rgb_msg.encoding)
    if depth_msg.encoding not in ("16UC1", "mono16", "32FC1"):
        raise ValueError("unsupported depth encoding: %s" % depth_msg.encoding)
    rgb = np.asarray(bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough"))
    depth_raw = np.asarray(
        bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    )
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB message must decode to uint8 HxWx3")
    if rgb_msg.encoding == "bgr8":
        rgb = rgb[..., ::-1]
    if depth_msg.encoding in ("16UC1", "mono16"):
        if depth_raw.dtype != np.uint16:
            raise ValueError("16-bit depth encoding did not decode as uint16")
        depth_m = depth_raw.astype(np.float32) * np.float32(args.depth_scale)
    else:
        if depth_raw.dtype not in (np.float32, np.float64):
            raise ValueError("32FC1 depth did not decode as floating point")
        depth_m = depth_raw.astype(np.float32)
    if depth_m.shape != rgb.shape[:2]:
        raise ValueError("aligned depth and RGB dimensions do not match")
    if int(info_msg.width) != rgb.shape[1] or int(info_msg.height) != rgb.shape[0]:
        raise ValueError("CameraInfo dimensions do not match RGB")
    K = np.asarray(info_msg.K, dtype=np.float32).reshape(3, 3)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("CameraInfo contains invalid intrinsics")
    rgb_frame = str(rgb_msg.header.frame_id)
    depth_frame = str(depth_msg.header.frame_id)
    info_frame = str(info_msg.header.frame_id)
    if not rgb_frame or rgb_frame != depth_frame or rgb_frame != info_frame:
        raise ValueError("RGB, aligned depth, and CameraInfo frame_id must match")
    if not rgb_frame.endswith("_optical_frame"):
        raise ValueError("camera frame_id must use a ROS optical frame")
    invalid = (~np.isfinite(depth_m)) | (depth_m < args.min_depth_m) | (
        depth_m > args.max_depth_m
    )
    depth_m[invalid] = 0.0
    stamp = rgb_msg.header.stamp
    timestamp_ns = int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)
    return SimpleNamespace(
        rgb=np.ascontiguousarray(rgb),
        depth_m=np.ascontiguousarray(depth_m, dtype=np.float32),
        K=K,
        frame_number=int(frame_number),
        timestamp_ns=timestamp_ns,
        frame_id=rgb_frame,
        acquisition_monotonic_ns=time.monotonic_ns(),
    )


def main(argv=None):
    args = arguments(argv)
    validate_options(args)
    try:
        import message_filters
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image
    except ImportError as exc:
        raise RuntimeError("ROS Noetic, message_filters, and cv_bridge are required") from exc

    rospy.init_node("kuka_rgbd_ipc_bridge", anonymous=False, disable_signals=True)
    server = LatestFrameServer(args.socket, args.request_timeout_s)
    server.start()
    cv_bridge = CvBridge()
    state = {"frame_number": 0, "conversion_errors": 0}

    def synchronized(rgb_msg, depth_msg, info_msg):
        try:
            frame = convert_ros_messages(
                rgb_msg,
                depth_msg,
                info_msg,
                cv_bridge,
                args,
                state["frame_number"],
            )
            server.publish(frame)
            state["frame_number"] += 1
        except Exception as exc:
            state["conversion_errors"] += 1
            rospy.logwarn_throttle(2.0, "RGB-D frame rejected: %s", exc)

    rgb_sub = message_filters.Subscriber(args.rgb_topic, Image, queue_size=1)
    depth_sub = message_filters.Subscriber(args.depth_topic, Image, queue_size=1)
    info_sub = message_filters.Subscriber(args.camera_info_topic, CameraInfo, queue_size=1)
    synchronizer = message_filters.ApproximateTimeSynchronizer(
        [rgb_sub, depth_sub, info_sub],
        queue_size=args.sync_queue_size,
        slop=args.sync_slop_s,
        allow_headerless=False,
    )
    synchronizer.registerCallback(synchronized)

    stopping = {"value": False}

    def stop(_signum=None, _frame=None):
        stopping["value"] = True
        rospy.signal_shutdown("interrupt")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    rospy.loginfo("read-only RGB-D IPC bridge listening at %s", args.socket)
    try:
        while not stopping["value"] and not rospy.is_shutdown():
            time.sleep(0.1)
    finally:
        server.close()
        rospy.loginfo(
            "bridge stopped: published=%d replaced=%d rejected=%d",
            server.published_frames,
            server.replaced_frames,
            state["conversion_errors"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
