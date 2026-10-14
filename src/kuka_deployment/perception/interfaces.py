"""Typed, dependency-free perception input and output contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt


class TrackingStatus(str, Enum):
    REGISTERED = "registered"
    TRACKED = "tracked"
    REREGISTERED = "reregistered"
    DETECTION_FAILED = "detection_failed"
    REGISTRATION_FAILED = "registration_failed"
    TRACKING_FAILED = "tracking_failed"


@dataclass(frozen=True)
class RGBDFrame:
    """Aligned RGB and metric depth in ``frame_id`` camera coordinates.

    Invalid depth is represented by 0.0. ``timestamp_ns=None`` means no reliable
    capture timestamp is available; derived video presentation time belongs in
    ``presentation_timestamp_ns`` and is never presented as capture time.
    """

    rgb: npt.NDArray[np.uint8]
    depth_m: npt.NDArray[np.float32]
    K: npt.NDArray[np.floating]
    frame_number: int
    timestamp_ns: Optional[int]
    frame_id: str
    presentation_timestamp_ns: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.rgb, np.ndarray) or self.rgb.dtype != np.uint8:
            raise TypeError("rgb must be a uint8 numpy array in RGB order")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (H, W, 3)")
        if not isinstance(self.depth_m, np.ndarray) or self.depth_m.dtype != np.float32:
            raise TypeError("depth_m must be float32 metres")
        if self.depth_m.ndim != 2 or self.depth_m.shape != self.rgb.shape[:2]:
            raise ValueError("depth and RGB image dimensions must match")
        if not np.all(np.isfinite(self.depth_m)) or np.any(self.depth_m < 0):
            raise ValueError("depth_m must be finite and non-negative; use 0 for invalid depth")
        K = np.asarray(self.K)
        if K.shape != (3, 3) or K.dtype.kind != "f" or not np.all(np.isfinite(K)):
            raise ValueError("K must be a finite 3x3 floating-point matrix")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        if isinstance(self.frame_number, bool) or self.frame_number < 0:
            raise ValueError("frame_number must be a non-negative integer")
        for name, value in (("timestamp_ns", self.timestamp_ns),
                            ("presentation_timestamp_ns", self.presentation_timestamp_ns)):
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be None or a non-negative integer")
        if not self.frame_id:
            raise ValueError("frame_id must be non-empty")


@dataclass(frozen=True)
class ObjectPoseEstimate:
    """Pose of an object in a declared parent frame.

    ``T_frame_object`` maps object-frame coordinates into ``frame_id`` coordinates;
    its translation is in metres. Invalid estimates must use ``None``. Detector
    confidence is segmentation confidence only, never tracking confidence.
    """

    object_id: str
    timestamp_ns: Optional[int]
    frame_id: str
    T_frame_object: Optional[npt.NDArray[np.floating]]
    valid: bool
    status: TrackingStatus
    source_frame_number: int
    detector_confidence: Optional[float] = None
    failure_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.object_id or not self.frame_id:
            raise ValueError("object_id and frame_id must be non-empty")
        if self.timestamp_ns is not None and self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be None or non-negative")
        if self.source_frame_number < 0:
            raise ValueError("source_frame_number must be non-negative")
        if self.detector_confidence is not None and not 0 <= self.detector_confidence <= 1:
            raise ValueError("detector_confidence must be in [0, 1]")
        if self.valid:
            if self.T_frame_object is None:
                raise ValueError("a valid estimate requires a pose")
            from .transforms import validate_transform
            validate_transform(self.T_frame_object)
            if self.failure_reason is not None:
                raise ValueError("a valid estimate cannot have a failure_reason")
        elif self.T_frame_object is not None:
            raise ValueError("an invalid estimate must not contain a pose")
