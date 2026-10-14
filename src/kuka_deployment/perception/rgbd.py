"""Explicit RGB and depth conversion helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def bgr_to_rgb(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("BGR image must be uint8 with shape (H, W, 3)")
    return np.ascontiguousarray(image[..., ::-1])


def depth_to_metres(
    raw_depth: npt.NDArray[np.generic], depth_scale_m: float, *,
    min_depth_m: float = 0.001, max_depth_m: float | None = None,
) -> npt.NDArray[np.float32]:
    """Convert raw depth exactly once; metric float input is intentionally rejected."""
    if raw_depth.ndim != 2:
        raise ValueError("raw depth must be a 2D image")
    if raw_depth.dtype.kind not in "ui":
        raise TypeError("raw depth must have an integer dtype to prevent accidental double scaling")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError("depth_scale_m must be finite and positive")
    if min_depth_m < 0 or (max_depth_m is not None and max_depth_m <= min_depth_m):
        raise ValueError("invalid depth range")
    depth = raw_depth.astype(np.float32) * np.float32(depth_scale_m)
    invalid = ~np.isfinite(depth) | (depth < min_depth_m)
    if max_depth_m is not None:
        invalid |= depth > max_depth_m
    depth[invalid] = 0.0
    return depth
