"""CPU-only CAD pose projection and RGB overlay helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .transforms import validate_transform


BOX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def box_corners(bounds: npt.ArrayLike) -> np.ndarray:
    """Return eight consistently ordered corners from 2x3 CAD-space bounds."""
    value = np.asarray(bounds, dtype=np.float64)
    if value.shape != (2, 3) or not np.all(np.isfinite(value)):
        raise ValueError("bounds must be a finite 2x3 array")
    if np.any(value[1] <= value[0]):
        raise ValueError("each upper mesh bound must exceed its lower bound")
    low, high = value
    return np.array(
        [
            [low[0], low[1], low[2]],
            [high[0], low[1], low[2]],
            [high[0], high[1], low[2]],
            [low[0], high[1], low[2]],
            [low[0], low[1], high[2]],
            [high[0], low[1], high[2]],
            [high[0], high[1], high[2]],
            [low[0], high[1], high[2]],
        ],
        dtype=np.float64,
    )


def transform_object_points(
    points_object: npt.ArrayLike, T_camera_object: npt.ArrayLike
) -> np.ndarray:
    """Transform Nx3 CAD/object points into the optical camera frame."""
    points = np.asarray(points_object, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("object points must be a finite Nx3 array")
    pose = validate_transform(np.asarray(T_camera_object, dtype=np.float64))
    return (pose[:3, :3] @ points.T).T + pose[:3, 3]


def project_camera_points(
    points_camera: npt.ArrayLike,
    K: npt.ArrayLike,
    distortion: npt.ArrayLike | None = None,
) -> np.ndarray | None:
    """Project optical-camera points with optional five-term plumb-bob distortion.

    Returns ``None`` when any point is on or behind the camera plane. The distortion
    order is ``[k1, k2, p1, p2, k3]``, matching the working Colab overlay.
    """
    points = np.asarray(points_camera, dtype=np.float64)
    intrinsics = np.asarray(K, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("camera points must be a finite Nx3 array")
    if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("K must be a finite 3x3 matrix")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("camera focal lengths must be positive")
    if np.any(points[:, 2] <= 1e-9):
        return None

    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    if distortion is not None:
        coefficients = np.asarray(distortion, dtype=np.float64).reshape(-1)
        if coefficients.size < 5 or not np.all(np.isfinite(coefficients[:5])):
            raise ValueError("distortion must provide finite k1,k2,p1,p2,k3")
        k1, k2, p1, p2, k3 = coefficients[:5]
        radius2 = x * x + y * y
        radial = 1 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
        distorted_x = x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x)
        distorted_y = y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y
        x, y = distorted_x, distorted_y

    u = intrinsics[0, 0] * x + intrinsics[0, 1] * y + intrinsics[0, 2]
    v = intrinsics[1, 0] * x + intrinsics[1, 1] * y + intrinsics[1, 2]
    return np.column_stack((u, v))


def project_object_points(
    points_object: npt.ArrayLike,
    T_camera_object: npt.ArrayLike,
    K: npt.ArrayLike,
    distortion: npt.ArrayLike | None = None,
) -> np.ndarray | None:
    return project_camera_points(
        transform_object_points(points_object, T_camera_object), K, distortion
    )


def draw_cad_overlay(
    rgb: np.ndarray,
    T_camera_object: npt.ArrayLike,
    K: npt.ArrayLike,
    corners_object: npt.ArrayLike,
    *,
    distortion: npt.ArrayLike | None = None,
    axis_length_m: float = 0.03,
    line_width: int = 2,
) -> np.ndarray:
    """Draw a CAD bounding box and object-frame axes on an RGB uint8 image."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be uint8 with shape (H,W,3)")
    if axis_length_m <= 0 or line_width <= 0:
        raise ValueError("axis length and line width must be positive")
    corners = np.asarray(corners_object, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError("corners_object must have shape (8,3)")

    projected = project_object_points(corners, T_camera_object, K, distortion)
    if projected is None:
        return rgb.copy()
    axis_points = np.array(
        [
            [0, 0, 0],
            [axis_length_m, 0, 0],
            [0, axis_length_m, 0],
            [0, 0, axis_length_m],
        ],
        dtype=np.float64,
    )
    projected_axes = project_object_points(
        axis_points, T_camera_object, K, distortion
    )
    if projected_axes is None:
        return rgb.copy()

    import cv2

    result = np.ascontiguousarray(rgb.copy())
    pixels = np.rint(projected).astype(np.int32)
    for start, end in BOX_EDGES:
        cv2.line(
            result,
            tuple(pixels[start]),
            tuple(pixels[end]),
            (0, 255, 0),
            line_width,
            cv2.LINE_AA,
        )
    axes = np.rint(projected_axes).astype(np.int32)
    origin = tuple(axes[0])
    for endpoint, color in zip(
        axes[1:], ((255, 0, 0), (0, 255, 0), (0, 0, 255)), strict=True
    ):
        cv2.line(result, origin, tuple(endpoint), color, line_width, cv2.LINE_AA)
    cv2.circle(result, origin, max(2, line_width + 1), (255, 255, 255), -1)
    return result
