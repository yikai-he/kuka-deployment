"""Pure SE(3) utilities; transforms map child coordinates into parent coordinates."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def validate_transform(T: npt.ArrayLike, *, atol: float = 1e-5) -> np.ndarray:
    out = np.asarray(T)
    if out.shape != (4, 4) or out.dtype.kind != "f" or not np.all(np.isfinite(out)):
        raise ValueError("transform must be a finite floating-point 4x4 matrix")
    if not np.allclose(out[3], [0, 0, 0, 1], atol=atol):
        raise ValueError("transform bottom row must be [0, 0, 0, 1]")
    R = out[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=atol) or not np.isclose(
        np.linalg.det(R), 1, atol=atol
    ):
        raise ValueError("rotation must be orthonormal with determinant +1")
    return out


def compose(
    T_parent_middle: npt.ArrayLike, T_middle_child: npt.ArrayLike
) -> np.ndarray:
    return validate_transform(
        validate_transform(T_parent_middle) @ validate_transform(T_middle_child)
    )


def invert(T_parent_child: npt.ArrayLike) -> np.ndarray:
    T = validate_transform(T_parent_child)
    result = np.eye(4, dtype=T.dtype)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T[:3, 3]
    return result


def transform_point(
    T_parent_child: npt.ArrayLike, point_child: npt.ArrayLike
) -> np.ndarray:
    T = validate_transform(T_parent_child)
    point = np.asarray(point_child)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point must be a finite 3-vector")
    return T[:3, :3] @ point + T[:3, 3]
