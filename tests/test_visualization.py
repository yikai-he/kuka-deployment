import numpy as np

from kuka_deployment.perception.visualization import (
    box_corners,
    draw_cad_overlay,
    project_camera_points,
    project_object_points,
    transform_object_points,
)


def test_box_corners_preserve_actual_cad_bounds():
    bounds = np.array([[-0.02, -0.01, -0.03], [0.02, 0.03, 0.01]])
    corners = box_corners(bounds)
    np.testing.assert_allclose(corners.min(axis=0), bounds[0])
    np.testing.assert_allclose(corners.max(axis=0), bounds[1])
    assert np.unique(corners, axis=0).shape == (8, 3)


def test_transform_and_pinhole_projection_follow_T_camera_object():
    pose = np.eye(4)
    pose[:3, 3] = [0.1, -0.2, 2.0]
    point = np.array([[0.1, 0.2, 0.0]])
    camera = transform_object_points(point, pose)
    np.testing.assert_allclose(camera, [[0.2, 0.0, 2.0]])
    K = np.array([[100.0, 0, 50], [0, 120.0, 40], [0, 0, 1]])
    projected = project_object_points(point, pose, K)
    np.testing.assert_allclose(projected, [[60.0, 40.0]])


def test_projection_supports_colab_plumb_bob_order():
    point = np.array([[1.0, 0.0, 2.0]])
    K = np.array([[100.0, 0, 0], [0, 100.0, 0], [0, 0, 1]])
    projected = project_camera_points(point, K, [1.0, 0, 0, 0, 0])
    # x=0.5, r^2=0.25, radial=1.25 -> u=62.5
    np.testing.assert_allclose(projected, [[62.5, 0.0]])


def test_projection_rejects_geometry_behind_camera():
    K = np.eye(3)
    assert project_camera_points([[0, 0, -1]], K) is None


def test_draw_overlay_is_cpu_only_and_does_not_mutate_rgb():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    original = rgb.copy()
    pose = np.eye(4)
    pose[2, 3] = 1.0
    K = np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])
    corners = box_corners([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])
    overlay = draw_cad_overlay(rgb, pose, K, corners, axis_length_m=0.1)
    np.testing.assert_array_equal(rgb, original)
    assert overlay.shape == rgb.shape
    assert overlay.dtype == np.uint8
    assert np.count_nonzero(overlay) > 0
