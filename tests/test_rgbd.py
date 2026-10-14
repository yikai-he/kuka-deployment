import numpy as np
import pytest

from kuka_deployment.perception.rgbd import bgr_to_rgb, depth_to_metres


def test_depth_is_explicitly_scaled_and_invalidated():
    raw = np.array([[0, 1000, 4000]], dtype=np.uint16)
    actual = depth_to_metres(raw, 0.001, max_depth_m=3.0)
    np.testing.assert_allclose(actual, [[0, 1, 0]])
    assert actual.dtype == np.float32


def test_float_depth_is_rejected_to_prevent_double_scaling():
    with pytest.raises(TypeError):
        depth_to_metres(np.ones((2, 2), np.float32), 0.001)


def test_bgr_to_rgb_changes_channel_order():
    result = bgr_to_rgb(np.array([[[1, 2, 3]]], np.uint8))
    np.testing.assert_array_equal(result, [[[3, 2, 1]]])
