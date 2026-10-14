import numpy as np
import pytest

from kuka_deployment.perception.transforms import compose, invert, transform_point, validate_transform


def test_composition_inversion_and_point_transform():
    a = np.eye(4); a[:3, 3] = [1, 2, 3]
    b = np.eye(4); b[:3, 3] = [0.5, 0, 0]
    combined = compose(a, b)
    np.testing.assert_allclose(combined[:3, 3], [1.5, 2, 3])
    np.testing.assert_allclose(compose(combined, invert(combined)), np.eye(4), atol=1e-7)
    np.testing.assert_allclose(transform_point(a, [1, 1, 1]), [2, 3, 4])


def test_non_rigid_matrix_rejected():
    bad = np.eye(4); bad[0, 0] = 2
    with pytest.raises(ValueError):
        validate_transform(bad)
