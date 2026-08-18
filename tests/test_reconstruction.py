import numpy as np
import pytest

from amr.numerics.reconstruction import minmod, monotonized_central_slopes


def test_minmod_selects_smallest_common_sign_magnitude() -> None:
    first = np.array([2.0, -3.0, 1.0, 0.0])
    second = np.array([1.0, -2.0, -4.0, 5.0])
    np.testing.assert_array_equal(minmod(first, second), [1.0, -2.0, 0.0, 0.0])


def test_minmod_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        minmod(np.ones(2), np.ones(3))


def test_mc_slopes_are_exact_for_bounded_linear_data() -> None:
    values = np.array([0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(
        monotonized_central_slopes(values, periodic=False),
        1.0,
    )


def test_periodic_mc_limiter_flattens_extrema() -> None:
    values = np.array([0.0, 1.0, 2.0, 1.0])
    slopes = monotonized_central_slopes(values)
    assert slopes[2] == 0.0
    assert slopes[0] == 0.0
