import numpy as np
import pytest

from amr.refinement.prolongation import (
    prolong_conservative_quadratic_2d,
    prolong_piecewise_constant_2d,
)
from amr.refinement.restriction import restrict_cell_averages_2d


@pytest.mark.parametrize("ratio", [2, 3, 4])
def test_piecewise_constant_2d_transfer_is_conservative(ratio: int) -> None:
    coarse = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    fine = prolong_piecewise_constant_2d(coarse, ratio)

    assert fine.shape == (coarse.shape[0] * ratio, coarse.shape[1] * ratio)
    np.testing.assert_allclose(restrict_cell_averages_2d(fine, ratio), coarse)
    assert np.sum(fine) / ratio**2 == pytest.approx(np.sum(coarse))


def test_restriction_averages_each_rectangular_fine_block() -> None:
    fine = np.arange(24, dtype=float).reshape(4, 6)
    coarse = restrict_cell_averages_2d(fine, refinement_ratio=2)
    expected = np.array(
        [
            [np.mean(fine[0:2, 0:2]), np.mean(fine[0:2, 2:4]), np.mean(fine[0:2, 4:6])],
            [np.mean(fine[2:4, 0:2]), np.mean(fine[2:4, 2:4]), np.mean(fine[2:4, 4:6])],
        ]
    )
    np.testing.assert_allclose(coarse, expected)


def test_restriction_rejects_nondivisible_dimension() -> None:
    with pytest.raises(ValueError, match="Both fine dimensions"):
        restrict_cell_averages_2d(np.ones((5, 6)), refinement_ratio=2)


@pytest.mark.parametrize("ratio", [2, 3])
def test_quadratic_2d_prolongation_preserves_every_parent_average(
    ratio: int,
) -> None:
    y, x = np.mgrid[:6, :8]
    coarse = 1.0 + 0.2 * x - 0.1 * y + 0.03 * x * y
    fine = prolong_conservative_quadratic_2d(coarse, ratio)
    restricted = restrict_cell_averages_2d(fine, ratio)
    np.testing.assert_allclose(restricted, coarse, atol=3.0e-15)


def test_quadratic_2d_prolongation_improves_smooth_periodic_data() -> None:
    ny, nx = 12, 16
    y, x = np.mgrid[:ny, :nx]
    coarse = np.sin(2.0 * np.pi * (x + 0.5) / nx) * np.cos(
        2.0 * np.pi * (y + 0.5) / ny
    )
    quadratic = prolong_conservative_quadratic_2d(coarse, 2)
    constant = prolong_piecewise_constant_2d(coarse, 2)
    fine_y, fine_x = np.mgrid[: 2 * ny, : 2 * nx]
    exact = np.sin(2.0 * np.pi * (fine_x + 0.5) / (2 * nx)) * np.cos(
        2.0 * np.pi * (fine_y + 0.5) / (2 * ny)
    )
    assert np.mean(np.abs(quadratic - exact)) < np.mean(np.abs(constant - exact))
