import numpy as np
import pytest

from amr.refinement.prolongation import (
    prolong_conservative_linear,
    prolong_piecewise_constant,
)
from amr.refinement.restriction import restrict_cell_averages


@pytest.mark.parametrize("ratio", [2, 3, 4])
def test_prolongation_preserves_every_parent_average(ratio: int) -> None:
    coarse = np.array([-2.0, 0.5, 4.0])
    fine = prolong_piecewise_constant(coarse, ratio)
    np.testing.assert_allclose(fine.reshape(-1, ratio).mean(axis=1), coarse)


@pytest.mark.parametrize("ratio", [2, 3])
def test_restriction_is_the_conservative_average(ratio: int) -> None:
    fine = np.arange(4 * ratio, dtype=float)
    expected = fine.reshape(-1, ratio).mean(axis=1)
    np.testing.assert_allclose(restrict_cell_averages(fine, ratio), expected)


def test_prolong_then_restrict_is_identity() -> None:
    coarse = np.array([0.2, -1.0, 3.4, 8.1])
    fine = prolong_piecewise_constant(coarse, 2)
    np.testing.assert_array_equal(restrict_cell_averages(fine, 2), coarse)


def test_transfer_preserves_integrated_quantity() -> None:
    coarse = np.array([1.0, 3.0, -0.5])
    coarse_dx = 0.2
    ratio = 3
    fine = prolong_piecewise_constant(coarse, ratio)
    assert np.sum(fine) * coarse_dx / ratio == pytest.approx(np.sum(coarse) * coarse_dx)


def test_restriction_rejects_incomplete_fine_groups() -> None:
    with pytest.raises(ValueError, match="divisible"):
        restrict_cell_averages([1.0, 2.0, 3.0], refinement_ratio=2)


@pytest.mark.parametrize("ratio", [2, 3, 4])
def test_linear_prolongation_preserves_parent_averages(ratio: int) -> None:
    coarse = np.array([1.0, 1.5, 2.5, 2.0, 0.5])
    fine = prolong_conservative_linear(coarse, ratio)
    np.testing.assert_allclose(fine.reshape(-1, ratio).mean(axis=1), coarse, atol=2.0e-15)


def test_limited_linear_prolongation_creates_no_new_extrema() -> None:
    coarse = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    fine = prolong_conservative_linear(coarse, 2)
    assert np.min(fine) >= 0.0
    assert np.max(fine) <= 1.0
