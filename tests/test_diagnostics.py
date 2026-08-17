import numpy as np
import pytest

from amr.diagnostics.errors import error_norms
from amr.diagnostics.conservation import composite_mass
from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D


def test_error_norm_definitions() -> None:
    errors = error_norms(np.array([1.0, -1.0]), np.zeros(2))
    assert errors.l1 == pytest.approx(1.0)
    assert errors.l2 == pytest.approx(1.0)
    assert errors.linf == pytest.approx(1.0)


def test_composite_mass_excludes_covered_coarse_cells() -> None:
    grid = UniformGrid1D(0.0, 1.0, 4)
    hierarchy = AMRHierarchy1D(grid, [1.0, 2.0, 3.0, 4.0])
    hierarchy.add_patch(hierarchy.root, 1, 3, values=[10.0, 10.0, 20.0, 20.0])
    expected = (1.0 + 4.0) * 0.25 + (10.0 + 10.0 + 20.0 + 20.0) * 0.125
    assert composite_mass(hierarchy) == pytest.approx(expected)
