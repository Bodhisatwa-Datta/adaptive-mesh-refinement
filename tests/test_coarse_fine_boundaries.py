import numpy as np
import pytest

from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.numerics.boundary_conditions import fill_coarse_fine_ghost_cells


def make_hierarchy() -> AMRHierarchy1D:
    grid = UniformGrid1D(0.0, 1.0, 8)
    return AMRHierarchy1D(grid, np.arange(8, dtype=float), refinement_ratio=2)


def test_parent_supplies_coarse_fine_ghost_values() -> None:
    hierarchy = make_hierarchy()
    child = hierarchy.add_patch(hierarchy.root, 2, 4)
    ghosted = fill_coarse_fine_ghost_cells(child)
    np.testing.assert_array_equal(ghosted, [1.0, 2.0, 2.0, 3.0, 3.0, 4.0])


def test_adjacent_fine_patch_takes_precedence_over_parent() -> None:
    hierarchy = make_hierarchy()
    left = hierarchy.add_patch(hierarchy.root, 2, 4, values=[20.0, 21.0, 22.0, 23.0])
    hierarchy.add_patch(hierarchy.root, 4, 6, values=[40.0, 41.0, 42.0, 43.0])
    ghosted = fill_coarse_fine_ghost_cells(left)
    assert ghosted[-1] == 40.0


def test_periodic_boundary_uses_fine_patch_across_domain() -> None:
    hierarchy = make_hierarchy()
    left = hierarchy.add_patch(hierarchy.root, 0, 2, values=[0.0, 1.0, 2.0, 3.0])
    hierarchy.add_patch(hierarchy.root, 6, 8, values=[60.0, 61.0, 62.0, 63.0])
    ghosted = fill_coarse_fine_ghost_cells(left, periodic=True)
    assert ghosted[0] == 63.0


def test_nonperiodic_physical_boundary_requires_explicit_data() -> None:
    hierarchy = make_hierarchy()
    child = hierarchy.add_patch(hierarchy.root, 0, 2)
    with pytest.raises(ValueError, match="physical boundary"):
        fill_coarse_fine_ghost_cells(child, periodic=False)

