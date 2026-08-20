import numpy as np
import pytest

from amr.grid.grid2d import UniformGrid2D
from amr.grid.hierarchy2d import AMRHierarchy2D
from amr.numerics.boundary_conditions import (
    fill_coarse_fine_ghost_cells_2d,
    fill_periodic_ghost_cells_2d,
)


def make_hierarchy() -> AMRHierarchy2D:
    grid = UniformGrid2D(0.0, 1.0, 4, 0.0, 1.0, 4)
    return AMRHierarchy2D(
        grid, np.arange(16, dtype=float).reshape(grid.shape), refinement_ratio=2
    )


def test_periodic_2d_padding_wraps_edges_and_corners() -> None:
    values = np.arange(12).reshape(3, 4)
    ghosted = fill_periodic_ghost_cells_2d(values)
    np.testing.assert_array_equal(ghosted[1:-1, 1:-1], values)
    np.testing.assert_array_equal(ghosted[0, 1:-1], values[-1])
    np.testing.assert_array_equal(ghosted[1:-1, 0], values[:, -1])
    assert ghosted[0, 0] == values[-1, -1]


def test_parent_supplies_all_four_coarse_fine_edges() -> None:
    hierarchy = make_hierarchy()
    child = hierarchy.add_patch(hierarchy.root, 1, 3, 1, 3)
    ghosted = fill_coarse_fine_ghost_cells_2d(child)

    np.testing.assert_array_equal(ghosted[1:-1, 0], [4.0, 4.0, 8.0, 8.0])
    np.testing.assert_array_equal(ghosted[1:-1, -1], [7.0, 7.0, 11.0, 11.0])
    np.testing.assert_array_equal(ghosted[0, 1:-1], [1.0, 1.0, 2.0, 2.0])
    np.testing.assert_array_equal(ghosted[-1, 1:-1], [13.0, 13.0, 14.0, 14.0])


def test_adjacent_fine_patch_takes_precedence_on_shared_edge() -> None:
    hierarchy = make_hierarchy()
    left = hierarchy.add_patch(
        hierarchy.root, 0, 2, 1, 3, values=np.arange(16).reshape(4, 4)
    )
    right_values = 100.0 + np.arange(16).reshape(4, 4)
    hierarchy.add_patch(hierarchy.root, 2, 4, 1, 3, values=right_values)
    ghosted = fill_coarse_fine_ghost_cells_2d(left)
    np.testing.assert_array_equal(ghosted[1:-1, -1], right_values[:, 0])


def test_periodic_boundary_prefers_fine_sibling_across_domain() -> None:
    hierarchy = make_hierarchy()
    left = hierarchy.add_patch(
        hierarchy.root, 0, 1, 1, 3, values=np.arange(8).reshape(4, 2)
    )
    right_values = 50.0 + np.arange(8).reshape(4, 2)
    hierarchy.add_patch(hierarchy.root, 3, 4, 1, 3, values=right_values)
    ghosted = fill_coarse_fine_ghost_cells_2d(left)
    np.testing.assert_array_equal(ghosted[1:-1, 0], right_values[:, -1])


def test_temporally_interpolated_parent_can_supply_ghosts() -> None:
    hierarchy = make_hierarchy()
    child = hierarchy.add_patch(hierarchy.root, 1, 3, 1, 3)
    ghosted = fill_coarse_fine_ghost_cells_2d(
        child, parent_values=hierarchy.root.values + 100.0
    )
    np.testing.assert_array_equal(ghosted[1:-1, 0], [104.0, 104.0, 108.0, 108.0])


def test_bilinear_parent_interpolation_is_exact_for_linear_field() -> None:
    grid = UniformGrid2D(0.0, 1.0, 6, 0.0, 1.0, 6)
    x, y = grid.cell_centres
    values = 2.0 * x - 3.0 * y + 1.0
    hierarchy = AMRHierarchy2D(grid, values)
    child = hierarchy.add_patch(hierarchy.root, 2, 4, 2, 4)
    ghosted = fill_coarse_fine_ghost_cells_2d(
        child, parent_interpolation="bilinear"
    )
    ghost_x = child.grid.x_min - 0.5 * child.grid.dx
    left_y = child.grid.y_centres
    np.testing.assert_allclose(
        ghosted[1:-1, 0], 2.0 * ghost_x - 3.0 * left_y + 1.0
    )


def test_nonperiodic_physical_boundary_requires_explicit_data() -> None:
    hierarchy = make_hierarchy()
    child = hierarchy.add_patch(hierarchy.root, 0, 2, 0, 2)
    with pytest.raises(ValueError, match="physical boundary"):
        fill_coarse_fine_ghost_cells_2d(child, periodic=False)
