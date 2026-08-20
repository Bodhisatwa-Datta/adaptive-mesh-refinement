import numpy as np
import pytest

from amr.diagnostics.conservation import composite_mass_2d
from amr.grid.grid2d import UniformGrid2D
from amr.grid.hierarchy2d import AMRHierarchy2D


def make_hierarchy(ratio: int = 2) -> AMRHierarchy2D:
    grid = UniformGrid2D(0.0, 3.0, 6, -1.0, 1.0, 4)
    values = np.arange(24, dtype=float).reshape(grid.shape)
    return AMRHierarchy2D(grid, values, refinement_ratio=ratio)


def test_rectangular_child_geometry_and_piecewise_prolongation() -> None:
    hierarchy = make_hierarchy()
    child = hierarchy.add_patch(hierarchy.root, 1, 4, 1, 3)

    assert child.grid.shape == (4, 6)
    assert child.level == 1
    assert child.parent is hierarchy.root
    assert child.parent_range == ((1, 3), (1, 4))
    assert child.physical_bounds == pytest.approx((0.5, 2.0, -0.5, 0.5))
    expected = np.repeat(
        np.repeat(hierarchy.root.values[1:3, 1:4], 2, axis=0), 2, axis=1
    )
    np.testing.assert_array_equal(child.values, expected)


def test_sibling_rectangles_may_touch_but_not_overlap() -> None:
    hierarchy = make_hierarchy()
    hierarchy.add_patch(hierarchy.root, 0, 2, 0, 2)
    hierarchy.add_patch(hierarchy.root, 2, 4, 0, 2)
    hierarchy.add_patch(hierarchy.root, 0, 2, 2, 4)

    with pytest.raises(ValueError, match="overlap"):
        hierarchy.add_patch(hierarchy.root, 1, 3, 1, 3)


def test_multilevel_tree_and_cell_counts() -> None:
    hierarchy = make_hierarchy()
    child = hierarchy.add_patch(hierarchy.root, 1, 4, 1, 3)
    grandchild = hierarchy.add_patch(child, 2, 4, 1, 3)

    assert hierarchy.patches == (hierarchy.root, child, grandchild)
    assert hierarchy.patches_at_level(2) == (grandchild,)
    assert hierarchy.n_stored_cells == 24 + 24 + 16
    assert hierarchy.n_active_cells == (24 - 6) + (24 - 4) + 16


def test_restriction_and_derefinement_preserve_composite_mass() -> None:
    hierarchy = make_hierarchy(ratio=3)
    initial_mass = composite_mass_2d(hierarchy)
    child = hierarchy.add_patch(hierarchy.root, 2, 5, 1, 3)
    assert composite_mass_2d(hierarchy) == pytest.approx(initial_mass)

    child.values += np.arange(child.n_valid_cells).reshape(child.grid.shape) / 100.0
    refined_mass = composite_mass_2d(hierarchy)
    hierarchy.remove_patch(child, restrict=True)

    assert composite_mass_2d(hierarchy) == pytest.approx(refined_mass)
    assert hierarchy.patches == (hierarchy.root,)


def test_nonleaf_patch_cannot_be_removed() -> None:
    hierarchy = make_hierarchy()
    child = hierarchy.add_patch(hierarchy.root, 1, 4, 1, 3)
    hierarchy.add_patch(child, 0, 2, 0, 2)
    with pytest.raises(ValueError, match="children"):
        hierarchy.remove_patch(child)


def test_patch_rejects_values_with_wrong_shape() -> None:
    hierarchy = make_hierarchy()
    with pytest.raises(ValueError, match="Expected field shape"):
        hierarchy.add_patch(
            hierarchy.root, 1, 3, 1, 3, values=np.ones((3, 4))
        )
