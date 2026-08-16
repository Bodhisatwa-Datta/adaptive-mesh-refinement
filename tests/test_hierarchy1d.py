import numpy as np
import pytest

from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D


def make_hierarchy() -> AMRHierarchy1D:
    grid = UniformGrid1D(0.0, 1.0, 8)
    return AMRHierarchy1D(grid, np.arange(8, dtype=float), refinement_ratio=2)


def test_child_coordinates_indexing_and_relationships() -> None:
    hierarchy = make_hierarchy()
    child = hierarchy.add_patch(hierarchy.root, 2, 5)

    assert child.parent is hierarchy.root
    assert hierarchy.root.children == [child]
    assert child.level == 1
    assert child.parent_range == (2, 5)
    assert child.physical_bounds == pytest.approx((0.25, 0.625))
    assert child.grid.dx == pytest.approx(hierarchy.root.grid.dx / 2)
    assert child.n_valid_cells == 6
    np.testing.assert_array_equal(child.values, [2.0, 2.0, 3.0, 3.0, 4.0, 4.0])


def test_touching_siblings_are_allowed_but_overlap_is_rejected() -> None:
    hierarchy = make_hierarchy()
    hierarchy.add_patch(hierarchy.root, 1, 3)
    hierarchy.add_patch(hierarchy.root, 3, 5)
    with pytest.raises(ValueError, match="overlap"):
        hierarchy.add_patch(hierarchy.root, 2, 4)


def test_active_and_stored_cell_counts_are_distinct() -> None:
    hierarchy = make_hierarchy()
    child = hierarchy.add_patch(hierarchy.root, 2, 4)
    assert hierarchy.n_stored_cells == 12
    assert hierarchy.n_active_cells == 10
    assert hierarchy.patches_at_level(1) == (child,)


def test_restriction_updates_only_covered_parent_cells() -> None:
    hierarchy = make_hierarchy()
    original = hierarchy.root.values.copy()
    child = hierarchy.add_patch(hierarchy.root, 2, 4)
    child.set_values([1.0, 3.0, 4.0, 8.0])
    hierarchy.restrict_patch(child)

    expected = original.copy()
    expected[2:4] = [2.0, 6.0]
    np.testing.assert_array_equal(hierarchy.root.values, expected)


def test_derefinement_restricts_then_removes_leaf_patch() -> None:
    hierarchy = make_hierarchy()
    child = hierarchy.add_patch(hierarchy.root, 4, 6)
    child.set_values([10.0, 12.0, 20.0, 24.0])
    hierarchy.remove_patch(child)

    assert hierarchy.root.children == []
    assert child not in hierarchy.patches
    np.testing.assert_array_equal(hierarchy.root.values[4:6], [11.0, 22.0])


def test_multiple_levels_form_a_tree() -> None:
    hierarchy = make_hierarchy()
    level_one = hierarchy.add_patch(hierarchy.root, 2, 6)
    level_two = hierarchy.add_patch(level_one, 2, 4)
    assert level_two.level == 2
    assert level_two.parent is level_one
    assert hierarchy.patches == (hierarchy.root, level_one, level_two)

