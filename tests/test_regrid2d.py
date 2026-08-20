import numpy as np
import pytest

from amr.benchmarks.advection2d import periodic_gaussian_2d
from amr.diagnostics.conservation import composite_mass_2d
from amr.grid.grid2d import UniformGrid2D
from amr.grid.hierarchy2d import AMRHierarchy2D
from amr.refinement.regrid2d import (
    GradientRegridConfig2D,
    level_one_boxes_2d,
    regrid_from_gradient_2d,
    replace_level_one_patches_2d,
)


def make_zero_hierarchy() -> AMRHierarchy2D:
    grid = UniformGrid2D(0.0, 1.0, 4, 0.0, 1.0, 4)
    return AMRHierarchy2D(grid, np.zeros(grid.shape), refinement_ratio=2)


def test_rectangular_replacement_preserves_mass_and_overlap_data() -> None:
    hierarchy = make_zero_hierarchy()
    old_values = np.arange(16, dtype=float).reshape(4, 4)
    hierarchy.add_patch(hierarchy.root, 1, 3, 1, 3, values=old_values)
    mass_before = composite_mass_2d(hierarchy)
    new_patch = replace_level_one_patches_2d(hierarchy, [(2, 4, 2, 4)])[0]

    assert composite_mass_2d(hierarchy) == pytest.approx(mass_before, abs=2.0e-15)
    np.testing.assert_array_equal(new_patch.values[:2, :2], old_values[2:4, 2:4])
    np.testing.assert_array_equal(new_patch.values[:2, 2:], np.zeros((2, 2)))


def test_derefining_all_cells_restricts_conservatively() -> None:
    hierarchy = make_zero_hierarchy()
    values = np.arange(16, dtype=float).reshape(4, 4)
    hierarchy.add_patch(hierarchy.root, 1, 3, 1, 3, values=values)
    mass_before = composite_mass_2d(hierarchy)
    replace_level_one_patches_2d(hierarchy, [])

    assert level_one_boxes_2d(hierarchy) == ()
    assert composite_mass_2d(hierarchy) == pytest.approx(mass_before, abs=2.0e-15)
    expected = values.reshape(2, 2, 2, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(hierarchy.root.values[1:3, 1:3], expected)


def test_uniform_solution_removes_unneeded_patch() -> None:
    grid = UniformGrid2D(0.0, 1.0, 12, 0.0, 1.0, 10)
    hierarchy = AMRHierarchy2D(grid, np.ones(grid.shape))
    hierarchy.add_patch(hierarchy.root, 2, 9, 2, 8)
    report = regrid_from_gradient_2d(
        hierarchy, GradientRegridConfig2D(1.0, 0.5, n_buffer=2)
    )

    assert report.changed
    assert report.new_boxes == ()
    assert report.mass_change == pytest.approx(0.0, abs=2.0e-15)


def test_hysteresis_retains_existing_box_below_refinement_threshold() -> None:
    grid = UniformGrid2D(0.0, 1.0, 12, 0.0, 1.0, 12)
    x, y = grid.cell_centres
    values = np.sin(2.0 * np.pi * x) + np.sin(2.0 * np.pi * y)
    hierarchy = AMRHierarchy2D(grid, values)
    hierarchy.add_patch(hierarchy.root, 0, 12, 0, 12)
    report = regrid_from_gradient_2d(
        hierarchy,
        GradientRegridConfig2D(20.0, 1.0, n_buffer=0),
    )
    assert not report.changed
    assert report.new_boxes == ((0, 12, 0, 12),)


def test_overlapping_replacement_boxes_are_rejected() -> None:
    hierarchy = make_zero_hierarchy()
    with pytest.raises(ValueError, match="overlap"):
        replace_level_one_patches_2d(
            hierarchy, [(0, 3, 0, 3), (2, 4, 2, 4)]
        )


def test_hysteresis_threshold_order_is_validated() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        GradientRegridConfig2D(1.0, 2.0)


def test_regridding_builds_separate_boxes_for_separated_features() -> None:
    grid = UniformGrid2D(0.0, 1.0, 32, 0.0, 1.0, 32)
    x, y = grid.cell_centres
    values = periodic_gaussian_2d(
        x, y, centre=(0.25, 0.3), width=(0.05, 0.05)
    ) + periodic_gaussian_2d(
        x, y, centre=(0.75, 0.7), width=(0.05, 0.05)
    )
    hierarchy = AMRHierarchy2D(grid, values)
    report = regrid_from_gradient_2d(
        hierarchy,
        GradientRegridConfig2D(
            2.0, 1.0, n_buffer=1, merge_gap=1, periodic=False
        ),
    )
    assert len(report.new_boxes) == 2
    assert len(hierarchy.root.children) == 2
    assert report.mass_change == pytest.approx(0.0, abs=3.0e-15)
