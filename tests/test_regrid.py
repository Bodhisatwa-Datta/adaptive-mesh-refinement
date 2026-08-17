import numpy as np
import pytest

from amr.benchmarks.advection import gaussian
from amr.diagnostics.conservation import composite_mass
from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.refinement.regrid import (
    GradientRegridConfig,
    level_one_regions,
    regrid_from_gradient,
    replace_level_one_patches,
)
from amr.solvers.amr_advection1d import AMRLinearAdvection1D


def test_patch_replacement_preserves_mass_and_overlap_data() -> None:
    grid = UniformGrid1D(0.0, 1.0, 8)
    hierarchy = AMRHierarchy1D(grid, np.zeros(8), refinement_ratio=2)
    hierarchy.add_patch(
        hierarchy.root,
        2,
        5,
        values=[1.0, 3.0, 4.0, 8.0, 10.0, 14.0],
    )
    mass_before = composite_mass(hierarchy)
    new_patch = replace_level_one_patches(hierarchy, [(3, 6)])[0]

    assert composite_mass(hierarchy) == pytest.approx(mass_before, abs=2.0e-15)
    np.testing.assert_array_equal(new_patch.values, [4.0, 8.0, 10.0, 14.0, 0.0, 0.0])


def test_derefining_all_cells_restricts_conservatively() -> None:
    grid = UniformGrid1D(0.0, 1.0, 8)
    hierarchy = AMRHierarchy1D(grid, np.zeros(8))
    hierarchy.add_patch(hierarchy.root, 2, 4, values=[1.0, 3.0, 5.0, 7.0])
    mass_before = composite_mass(hierarchy)
    replace_level_one_patches(hierarchy, [])

    assert level_one_regions(hierarchy) == ()
    assert composite_mass(hierarchy) == pytest.approx(mass_before, abs=2.0e-15)
    np.testing.assert_array_equal(hierarchy.root.values[2:4], [2.0, 6.0])


def test_uniform_solution_removes_unneeded_patch() -> None:
    grid = UniformGrid1D(0.0, 1.0, 16)
    hierarchy = AMRHierarchy1D(grid, np.ones(16))
    hierarchy.add_patch(hierarchy.root, 4, 12)
    config = GradientRegridConfig(1.0, 0.5, n_buffer=2)
    report = regrid_from_gradient(hierarchy, config)

    assert report.changed
    assert report.new_regions == ()
    assert report.mass_change == pytest.approx(0.0, abs=2.0e-15)


def test_hysteresis_threshold_order_is_validated() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        GradientRegridConfig(refine_threshold=1.0, derefine_threshold=2.0)


def test_hysteresis_retains_existing_patch_below_refinement_threshold() -> None:
    grid = UniformGrid1D(0.0, 1.0, 16)
    values = np.sin(2.0 * np.pi * grid.cell_centres)
    hierarchy = AMRHierarchy1D(grid, values)
    hierarchy.add_patch(hierarchy.root, 0, 3)
    config = GradientRegridConfig(
        refine_threshold=10.0,
        derefine_threshold=2.0,
        n_buffer=0,
    )
    report = regrid_from_gradient(hierarchy, config)

    assert not report.changed
    assert report.new_regions == ((0, 3),)


def test_dynamic_regridding_tracks_translated_gaussian_conservatively() -> None:
    grid = UniformGrid1D(0.0, 1.0, 64)
    initial = gaussian(grid.cell_centres, centre=0.25, width=0.07)
    hierarchy = AMRHierarchy1D(grid, initial)
    config = GradientRegridConfig(
        refine_threshold=3.0,
        derefine_threshold=1.5,
        n_buffer=6,
        merge_gap=4,
    )
    regrid_from_gradient(hierarchy, config)
    initial_start = level_one_regions(hierarchy)[0][0]

    result = AMRLinearAdvection1D(
        hierarchy,
        velocity=1.0,
        regrid_config=config,
        regrid_interval=4,
    ).solve(0.3)
    final_start = level_one_regions(hierarchy)[0][0]

    assert final_start > initial_start
    assert len(result.regrid_events) >= 2
    assert max(abs(event.mass_change) for event in result.regrid_events) < 2.0e-14
    assert result.cell_updates > 0
