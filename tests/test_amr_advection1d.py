import numpy as np
import pytest

from amr.benchmarks.advection import gaussian, sinusoid, translated_profile
from amr.diagnostics.conservation import composite_mass
from amr.diagnostics.errors import composite_error_norms
from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.solvers.advection1d import LinearAdvection1D
from amr.solvers.amr_advection1d import AMRLinearAdvection1D


def test_global_timestep_uses_finest_spacing() -> None:
    grid = UniformGrid1D(0.0, 1.0, 32)
    hierarchy = AMRHierarchy1D(grid, np.zeros(32), refinement_ratio=2)
    hierarchy.add_patch(hierarchy.root, 8, 16)
    solver = AMRLinearAdvection1D(hierarchy, velocity=1.5, cfl=0.75)
    assert solver.stable_timestep == pytest.approx(0.75 * grid.dx / (2.0 * 1.5))


def test_subcycling_uses_root_timestep() -> None:
    grid = UniformGrid1D(0.0, 1.0, 32)
    hierarchy = AMRHierarchy1D(grid, np.zeros(32), refinement_ratio=2)
    hierarchy.add_patch(hierarchy.root, 8, 16)
    solver = AMRLinearAdvection1D(
        hierarchy,
        velocity=1.5,
        cfl=0.75,
        subcycling=True,
    )
    assert solver.stable_timestep == pytest.approx(0.75 * grid.dx / 1.5)


@pytest.mark.parametrize("velocity", [-1.0, 1.0])
def test_uniform_state_and_composite_mass_are_preserved(velocity: float) -> None:
    grid = UniformGrid1D(0.0, 1.0, 40)
    hierarchy = AMRHierarchy1D(grid, np.full(40, 2.5))
    hierarchy.add_patch(hierarchy.root, 8, 24)
    result = AMRLinearAdvection1D(hierarchy, velocity=velocity).solve(0.2)

    for patch in hierarchy.patches:
        np.testing.assert_allclose(patch.values, 2.5, atol=2.0e-14)
    assert result.mass_error == pytest.approx(0.0, abs=2.0e-14)
    assert composite_mass(hierarchy) == pytest.approx(2.5, abs=2.0e-14)


def test_hierarchy_without_children_matches_uniform_solver() -> None:
    grid = UniformGrid1D(0.0, 1.0, 80)
    initial = sinusoid(grid.cell_centres)
    expected = LinearAdvection1D(grid, velocity=-0.7).solve(initial, 0.3)
    hierarchy = AMRHierarchy1D(grid, initial)
    result = AMRLinearAdvection1D(hierarchy, velocity=-0.7).solve(0.3)

    np.testing.assert_array_equal(hierarchy.root.values, expected.values)
    assert result.n_steps == expected.n_steps
    assert result.time == expected.time


def test_static_amr_gaussian_is_compared_with_exact_translation() -> None:
    grid = UniformGrid1D(0.0, 1.0, 64)
    profile = lambda x: gaussian(x, centre=0.3, width=0.07)
    hierarchy = AMRHierarchy1D(grid, profile(grid.cell_centres))
    hierarchy.add_patch(hierarchy.root, 8, 40)
    final_time = 0.1
    result = AMRLinearAdvection1D(hierarchy, velocity=1.0).solve(final_time)
    exact = lambda x: translated_profile(x, final_time, 1.0, profile)
    errors = composite_error_norms(hierarchy, exact)

    assert result.time == final_time
    assert errors.l1 < 3.0e-2
    assert np.isfinite(result.mass_error)


def test_multilevel_hierarchy_is_explicitly_rejected() -> None:
    grid = UniformGrid1D(0.0, 1.0, 16)
    hierarchy = AMRHierarchy1D(grid, np.zeros(16))
    child = hierarchy.add_patch(hierarchy.root, 4, 12)
    hierarchy.add_patch(child, 2, 6)
    with pytest.raises(NotImplementedError, match="one fine level"):
        AMRLinearAdvection1D(hierarchy, velocity=1.0)


def test_subcycling_reduces_root_updates_and_preserves_uniform_state() -> None:
    grid = UniformGrid1D(0.0, 1.0, 64)
    hierarchy = AMRHierarchy1D(grid, np.full(64, 3.0))
    child = hierarchy.add_patch(hierarchy.root, 8, 40)
    result = AMRLinearAdvection1D(
        hierarchy,
        velocity=1.0,
        cfl=0.8,
        subcycling=True,
    ).solve(0.1)

    assert result.n_steps == 8
    assert result.fine_steps == 16
    assert result.cell_updates == 64 * 8 + child.n_valid_cells * 16
    assert result.mass_error == pytest.approx(0.0, abs=2.0e-14)
    for patch in hierarchy.patches:
        np.testing.assert_allclose(patch.values, 3.0, atol=2.0e-14)


def test_subcycled_gaussian_remains_close_to_synchronized_solution() -> None:
    grid = UniformGrid1D(0.0, 1.0, 64)
    profile = lambda x: gaussian(x, centre=0.3, width=0.07)
    synchronized = AMRHierarchy1D(grid, profile(grid.cell_centres))
    subcycled = AMRHierarchy1D(grid, profile(grid.cell_centres))
    synchronized.add_patch(synchronized.root, 8, 40)
    subcycled.add_patch(subcycled.root, 8, 40)
    AMRLinearAdvection1D(synchronized, velocity=1.0).solve(0.1)
    AMRLinearAdvection1D(subcycled, velocity=1.0, subcycling=True).solve(0.1)

    sync_error = composite_error_norms(
        synchronized,
        lambda x: translated_profile(x, 0.1, 1.0, profile),
    ).l1
    subcycled_error = composite_error_norms(
        subcycled,
        lambda x: translated_profile(x, 0.1, 1.0, profile),
    ).l1
    assert subcycled_error < 1.5 * sync_error


@pytest.mark.parametrize("velocity", [-1.0, 1.0])
@pytest.mark.parametrize("subcycling", [False, True])
def test_refluxing_restores_composite_conservation(
    velocity: float, subcycling: bool
) -> None:
    grid = UniformGrid1D(0.0, 1.0, 64)
    initial = gaussian(grid.cell_centres, centre=0.3, width=0.07)
    hierarchy = AMRHierarchy1D(grid, initial)
    hierarchy.add_patch(hierarchy.root, 8, 40)
    result = AMRLinearAdvection1D(
        hierarchy,
        velocity=velocity,
        subcycling=subcycling,
        reflux=True,
    ).solve(0.1)

    assert result.mass_error == pytest.approx(0.0, abs=2.0e-14)


def test_refluxing_remains_conservative_across_periodic_patch_boundary() -> None:
    grid = UniformGrid1D(0.0, 1.0, 64)
    initial = gaussian(grid.cell_centres, centre=0.95, width=0.07)
    hierarchy = AMRHierarchy1D(grid, initial)
    hierarchy.add_patch(hierarchy.root, 0, 12)
    hierarchy.add_patch(hierarchy.root, 52, 64)
    result = AMRLinearAdvection1D(
        hierarchy,
        velocity=1.0,
        subcycling=True,
        reflux=True,
    ).solve(0.1)

    assert result.mass_error == pytest.approx(0.0, abs=3.0e-14)
