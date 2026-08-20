import numpy as np
import pytest

from amr.benchmarks.advection2d import (
    periodic_gaussian_2d,
    translated_gaussian_2d,
)
from amr.diagnostics.conservation import composite_mass_2d
from amr.diagnostics.errors import composite_error_norms_2d, error_norms
from amr.grid.grid2d import UniformGrid2D
from amr.grid.hierarchy2d import AMRHierarchy2D
from amr.refinement.regrid2d import (
    GradientRegridConfig2D,
    level_one_boxes_2d,
    regrid_from_gradient_2d,
)
from amr.solvers.advection2d import LinearAdvection2D
from amr.solvers.amr_advection2d import AMRLinearAdvection2D


def gaussian_values(grid: UniformGrid2D) -> np.ndarray:
    x, y = grid.cell_centres
    return periodic_gaussian_2d(
        x, y, centre=(0.3, 0.4), width=(0.08, 0.07)
    )


def test_global_timestep_uses_fine_multidimensional_cfl_limit() -> None:
    grid = UniformGrid2D(0.0, 1.0, 20, 0.0, 2.0, 16)
    hierarchy = AMRHierarchy2D(grid, np.zeros(grid.shape), refinement_ratio=2)
    hierarchy.add_patch(hierarchy.root, 4, 12, 3, 10)
    solver = AMRLinearAdvection2D(hierarchy, 0.7, -0.4, cfl=0.75)
    child = hierarchy.root.children[0]
    expected = 0.75 / (0.7 / child.grid.dx + 0.4 / child.grid.dy)
    assert solver.stable_timestep == pytest.approx(expected)


def test_subcycling_uses_root_multidimensional_cfl_limit() -> None:
    grid = UniformGrid2D(0.0, 1.0, 20, 0.0, 2.0, 16)
    hierarchy = AMRHierarchy2D(grid, np.zeros(grid.shape), refinement_ratio=2)
    hierarchy.add_patch(hierarchy.root, 4, 12, 3, 10)
    solver = AMRLinearAdvection2D(
        hierarchy, 0.7, -0.4, cfl=0.75, subcycling=True
    )
    expected = 0.75 / (0.7 / grid.dx + 0.4 / grid.dy)
    assert solver.stable_timestep == pytest.approx(expected)


@pytest.mark.parametrize(
    "velocity", [(-0.7, -0.4), (-0.7, 0.4), (0.0, 0.0), (0.7, -0.4)]
)
@pytest.mark.parametrize("reflux", [False, True])
def test_uniform_state_is_preserved(
    velocity: tuple[float, float], reflux: bool
) -> None:
    grid = UniformGrid2D(0.0, 1.0, 24, 0.0, 1.0, 20)
    hierarchy = AMRHierarchy2D(grid, np.full(grid.shape, 2.5))
    hierarchy.add_patch(hierarchy.root, 4, 15, 3, 14)
    result = AMRLinearAdvection2D(
        hierarchy, *velocity, reflux=reflux
    ).solve(0.2)

    for patch in hierarchy.patches:
        np.testing.assert_allclose(patch.values, 2.5, atol=3.0e-14)
    assert result.mass_error == pytest.approx(0.0, abs=3.0e-14)


def test_hierarchy_without_children_matches_uniform_solver() -> None:
    grid = UniformGrid2D(0.0, 1.0, 24, 0.0, 1.0, 20)
    initial = gaussian_values(grid)
    expected = LinearAdvection2D(grid, -0.6, 0.3).solve(initial, 0.2)
    hierarchy = AMRHierarchy2D(grid, initial)
    result = AMRLinearAdvection2D(hierarchy, -0.6, 0.3).solve(0.2)

    np.testing.assert_array_equal(hierarchy.root.values, expected.values)
    assert result.n_steps == expected.n_steps


@pytest.mark.parametrize(
    "velocity", [(-0.7, -0.4), (-0.7, 0.4), (0.7, -0.4), (0.7, 0.4)]
)
def test_four_edge_refluxing_restores_composite_conservation(
    velocity: tuple[float, float],
) -> None:
    grid = UniformGrid2D(0.0, 1.0, 32, 0.0, 1.0, 32)
    hierarchy = AMRHierarchy2D(grid, gaussian_values(grid))
    hierarchy.add_patch(hierarchy.root, 4, 23, 6, 27)
    result = AMRLinearAdvection2D(
        hierarchy, *velocity, reflux=True
    ).solve(0.15)
    assert result.mass_error == pytest.approx(0.0, abs=3.0e-14)


def test_refluxing_handles_patches_touching_periodic_boundaries() -> None:
    grid = UniformGrid2D(0.0, 1.0, 32, 0.0, 1.0, 32)
    x, y = grid.cell_centres
    initial = periodic_gaussian_2d(x, y, centre=(0.95, 0.95))
    hierarchy = AMRHierarchy2D(grid, initial)
    hierarchy.add_patch(hierarchy.root, 0, 8, 0, 8)
    hierarchy.add_patch(hierarchy.root, 24, 32, 24, 32)
    result = AMRLinearAdvection2D(
        hierarchy, 0.6, 0.4, reflux=True
    ).solve(0.1)
    assert result.mass_error == pytest.approx(0.0, abs=4.0e-14)


def test_refluxing_skips_shared_fine_fine_interfaces() -> None:
    grid = UniformGrid2D(0.0, 1.0, 32, 0.0, 1.0, 32)
    hierarchy = AMRHierarchy2D(grid, gaussian_values(grid))
    hierarchy.add_patch(hierarchy.root, 4, 16, 6, 26)
    hierarchy.add_patch(hierarchy.root, 16, 28, 6, 26)
    result = AMRLinearAdvection2D(
        hierarchy, 0.7, -0.4, reflux=True
    ).solve(0.1)
    assert result.mass_error == pytest.approx(0.0, abs=4.0e-14)


@pytest.mark.parametrize("velocity", [(-0.7, 0.4), (0.7, -0.4)])
def test_subcycled_refluxing_is_conservative(
    velocity: tuple[float, float],
) -> None:
    grid = UniformGrid2D(0.0, 1.0, 32, 0.0, 1.0, 32)
    hierarchy = AMRHierarchy2D(grid, gaussian_values(grid))
    child = hierarchy.add_patch(hierarchy.root, 4, 23, 6, 27)
    result = AMRLinearAdvection2D(
        hierarchy, *velocity, reflux=True, subcycling=True
    ).solve(0.15)

    assert result.mass_error == pytest.approx(0.0, abs=4.0e-14)
    assert result.fine_steps == result.n_steps * hierarchy.refinement_ratio
    assert result.cell_updates == result.n_steps * (
        hierarchy.root.n_valid_cells
        + child.n_valid_cells * hierarchy.refinement_ratio
    )


def test_subcycled_solution_remains_close_to_synchronized_solution() -> None:
    grid = UniformGrid2D(0.0, 1.0, 32, 0.0, 1.0, 32)
    initial = gaussian_values(grid)
    synchronized = AMRHierarchy2D(grid, initial)
    subcycled = AMRHierarchy2D(grid, initial)
    synchronized.add_patch(synchronized.root, 4, 23, 6, 27)
    subcycled.add_patch(subcycled.root, 4, 23, 6, 27)
    AMRLinearAdvection2D(
        synchronized, 0.5, 0.3, reflux=True
    ).solve(0.1)
    AMRLinearAdvection2D(
        subcycled, 0.5, 0.3, reflux=True, subcycling=True
    ).solve(0.1)
    exact = lambda x, y: translated_gaussian_2d(
        x,
        y,
        0.1,
        (0.5, 0.3),
        centre=(0.3, 0.4),
        width=(0.08, 0.07),
    )
    synchronized_error = composite_error_norms_2d(synchronized, exact).l1
    subcycled_error = composite_error_norms_2d(subcycled, exact).l1
    assert subcycled_error < 1.5 * synchronized_error


def test_static_amr_improves_over_coarse_grid_for_covered_gaussian() -> None:
    velocity = (0.5, 0.3)
    final_time = 0.1
    grid = UniformGrid2D(0.0, 1.0, 32, 0.0, 1.0, 32)
    initial = gaussian_values(grid)
    hierarchy = AMRHierarchy2D(grid, initial)
    hierarchy.add_patch(hierarchy.root, 4, 22, 6, 25)
    result = AMRLinearAdvection2D(
        hierarchy, *velocity, reflux=True
    ).solve(final_time)

    x, y = grid.cell_centres
    exact_root = translated_gaussian_2d(
        x,
        y,
        final_time,
        velocity,
        centre=(0.3, 0.4),
        width=(0.08, 0.07),
    )
    coarse = LinearAdvection2D(grid, *velocity).solve(initial, final_time)
    coarse_error = error_norms(coarse.values, exact_root).l1
    amr_error = composite_error_norms_2d(
        hierarchy,
        lambda x, y: translated_gaussian_2d(
            x,
            y,
            final_time,
            velocity,
            centre=(0.3, 0.4),
            width=(0.08, 0.07),
        ),
    ).l1
    assert amr_error < coarse_error
    assert result.mass_error == pytest.approx(0.0, abs=3.0e-14)
    assert np.isfinite(composite_mass_2d(hierarchy))


def test_multilevel_hierarchy_is_explicitly_rejected() -> None:
    grid = UniformGrid2D(0.0, 1.0, 12, 0.0, 1.0, 12)
    hierarchy = AMRHierarchy2D(grid, np.zeros(grid.shape))
    child = hierarchy.add_patch(hierarchy.root, 2, 8, 2, 8)
    hierarchy.add_patch(child, 2, 6, 2, 6)
    with pytest.raises(NotImplementedError, match="one fine level"):
        AMRLinearAdvection2D(hierarchy, 1.0, 0.0)


def test_dynamic_regridding_tracks_diagonal_translation_conservatively() -> None:
    grid = UniformGrid2D(0.0, 1.0, 32, 0.0, 1.0, 32)
    x, y = grid.cell_centres
    profile = {"centre": (0.3, 0.3), "width": (0.06, 0.06)}
    hierarchy = AMRHierarchy2D(
        grid, periodic_gaussian_2d(x, y, **profile)
    )
    config = GradientRegridConfig2D(2.0, 1.0, n_buffer=3)
    regrid_from_gradient_2d(hierarchy, config)
    initial_box = level_one_boxes_2d(hierarchy)[0]

    result = AMRLinearAdvection2D(
        hierarchy,
        0.6,
        0.3,
        reflux=True,
        subcycling=True,
        regrid_config=config,
        regrid_interval=2,
    ).solve(0.3)
    final_box = level_one_boxes_2d(hierarchy)[0]

    assert final_box[0] > initial_box[0]
    assert final_box[2] > initial_box[2]
    assert len(result.regrid_events) >= 2
    assert result.mass_error == pytest.approx(0.0, abs=4.0e-14)
    assert max(abs(event.mass_change) for event in result.regrid_events) < 4.0e-14
