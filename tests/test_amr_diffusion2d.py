import numpy as np
import pytest

from amr.benchmarks.diffusion2d import (
    periodic_gaussian_diffusion_2d_cell_averages,
)
from amr.diagnostics.errors import (
    composite_cell_average_error_norms_2d,
    error_norms,
)
from amr.grid.grid2d import UniformGrid2D
from amr.grid.hierarchy2d import AMRHierarchy2D
from amr.refinement.regrid2d import (
    GradientRegridConfig2D,
    level_one_boxes_2d,
    regrid_from_gradient_2d,
    replace_level_one_patches_2d,
)
from amr.solvers.amr_diffusion2d import AMRExplicitDiffusion2D
from amr.solvers.diffusion2d import ExplicitDiffusion2D


def gaussian_averages(grid: UniformGrid2D) -> np.ndarray:
    return periodic_gaussian_diffusion_2d_cell_averages(
        grid.x_edges,
        grid.y_edges,
        0.0,
        0.01,
        centre=(0.4, 0.5),
        initial_width=(0.06, 0.07),
    )


def test_global_timestep_uses_fine_parabolic_limit() -> None:
    grid = UniformGrid2D(0.0, 1.0, 20, 0.0, 2.0, 16)
    hierarchy = AMRHierarchy2D(grid, np.zeros(grid.shape), refinement_ratio=2)
    hierarchy.add_patch(hierarchy.root, 4, 12, 3, 10)
    solver = AMRExplicitDiffusion2D(hierarchy, 0.03, stability_factor=0.75)
    child = hierarchy.root.children[0]
    expected = ExplicitDiffusion2D(
        child.grid, 0.03, stability_factor=0.75
    ).stable_timestep
    assert solver.stable_timestep == pytest.approx(expected)


def test_subcycling_uses_root_parabolic_limit() -> None:
    grid = UniformGrid2D(0.0, 1.0, 20, 0.0, 2.0, 16)
    hierarchy = AMRHierarchy2D(grid, np.zeros(grid.shape), refinement_ratio=2)
    hierarchy.add_patch(hierarchy.root, 4, 12, 3, 10)
    solver = AMRExplicitDiffusion2D(
        hierarchy, 0.03, stability_factor=0.75, subcycling=True
    )
    expected = ExplicitDiffusion2D(
        grid, 0.03, stability_factor=0.75
    ).stable_timestep
    assert solver.stable_timestep == pytest.approx(expected)


@pytest.mark.parametrize("subcycling", [False, True])
@pytest.mark.parametrize("reflux", [False, True])
def test_uniform_state_is_preserved(subcycling: bool, reflux: bool) -> None:
    grid = UniformGrid2D(0.0, 1.0, 24, 0.0, 1.0, 20)
    hierarchy = AMRHierarchy2D(grid, np.full(grid.shape, 2.5))
    hierarchy.add_patch(hierarchy.root, 4, 15, 3, 14)
    result = AMRExplicitDiffusion2D(
        hierarchy, 0.02, reflux=reflux, subcycling=subcycling
    ).solve(0.03)

    for patch in hierarchy.patches:
        np.testing.assert_allclose(patch.values, 2.5, atol=3.0e-14)
    assert result.mass_error == pytest.approx(0.0, abs=3.0e-14)


def test_hierarchy_without_children_matches_uniform_solver() -> None:
    grid = UniformGrid2D(0.0, 1.0, 24, 0.0, 1.0, 20)
    initial = gaussian_averages(grid)
    expected = ExplicitDiffusion2D(grid, 0.01).solve(initial, 0.05)
    hierarchy = AMRHierarchy2D(grid, initial)
    result = AMRExplicitDiffusion2D(hierarchy, 0.01).solve(0.05)

    np.testing.assert_array_equal(hierarchy.root.values, expected.values)
    assert result.n_steps == expected.n_steps


@pytest.mark.parametrize("subcycling", [False, True])
def test_refluxing_restores_composite_conservation(subcycling: bool) -> None:
    grid = UniformGrid2D(0.0, 1.0, 28, 0.0, 1.0, 28)
    hierarchy = AMRHierarchy2D(grid, gaussian_averages(grid))
    replace_level_one_patches_2d(
        hierarchy,
        [(5, 22, 6, 23)],
        prolongation="conservative_quadratic",
    )
    result = AMRExplicitDiffusion2D(
        hierarchy, 0.01, reflux=True, subcycling=subcycling
    ).solve(0.04)
    assert result.mass_error == pytest.approx(0.0, abs=4.0e-14)


def test_parabolic_subcycling_takes_ratio_squared_fine_steps() -> None:
    grid = UniformGrid2D(0.0, 1.0, 24, 0.0, 1.0, 24)
    hierarchy = AMRHierarchy2D(grid, gaussian_averages(grid), refinement_ratio=2)
    child = replace_level_one_patches_2d(
        hierarchy,
        [(4, 19, 4, 20)],
        prolongation="conservative_quadratic",
    )[0]
    result = AMRExplicitDiffusion2D(
        hierarchy, 0.01, reflux=True, subcycling=True
    ).solve(0.05)
    substeps = hierarchy.refinement_ratio**2
    assert result.fine_steps == result.n_steps * substeps
    assert result.cell_updates == result.n_steps * (
        hierarchy.root.n_valid_cells + child.n_valid_cells * substeps
    )


def test_dynamic_diffusion_expands_patch_and_improves_on_root_grid() -> None:
    grid = UniformGrid2D(0.0, 1.0, 32, 0.0, 1.0, 32)
    diffusivity = 0.01
    final_time = 0.1
    parameters = {"centre": (0.5, 0.5), "initial_width": (0.06, 0.06)}
    initial = periodic_gaussian_diffusion_2d_cell_averages(
        grid.x_edges, grid.y_edges, 0.0, diffusivity, **parameters
    )
    hierarchy = AMRHierarchy2D(grid, initial)
    config = GradientRegridConfig2D(
        1.0,
        0.5,
        n_buffer=4,
        prolongation="conservative_quadratic",
    )
    regrid_from_gradient_2d(hierarchy, config)
    initial_box = level_one_boxes_2d(hierarchy)[0]
    result = AMRExplicitDiffusion2D(
        hierarchy,
        diffusivity,
        reflux=True,
        subcycling=True,
        regrid_config=config,
    ).solve(final_time)
    final_box = level_one_boxes_2d(hierarchy)[0]

    exact = lambda x_edges, y_edges: (
        periodic_gaussian_diffusion_2d_cell_averages(
            x_edges,
            y_edges,
            final_time,
            diffusivity,
            **parameters,
        )
    )
    amr_error = composite_cell_average_error_norms_2d(hierarchy, exact).l1
    root_result = ExplicitDiffusion2D(grid, diffusivity).solve(
        initial, final_time
    )
    root_error = error_norms(
        root_result.values, exact(grid.x_edges, grid.y_edges)
    ).l1

    assert final_box[0] < initial_box[0]
    assert final_box[1] > initial_box[1]
    assert amr_error < root_error
    assert result.mass_error == pytest.approx(0.0, abs=4.0e-14)
    assert max(abs(event.mass_change) for event in result.regrid_events) < 4.0e-14
