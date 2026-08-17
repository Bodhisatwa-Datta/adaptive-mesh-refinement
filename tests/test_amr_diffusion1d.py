import numpy as np
import pytest

from amr.benchmarks.diffusion import periodic_gaussian_diffusion_cell_averages
from amr.diagnostics.errors import composite_cell_average_error_norms, error_norms
from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.refinement.regrid import GradientRegridConfig, regrid_from_gradient
from amr.solvers.amr_diffusion1d import AMRExplicitDiffusion1D
from amr.solvers.diffusion1d import ExplicitDiffusion1D


def diffusion_config() -> GradientRegridConfig:
    return GradientRegridConfig(
        refine_threshold=1.0,
        derefine_threshold=0.5,
        n_buffer=4,
        merge_gap=4,
        prolongation="conservative_quadratic",
    )


def test_amr_diffusion_preserves_uniform_state_and_mass() -> None:
    grid = UniformGrid1D(0.0, 1.0, 48)
    hierarchy = AMRHierarchy1D(grid, np.full(48, 2.0))
    hierarchy.add_patch(hierarchy.root, 8, 32)
    result = AMRExplicitDiffusion1D(hierarchy, diffusivity=0.01).solve(0.05)
    for patch in hierarchy.patches:
        np.testing.assert_allclose(patch.values, 2.0, atol=2.0e-14)
    assert result.mass_error == pytest.approx(0.0, abs=2.0e-14)


def test_parabolic_subcycling_takes_ratio_squared_fine_steps() -> None:
    grid = UniformGrid1D(0.0, 1.0, 32)
    hierarchy = AMRHierarchy1D(grid, np.ones(32), refinement_ratio=2)
    child = hierarchy.add_patch(hierarchy.root, 8, 24)
    solver = AMRExplicitDiffusion1D(hierarchy, diffusivity=0.01, subcycling=True)
    result = solver.solve(solver.stable_timestep)
    assert result.n_steps == 1
    assert result.fine_steps == 4
    assert result.cell_updates == grid.n_cells + 4 * child.n_valid_cells


def test_dynamic_amr_diffusion_improves_on_base_grid() -> None:
    diffusivity = 0.01
    final_time = 0.05
    grid = UniformGrid1D(0.0, 1.0, 64)
    initial = periodic_gaussian_diffusion_cell_averages(
        grid.cell_edges, 0.0, diffusivity
    )
    exact = lambda edges: periodic_gaussian_diffusion_cell_averages(
        edges, final_time, diffusivity
    )

    base = ExplicitDiffusion1D(grid, diffusivity).solve(initial, final_time)
    base_error = error_norms(base.values, exact(grid.cell_edges)).l1

    hierarchy = AMRHierarchy1D(grid, initial)
    config = diffusion_config()
    regrid_from_gradient(hierarchy, config)
    result = AMRExplicitDiffusion1D(
        hierarchy,
        diffusivity,
        regrid_config=config,
        regrid_interval=2,
        subcycling=True,
        reflux=True,
    ).solve(final_time)
    amr_error = composite_cell_average_error_norms(hierarchy, exact).l1

    assert amr_error < base_error
    assert result.mass_error == pytest.approx(0.0, abs=3.0e-14)
    assert max(abs(event.mass_change) for event in result.regrid_events) < 3.0e-14


def test_zero_diffusivity_does_not_advance_hierarchy() -> None:
    grid = UniformGrid1D(0.0, 1.0, 16)
    hierarchy = AMRHierarchy1D(grid, np.arange(16, dtype=float))
    original = hierarchy.root.values.copy()
    result = AMRExplicitDiffusion1D(hierarchy, diffusivity=0.0).solve(2.0)
    np.testing.assert_array_equal(hierarchy.root.values, original)
    assert result.n_steps == 0
