import numpy as np
import pytest

from amr.benchmarks.diffusion import periodic_gaussian_diffusion
from amr.diagnostics.conservation import total_mass
from amr.diagnostics.errors import error_norms
from amr.grid.grid1d import UniformGrid1D
from amr.solvers.diffusion1d import ExplicitDiffusion1D


def test_uniform_state_is_preserved() -> None:
    grid = UniformGrid1D(0.0, 1.0, 64)
    initial = np.full(64, 3.5)
    result = ExplicitDiffusion1D(grid, diffusivity=0.2).solve(initial, 0.01)
    np.testing.assert_allclose(result.values, initial, atol=2.0e-14)


def test_zero_diffusivity_leaves_field_unchanged() -> None:
    grid = UniformGrid1D(0.0, 1.0, 32)
    initial = np.sin(2.0 * np.pi * grid.cell_centres)
    result = ExplicitDiffusion1D(grid, diffusivity=0.0).solve(initial, 1.0)
    np.testing.assert_array_equal(result.values, initial)
    assert result.n_steps == 0


def test_periodic_diffusion_conserves_mass() -> None:
    grid = UniformGrid1D(0.0, 1.0, 100)
    initial = periodic_gaussian_diffusion(grid.cell_centres, 0.0, 0.01)
    result = ExplicitDiffusion1D(grid, 0.01).solve(initial, 0.05)
    assert total_mass(result.values, grid) == pytest.approx(
        total_mass(initial, grid), abs=2.0e-14
    )


def test_periodic_gaussian_converges_at_second_order() -> None:
    errors = []
    diffusivity = 0.01
    final_time = 0.05
    for n_cells in (50, 100, 200, 400):
        grid = UniformGrid1D(0.0, 1.0, n_cells)
        initial = periodic_gaussian_diffusion(grid.cell_centres, 0.0, diffusivity)
        result = ExplicitDiffusion1D(grid, diffusivity).solve(initial, final_time)
        exact = periodic_gaussian_diffusion(grid.cell_centres, final_time, diffusivity)
        errors.append(error_norms(result.values, exact).l1)
    orders = np.log(np.asarray(errors[:-1]) / np.asarray(errors[1:])) / np.log(2.0)
    assert np.all(orders > 1.85)
    assert np.all(orders < 2.15)


def test_unstable_explicit_timestep_is_rejected() -> None:
    grid = UniformGrid1D(0.0, 1.0, 40)
    solver = ExplicitDiffusion1D(grid, diffusivity=0.1, stability_factor=0.8)
    with pytest.raises(ValueError, match="stability"):
        solver.step(np.ones(40), 1.01 * solver.stable_timestep)


def test_stable_update_obeys_discrete_maximum_principle() -> None:
    grid = UniformGrid1D(0.0, 1.0, 40)
    initial = np.zeros(40)
    initial[20] = 1.0
    solver = ExplicitDiffusion1D(grid, diffusivity=0.1, stability_factor=0.8)
    updated = solver.step(initial, solver.stable_timestep)
    assert np.min(updated) >= 0.0
    assert np.max(updated) <= 1.0

