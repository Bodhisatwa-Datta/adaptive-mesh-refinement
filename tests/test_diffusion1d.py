import numpy as np
import pytest

from amr.benchmarks.diffusion import (
    periodic_gaussian_diffusion_cell_averages,
    periodic_sine_diffusion_cell_averages,
)
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
    initial = periodic_gaussian_diffusion_cell_averages(grid.cell_edges, 0.0, 0.01)
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
        initial = periodic_gaussian_diffusion_cell_averages(
            grid.cell_edges, 0.0, diffusivity
        )
        result = ExplicitDiffusion1D(grid, diffusivity).solve(initial, final_time)
        exact = periodic_gaussian_diffusion_cell_averages(
            grid.cell_edges, final_time, diffusivity
        )
        errors.append(error_norms(result.values, exact).l1)
    orders = np.log(np.asarray(errors[:-1]) / np.asarray(errors[1:])) / np.log(2.0)
    assert np.all(orders > 1.85)
    assert np.all(orders < 2.15)


def test_periodic_gaussian_cell_average_mass_is_grid_independent() -> None:
    masses = []
    for n_cells in (16, 63, 256):
        grid = UniformGrid1D(0.0, 1.0, n_cells)
        averages = periodic_gaussian_diffusion_cell_averages(
            grid.cell_edges, 0.05, 0.01
        )
        masses.append(total_mass(averages, grid))
    np.testing.assert_allclose(masses, masses[0], atol=2.0e-15)


def test_periodic_sine_cell_averages_preserve_the_mean() -> None:
    grid = UniformGrid1D(-1.0, 2.0, 63)
    averages = periodic_sine_diffusion_cell_averages(
        grid.cell_edges,
        0.2,
        0.03,
        mean=1.25,
        mode=3,
        x_min=-1.0,
        x_max=2.0,
    )
    assert total_mass(averages, grid) == pytest.approx(1.25 * 3.0, abs=2.0e-14)


def test_periodic_sine_diffusion_converges_at_second_order() -> None:
    errors = []
    diffusivity = 0.01
    final_time = 0.05
    for n_cells in (40, 80, 160, 320):
        grid = UniformGrid1D(0.0, 1.0, n_cells)
        initial = periodic_sine_diffusion_cell_averages(
            grid.cell_edges, 0.0, diffusivity, mode=2
        )
        result = ExplicitDiffusion1D(grid, diffusivity).solve(initial, final_time)
        exact = periodic_sine_diffusion_cell_averages(
            grid.cell_edges, final_time, diffusivity, mode=2
        )
        errors.append(error_norms(result.values, exact).l1)
    orders = np.log(np.asarray(errors[:-1]) / np.asarray(errors[1:])) / np.log(2.0)
    assert np.all(orders > 1.9)
    assert np.all(orders < 2.1)


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


@pytest.mark.parametrize("mode", [1, 2, 7, 16])
def test_diffusion_step_matches_discrete_fourier_amplification(mode: int) -> None:
    grid = UniformGrid1D(0.0, 1.0, 32)
    solver = ExplicitDiffusion1D(grid, diffusivity=0.03, stability_factor=0.8)
    dt = 0.7 * solver.stable_timestep
    phase = 2.0 * np.pi * mode * np.arange(grid.n_cells) / grid.n_cells
    values = np.sin(phase)
    expected = solver.fourier_amplification_factor(mode, dt) * values
    np.testing.assert_allclose(solver.step(values, dt), expected, atol=2.0e-14)


def test_fourier_amplification_validates_mode_and_timestep() -> None:
    solver = ExplicitDiffusion1D(UniformGrid1D(0.0, 1.0, 16), diffusivity=0.02)
    assert solver.fourier_amplification_factor(0, solver.stable_timestep) == 1.0
    with pytest.raises(ValueError, match="mode"):
        solver.fourier_amplification_factor(16, solver.stable_timestep)
    with pytest.raises(ValueError, match="stability"):
        solver.fourier_amplification_factor(1, 1.01 * solver.stable_timestep)
