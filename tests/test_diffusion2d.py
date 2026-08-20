import numpy as np
import pytest

from amr.benchmarks.diffusion2d import (
    periodic_fourier_diffusion_2d_cell_averages,
    periodic_gaussian_diffusion_2d_cell_averages,
)
from amr.diagnostics.conservation import total_mass_2d
from amr.diagnostics.errors import error_norms
from amr.grid.grid2d import UniformGrid2D
from amr.solvers.diffusion2d import ExplicitDiffusion2D


def test_uniform_state_is_preserved() -> None:
    grid = UniformGrid2D(0.0, 1.0, 24, 0.0, 2.0, 18)
    initial = np.full(grid.shape, 3.5)
    result = ExplicitDiffusion2D(grid, diffusivity=0.1).solve(initial, 0.02)
    np.testing.assert_allclose(result.values, initial, atol=2.0e-14)


def test_zero_diffusivity_leaves_field_unchanged() -> None:
    grid = UniformGrid2D(0.0, 1.0, 16, 0.0, 1.0, 12)
    x, y = grid.cell_centres
    initial = np.sin(2.0 * np.pi * x) * np.cos(4.0 * np.pi * y)
    result = ExplicitDiffusion2D(grid, diffusivity=0.0).solve(initial, 1.0)
    np.testing.assert_array_equal(result.values, initial)
    assert result.n_steps == 0


def test_periodic_diffusion_conserves_mass() -> None:
    grid = UniformGrid2D(-1.0, 2.0, 30, 0.0, 2.0, 24)
    initial = periodic_fourier_diffusion_2d_cell_averages(
        grid.x_edges, grid.y_edges, 0.0, 0.03, mean=1.25, modes=(2, 3)
    )
    result = ExplicitDiffusion2D(grid, 0.03).solve(initial, 0.1)
    assert total_mass_2d(result.values, grid) == pytest.approx(
        total_mass_2d(initial, grid), abs=3.0e-14
    )


def test_fourier_mode_converges_at_second_order() -> None:
    errors = []
    diffusivity = 0.01
    final_time = 0.05
    for n_cells in (40, 80, 160, 320):
        grid = UniformGrid2D(0.0, 1.0, n_cells, 0.0, 1.0, n_cells)
        initial = periodic_fourier_diffusion_2d_cell_averages(
            grid.x_edges, grid.y_edges, 0.0, diffusivity
        )
        result = ExplicitDiffusion2D(grid, diffusivity).solve(initial, final_time)
        exact = periodic_fourier_diffusion_2d_cell_averages(
            grid.x_edges, grid.y_edges, final_time, diffusivity
        )
        errors.append(error_norms(result.values, exact).l1)

    orders = np.log(np.asarray(errors[:-1]) / np.asarray(errors[1:])) / np.log(2.0)
    assert np.all(orders > 1.85)
    assert np.all(orders < 2.15)


def test_stability_limit_accounts_for_both_spacings() -> None:
    grid = UniformGrid2D(0.0, 2.0, 20, -1.0, 1.0, 40)
    solver = ExplicitDiffusion2D(grid, diffusivity=0.2, stability_factor=0.8)
    expected = 0.8 / (2.0 * 0.2 * (1.0 / grid.dx**2 + 1.0 / grid.dy**2))
    assert solver.stable_timestep == pytest.approx(expected)
    with pytest.raises(ValueError, match="stability"):
        solver.step(np.ones(grid.shape), 1.01 * solver.stable_timestep)


def test_stable_update_obeys_discrete_maximum_principle() -> None:
    grid = UniformGrid2D(0.0, 1.0, 20, 0.0, 1.0, 20)
    initial = np.zeros(grid.shape)
    initial[10, 10] = 1.0
    solver = ExplicitDiffusion2D(grid, diffusivity=0.1, stability_factor=1.0)
    updated = solver.step(initial, solver.stable_timestep)
    assert np.min(updated) >= 0.0
    assert np.max(updated) <= 1.0


@pytest.mark.parametrize("modes", [(0, 0), (1, 2), (3, 5), (8, 6)])
def test_step_matches_discrete_fourier_amplification(
    modes: tuple[int, int],
) -> None:
    grid = UniformGrid2D(0.0, 1.0, 24, 0.0, 1.0, 20)
    solver = ExplicitDiffusion2D(grid, diffusivity=0.03)
    dt = 0.7 * solver.stable_timestep
    phase_x = 2.0 * np.pi * modes[0] * np.arange(grid.nx) / grid.nx
    phase_y = 2.0 * np.pi * modes[1] * np.arange(grid.ny) / grid.ny
    values = np.cos(phase_y)[:, None] * np.cos(phase_x)[None, :]
    gain = solver.fourier_amplification_factor(*modes, dt)
    np.testing.assert_allclose(solver.step(values, dt), gain * values, atol=2.0e-14)


def test_fourier_amplification_rejects_invalid_mode() -> None:
    grid = UniformGrid2D(0.0, 1.0, 12, 0.0, 1.0, 10)
    solver = ExplicitDiffusion2D(grid, diffusivity=0.02)
    with pytest.raises(ValueError, match="mode_x"):
        solver.fourier_amplification_factor(12, 1, solver.stable_timestep)


def test_analytical_fourier_profile_validates_mode_pair() -> None:
    grid = UniformGrid2D(0.0, 1.0, 8, 0.0, 1.0, 8)
    with pytest.raises(ValueError, match="two entries"):
        periodic_fourier_diffusion_2d_cell_averages(
            grid.x_edges, grid.y_edges, 0.0, 0.01, modes=(1,)
        )


def test_periodic_gaussian_2d_cell_average_mass_is_grid_independent() -> None:
    masses = []
    for n_cells in (16, 31, 64):
        grid = UniformGrid2D(0.0, 1.0, n_cells, 0.0, 1.0, n_cells)
        averages = periodic_gaussian_diffusion_2d_cell_averages(
            grid.x_edges, grid.y_edges, 0.05, 0.01
        )
        masses.append(total_mass_2d(averages, grid))
    np.testing.assert_allclose(masses, masses[0], atol=3.0e-15)
