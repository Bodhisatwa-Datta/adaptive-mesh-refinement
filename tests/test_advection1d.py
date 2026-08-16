import numpy as np
import pytest

from amr.benchmarks.advection import gaussian, sinusoid, square_pulse, translated_profile
from amr.diagnostics.conservation import total_mass
from amr.diagnostics.errors import error_norms
from amr.grid.grid1d import UniformGrid1D
from amr.solvers.advection1d import LinearAdvection1D


@pytest.mark.parametrize("velocity", [-1.25, 0.0, 0.75])
def test_uniform_state_is_preserved(velocity: float) -> None:
    grid = UniformGrid1D(0.0, 1.0, 64)
    initial = np.full(grid.n_cells, 2.5)
    result = LinearAdvection1D(grid, velocity=velocity).solve(initial, 0.37)
    np.testing.assert_allclose(result.values, initial, atol=2.0e-14)


@pytest.mark.parametrize("velocity", [-1.0, 1.0])
def test_periodic_advection_conserves_mass(velocity: float) -> None:
    grid = UniformGrid1D(0.0, 1.0, 100)
    initial = gaussian(grid.cell_centres, centre=0.3)
    result = LinearAdvection1D(grid, velocity=velocity).solve(initial, 0.43)
    assert total_mass(result.values, grid) == pytest.approx(total_mass(initial, grid), abs=2.0e-14)


@pytest.mark.parametrize("velocity", [-0.8, 1.2])
def test_smooth_profile_moves_in_both_directions(velocity: float) -> None:
    grid = UniformGrid1D(0.0, 1.0, 200)
    initial = sinusoid(grid.cell_centres)
    final_time = 0.2
    numerical = LinearAdvection1D(grid, velocity=velocity).solve(initial, final_time).values
    exact = translated_profile(grid.cell_centres, final_time, velocity, sinusoid)
    assert error_norms(numerical, exact).l1 < 1.5e-2


def test_square_pulse_is_advected_monotonically() -> None:
    grid = UniformGrid1D(0.0, 1.0, 200)
    profile = lambda x: square_pulse(x, centre=0.3, width=0.2)
    initial = profile(grid.cell_centres)
    final_time = 0.2
    numerical = LinearAdvection1D(grid, velocity=1.0).solve(initial, final_time).values
    exact = translated_profile(grid.cell_centres, final_time, 1.0, profile)
    assert error_norms(numerical, exact).l1 < 3.0e-2
    assert np.min(numerical) >= 0.0
    assert np.max(numerical) <= 1.0


def test_gaussian_converges_at_first_order() -> None:
    errors = []
    profile = lambda x: gaussian(x, centre=0.25, width=0.07)
    for n_cells in (50, 100, 200, 400):
        grid = UniformGrid1D(0.0, 1.0, n_cells)
        initial = profile(grid.cell_centres)
        result = LinearAdvection1D(grid, velocity=1.0, cfl=0.8).solve(initial, 0.5)
        exact = translated_profile(grid.cell_centres, 0.5, 1.0, profile)
        errors.append(error_norms(result.values, exact).l1)

    observed_orders = np.log(np.asarray(errors[:-1]) / np.asarray(errors[1:])) / np.log(2.0)
    assert np.all(observed_orders > 0.75)
    assert np.all(observed_orders < 1.15)


def test_timestep_above_cfl_limit_is_rejected() -> None:
    grid = UniformGrid1D(0.0, 1.0, 20)
    solver = LinearAdvection1D(grid, velocity=1.0, cfl=0.8)
    with pytest.raises(ValueError, match="CFL"):
        solver.step(np.ones(grid.n_cells), 1.01 * solver.stable_timestep)
