import numpy as np
import pytest

from amr.benchmarks.advection2d import (
    periodic_gaussian_2d,
    translated_gaussian_2d,
)
from amr.diagnostics.conservation import total_mass_2d
from amr.diagnostics.errors import error_norms
from amr.grid.grid2d import UniformGrid2D
from amr.solvers.advection2d import LinearAdvection2D


@pytest.mark.parametrize(
    "velocity", [(-0.7, -0.4), (-0.7, 0.4), (0.0, 0.0), (0.7, -0.4)]
)
def test_uniform_state_is_preserved(velocity: tuple[float, float]) -> None:
    grid = UniformGrid2D(0.0, 1.0, 30, 0.0, 1.0, 24)
    initial = np.full(grid.shape, 2.5)
    result = LinearAdvection2D(grid, *velocity).solve(initial, 0.3)
    np.testing.assert_allclose(result.values, initial, atol=2.0e-14)


@pytest.mark.parametrize("velocity", [(-0.8, 0.3), (0.6, -0.5)])
def test_periodic_advection_conserves_mass(velocity: tuple[float, float]) -> None:
    grid = UniformGrid2D(0.0, 1.0, 50, 0.0, 1.0, 40)
    x, y = grid.cell_centres
    initial = periodic_gaussian_2d(x, y, centre=(0.3, 0.4))
    result = LinearAdvection2D(grid, *velocity).solve(initial, 0.37)
    assert total_mass_2d(result.values, grid) == pytest.approx(
        total_mass_2d(initial, grid), abs=3.0e-14
    )


def test_cfl_one_translates_exactly_by_one_x_cell() -> None:
    grid = UniformGrid2D(0.0, 1.0, 12, 0.0, 1.0, 8)
    initial = np.arange(grid.nx * grid.ny, dtype=float).reshape(grid.shape)
    solver = LinearAdvection2D(grid, velocity_x=1.0, velocity_y=0.0, cfl=1.0)
    numerical = solver.step(initial, grid.dx)
    np.testing.assert_array_equal(numerical, np.roll(initial, 1, axis=1))


def test_diagonal_gaussian_converges_at_first_order() -> None:
    errors = []
    velocity = (0.7, -0.4)
    final_time = 0.25
    for n_cells in (24, 48, 96, 192):
        grid = UniformGrid2D(0.0, 1.0, n_cells, 0.0, 1.0, n_cells)
        x, y = grid.cell_centres
        parameters = {"centre": (0.25, 0.35), "width": (0.09, 0.07)}
        initial = periodic_gaussian_2d(x, y, **parameters)
        result = LinearAdvection2D(grid, *velocity).solve(initial, final_time)
        exact = translated_gaussian_2d(
            x, y, final_time, velocity, **parameters
        )
        errors.append(error_norms(result.values, exact).l1)

    orders = np.log(np.asarray(errors[:-1]) / np.asarray(errors[1:])) / np.log(2.0)
    assert np.all(orders > 0.65)
    assert np.all(orders < 1.15)


def test_timestep_above_multidimensional_cfl_limit_is_rejected() -> None:
    grid = UniformGrid2D(0.0, 2.0, 20, -1.0, 1.0, 10)
    solver = LinearAdvection2D(grid, velocity_x=1.2, velocity_y=-0.5, cfl=0.8)
    expected = 0.8 / (1.2 / grid.dx + 0.5 / grid.dy)
    assert solver.stable_timestep == pytest.approx(expected)
    with pytest.raises(ValueError, match="CFL"):
        solver.step(np.ones(grid.shape), 1.01 * solver.stable_timestep)


def test_zero_velocity_reaches_requested_time_without_steps() -> None:
    grid = UniformGrid2D(0.0, 1.0, 8, 0.0, 1.0, 6)
    result = LinearAdvection2D(grid, 0.0, 0.0).solve(np.ones(grid.shape), 2.0)
    assert result.time == pytest.approx(2.0)
    assert result.n_steps == 0
