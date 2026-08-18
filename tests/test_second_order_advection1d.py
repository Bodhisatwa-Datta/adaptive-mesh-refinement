import numpy as np
import pytest

from amr.benchmarks.advection import sinusoid, square_pulse, translated_profile
from amr.diagnostics.conservation import total_mass
from amr.diagnostics.errors import error_norms
from amr.diagnostics.variation import total_variation
from amr.grid.grid1d import UniformGrid1D
from amr.solvers.second_order_advection1d import SecondOrderLinearAdvection1D


@pytest.mark.parametrize("velocity", [-1.0, 0.0, 1.0])
def test_second_order_advection_preserves_uniform_state(velocity: float) -> None:
    grid = UniformGrid1D(0.0, 1.0, 64)
    initial = np.full(grid.n_cells, 1.75)
    result = SecondOrderLinearAdvection1D(grid, velocity).solve(initial, 0.3)
    np.testing.assert_allclose(result.values, initial, atol=2.0e-14)


@pytest.mark.parametrize("velocity", [-0.8, 1.2])
def test_second_order_advection_conserves_mass(velocity: float) -> None:
    grid = UniformGrid1D(0.0, 1.0, 100)
    initial = sinusoid(grid.cell_centres)
    result = SecondOrderLinearAdvection1D(grid, velocity).solve(initial, 0.37)
    assert total_mass(result.values, grid) == pytest.approx(
        total_mass(initial, grid), abs=3.0e-14
    )


def test_smooth_advection_converges_at_second_order() -> None:
    errors = []
    for n_cells in (40, 80, 160, 320):
        grid = UniformGrid1D(0.0, 1.0, n_cells)
        initial = sinusoid(grid.cell_centres)
        result = SecondOrderLinearAdvection1D(grid, 1.0, cfl=0.6).solve(initial, 0.5)
        exact = translated_profile(grid.cell_centres, 0.5, 1.0, sinusoid)
        errors.append(error_norms(result.values, exact).l1)
    orders = np.log(np.asarray(errors[:-1]) / np.asarray(errors[1:])) / np.log(2.0)
    assert np.all(orders > 1.75)
    assert np.all(orders < 2.2)


def test_limiter_keeps_square_pulse_bounded() -> None:
    grid = UniformGrid1D(0.0, 1.0, 200)
    initial = square_pulse(grid.cell_centres, centre=0.3, width=0.2)
    result = SecondOrderLinearAdvection1D(grid, 1.0, cfl=0.8).solve(initial, 0.2)
    assert np.min(result.values) >= -2.0e-14
    assert np.max(result.values) <= 1.0 + 2.0e-14
    assert total_variation(result.values) <= total_variation(initial) + 2.0e-13


def test_second_order_solver_rejects_unstable_timestep() -> None:
    grid = UniformGrid1D(0.0, 1.0, 20)
    solver = SecondOrderLinearAdvection1D(grid, 1.0, cfl=0.8)
    with pytest.raises(ValueError, match="CFL"):
        solver.step(np.ones(grid.n_cells), 1.01 * solver.stable_timestep)
