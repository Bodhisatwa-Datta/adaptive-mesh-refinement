import numpy as np
import pytest

from amr.benchmarks.burgers import exact_smooth_solution, smooth_periodic_profile
from amr.diagnostics.conservation import total_mass
from amr.diagnostics.errors import error_norms
from amr.diagnostics.variation import total_variation
from amr.grid.grid1d import UniformGrid1D
from amr.solvers.second_order_burgers1d import SecondOrderInviscidBurgers1D


@pytest.mark.parametrize("state", [-1.2, 0.0, 1.7])
def test_second_order_burgers_preserves_uniform_state(state: float) -> None:
    grid = UniformGrid1D(0.0, 1.0, 64)
    initial = np.full(grid.n_cells, state)
    result = SecondOrderInviscidBurgers1D(grid).solve(initial, 0.2)
    np.testing.assert_allclose(result.values, initial, atol=2.0e-14)


def test_second_order_burgers_conserves_mass() -> None:
    grid = UniformGrid1D(0.0, 1.0, 100)
    initial = smooth_periodic_profile(grid.cell_centres)
    result = SecondOrderInviscidBurgers1D(grid).solve(initial, 0.3)
    assert total_mass(result.values, grid) == pytest.approx(
        total_mass(initial, grid), abs=3.0e-14
    )


def test_second_order_burgers_pre_shock_convergence() -> None:
    errors = []
    for n_cells in (50, 100, 200, 400):
        grid = UniformGrid1D(0.0, 1.0, n_cells)
        initial = smooth_periodic_profile(grid.cell_centres)
        result = SecondOrderInviscidBurgers1D(grid, cfl=0.6).solve(initial, 0.2)
        exact = exact_smooth_solution(grid.cell_centres, 0.2)
        errors.append(error_norms(result.values, exact).l1)
    orders = np.log(np.asarray(errors[:-1]) / np.asarray(errors[1:])) / np.log(2.0)
    assert np.all(orders > 1.7)
    assert np.all(orders < 2.2)


def test_second_order_burgers_remains_bounded_through_shock() -> None:
    grid = UniformGrid1D(0.0, 1.0, 200)
    initial = smooth_periodic_profile(grid.cell_centres)
    result = SecondOrderInviscidBurgers1D(grid, cfl=0.6).solve(initial, 1.0)
    assert np.min(result.values) >= np.min(initial) - 2.0e-14
    assert np.max(result.values) <= np.max(initial) + 2.0e-14
    assert total_variation(result.values) <= total_variation(initial) + 2.0e-13


def test_second_order_burgers_rejects_unstable_timestep() -> None:
    grid = UniformGrid1D(0.0, 1.0, 32)
    values = smooth_periodic_profile(grid.cell_centres)
    solver = SecondOrderInviscidBurgers1D(grid, cfl=0.6)
    with pytest.raises(ValueError, match="CFL"):
        solver.step(values, 1.01 * solver.stable_timestep(values))
