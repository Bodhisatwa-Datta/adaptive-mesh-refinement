import numpy as np
import pytest

from amr.benchmarks.burgers import (
    exact_smooth_solution,
    shock_formation_time,
    smooth_periodic_profile,
)
from amr.diagnostics.conservation import total_mass
from amr.diagnostics.errors import error_norms
from amr.grid.grid1d import UniformGrid1D
from amr.solvers.burgers1d import InviscidBurgers1D


def test_rusanov_flux_is_consistent_with_physical_flux() -> None:
    states = np.array([-2.0, -0.5, 0.0, 1.5])
    np.testing.assert_array_equal(
        InviscidBurgers1D.rusanov_flux(states, states),
        0.5 * states**2,
    )


@pytest.mark.parametrize("state", [-1.5, 0.0, 2.0])
def test_uniform_state_is_preserved(state: float) -> None:
    grid = UniformGrid1D(0.0, 1.0, 64)
    initial = np.full(grid.n_cells, state)
    result = InviscidBurgers1D(grid).solve(initial, 0.25)
    np.testing.assert_allclose(result.values, initial, atol=2.0e-14)


def test_periodic_burgers_update_conserves_mass() -> None:
    grid = UniformGrid1D(0.0, 1.0, 100)
    initial = smooth_periodic_profile(grid.cell_centres)
    result = InviscidBurgers1D(grid).solve(initial, 0.3)
    assert total_mass(result.values, grid) == pytest.approx(
        total_mass(initial, grid), abs=2.0e-14
    )


def test_smooth_pre_shock_solution_converges_at_first_order() -> None:
    errors = []
    for n_cells in (50, 100, 200, 400):
        grid = UniformGrid1D(0.0, 1.0, n_cells)
        initial = smooth_periodic_profile(grid.cell_centres)
        result = InviscidBurgers1D(grid).solve(initial, 0.2)
        exact = exact_smooth_solution(grid.cell_centres, 0.2)
        errors.append(error_norms(result.values, exact).l1)

    orders = np.log(np.asarray(errors[:-1]) / np.asarray(errors[1:])) / np.log(2.0)
    assert np.all(orders > 0.9)
    assert np.all(orders < 1.15)


def test_smooth_exact_solution_rejects_post_shock_time() -> None:
    with pytest.raises(ValueError, match="before shock"):
        exact_smooth_solution([0.2, 0.4], shock_formation_time())


def test_unstable_timestep_is_rejected() -> None:
    grid = UniformGrid1D(0.0, 1.0, 32)
    values = smooth_periodic_profile(grid.cell_centres)
    solver = InviscidBurgers1D(grid, cfl=0.8)
    with pytest.raises(ValueError, match="CFL"):
        solver.step(values, 1.01 * solver.stable_timestep(values))

