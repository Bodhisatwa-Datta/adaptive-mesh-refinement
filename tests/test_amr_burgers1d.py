import numpy as np
import pytest

from amr.benchmarks.burgers import exact_smooth_solution, smooth_periodic_profile
from amr.diagnostics.errors import composite_error_norms
from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.refinement.regrid import GradientRegridConfig, regrid_from_gradient
from amr.solvers.amr_burgers1d import AMRInviscidBurgers1D


def configuration() -> GradientRegridConfig:
    return GradientRegridConfig(
        refine_threshold=1.0,
        derefine_threshold=0.7,
        n_buffer=2,
        merge_gap=0,
    )


def make_hierarchy(n_cells: int = 64) -> tuple[AMRHierarchy1D, GradientRegridConfig]:
    grid = UniformGrid1D(0.0, 1.0, n_cells)
    hierarchy = AMRHierarchy1D(grid, smooth_periodic_profile(grid.cell_centres))
    config = configuration()
    regrid_from_gradient(hierarchy, config)
    return hierarchy, config


def test_amr_burgers_preserves_uniform_state() -> None:
    grid = UniformGrid1D(0.0, 1.0, 48)
    hierarchy = AMRHierarchy1D(grid, np.full(48, -0.75))
    hierarchy.add_patch(hierarchy.root, 8, 32)
    result = AMRInviscidBurgers1D(hierarchy).solve(0.2)
    for patch in hierarchy.patches:
        np.testing.assert_allclose(patch.values, -0.75, atol=2.0e-14)
    assert result.mass_error == pytest.approx(0.0, abs=2.0e-14)


def test_dynamic_amr_matches_smooth_pre_shock_solution() -> None:
    hierarchy, config = make_hierarchy()
    result = AMRInviscidBurgers1D(
        hierarchy,
        regrid_config=config,
        regrid_interval=2,
        subcycling=True,
        reflux=True,
    ).solve(0.2)
    errors = composite_error_norms(
        hierarchy,
        lambda x: exact_smooth_solution(x, 0.2),
    )
    assert errors.l1 < 1.3e-3
    assert result.mass_error == pytest.approx(0.0, abs=3.0e-14)


def test_refluxed_shock_evolution_is_conservative_and_bounded() -> None:
    hierarchy, config = make_hierarchy()
    result = AMRInviscidBurgers1D(
        hierarchy,
        regrid_config=config,
        regrid_interval=2,
        subcycling=True,
        reflux=True,
    ).solve(1.0)
    minimum = min(float(patch.values.min()) for patch in hierarchy.patches)
    maximum = max(float(patch.values.max()) for patch in hierarchy.patches)

    assert result.mass_error == pytest.approx(0.0, abs=4.0e-14)
    assert minimum >= 0.3 - 2.0e-12
    assert maximum <= 0.7 + 2.0e-12
    assert len(result.regrid_events) > 0
    assert max(abs(event.mass_change) for event in result.regrid_events) < 3.0e-14


def test_amr_burgers_uses_state_dependent_cfl() -> None:
    hierarchy, _ = make_hierarchy()
    solver = AMRInviscidBurgers1D(hierarchy, cfl=0.8, subcycling=True)
    maximum_speed = max(float(np.max(np.abs(patch.values))) for patch in hierarchy.patches)
    assert solver.stable_timestep == pytest.approx(
        0.8 * hierarchy.root.grid.dx / maximum_speed
    )
