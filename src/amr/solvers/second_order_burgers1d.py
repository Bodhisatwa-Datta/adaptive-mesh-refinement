"""Second-order finite-volume solver for inviscid Burgers' equation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amr.grid.grid1d import UniformGrid1D
from amr.numerics.reconstruction import monotonized_central_slopes
from amr.solvers.burgers1d import BurgersResult, InviscidBurgers1D


@dataclass(frozen=True, slots=True)
class SecondOrderInviscidBurgers1D:
    """Solve Burgers with MC-limited MUSCL, Rusanov fluxes, and SSP-RK2."""

    grid: UniformGrid1D
    cfl: float = 0.6

    def __post_init__(self) -> None:
        if not np.isfinite(self.cfl) or not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must lie in (0, 1]")

    def stable_timestep(self, values: ArrayLike) -> float:
        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        maximum_speed = float(np.max(np.abs(field)))
        if maximum_speed == 0.0:
            return np.inf
        return self.cfl * self.grid.dx / maximum_speed

    def limited_slopes(self, values: ArrayLike) -> NDArray[np.float64]:
        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        return monotonized_central_slopes(field)

    def interface_fluxes(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return local Rusanov fluxes from reconstructed periodic states."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        slopes = self.limited_slopes(field)
        left = field + 0.5 * slopes
        right = np.roll(field - 0.5 * slopes, -1)
        interface = InviscidBurgers1D.rusanov_flux(left, right)
        return np.concatenate(([interface[-1]], interface))

    def spatial_operator(self, values: ArrayLike) -> NDArray[np.float64]:
        fluxes = self.interface_fluxes(values)
        return -(fluxes[1:] - fluxes[:-1]) / self.grid.dx

    def _validate_timestep(self, values: NDArray[np.float64], dt: float) -> None:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        limit = self.stable_timestep(values)
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, limit)
        if dt > limit + tolerance:
            raise ValueError("dt exceeds the Burgers CFL stability limit")

    def step(self, values: ArrayLike, dt: float) -> NDArray[np.float64]:
        """Advance one SSP-RK2 timestep."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        self._validate_timestep(field, dt)
        stage_one = field + dt * self.spatial_operator(field)
        return 0.5 * field + 0.5 * (
            stage_one + dt * self.spatial_operator(stage_one)
        )

    def solve(self, initial_values: ArrayLike, final_time: float) -> BurgersResult:
        values = np.array(initial_values, dtype=float, copy=True)
        self.grid.validate_field(values)
        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and non-negative")
        if final_time == 0.0 or np.max(np.abs(values)) == 0.0:
            return BurgersResult(values, float(final_time), 0)

        time = 0.0
        steps = 0
        tolerance = 16.0 * np.spacing(final_time)
        while final_time - time > tolerance:
            dt = min(self.stable_timestep(values), final_time - time)
            values = self.step(values, dt)
            time += dt
            steps += 1
        return BurgersResult(values, float(final_time), steps)
