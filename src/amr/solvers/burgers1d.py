"""Conservative finite-volume solver for inviscid Burgers' equation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amr.grid.grid1d import UniformGrid1D
from amr.numerics.boundary_conditions import fill_periodic_ghost_cells


@dataclass(frozen=True, slots=True)
class BurgersResult:
    """Final Burgers state and integration metadata."""

    values: NDArray[np.float64]
    time: float
    n_steps: int


@dataclass(frozen=True, slots=True)
class InviscidBurgers1D:
    """Solve ``u_t + (u^2/2)_x = 0`` with a local Rusanov flux."""

    grid: UniformGrid1D
    cfl: float = 0.8

    def __post_init__(self) -> None:
        if not np.isfinite(self.cfl) or not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must lie in (0, 1]")

    @staticmethod
    def physical_flux(values: ArrayLike) -> NDArray[np.float64]:
        """Return ``f(u)=u^2/2``."""

        field = np.asarray(values, dtype=float)
        return 0.5 * field**2

    @classmethod
    def rusanov_flux(
        cls, left: ArrayLike, right: ArrayLike
    ) -> NDArray[np.float64]:
        """Return the local Lax-Friedrichs/Rusanov interface flux."""

        left_state = np.asarray(left, dtype=float)
        right_state = np.asarray(right, dtype=float)
        if left_state.shape != right_state.shape:
            raise ValueError("left and right states must have equal shape")
        wave_speed = np.maximum(np.abs(left_state), np.abs(right_state))
        return 0.5 * (cls.physical_flux(left_state) + cls.physical_flux(right_state)) - (
            0.5 * wave_speed * (right_state - left_state)
        )

    def stable_timestep(self, values: ArrayLike) -> float:
        """Return the state-dependent CFL timestep."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        maximum_speed = float(np.max(np.abs(field)))
        if maximum_speed == 0.0:
            return np.inf
        return self.cfl * self.grid.dx / maximum_speed

    def interface_fluxes(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return Rusanov fluxes at all periodic cell interfaces."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        ghosted = fill_periodic_ghost_cells(field)
        return self.interface_fluxes_with_ghost_cells(field, ghosted)

    def interface_fluxes_with_ghost_cells(
        self, values: ArrayLike, ghosted_values: ArrayLike
    ) -> NDArray[np.float64]:
        """Return Rusanov fluxes using one caller-provided ghost per side."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        ghosted = np.asarray(ghosted_values, dtype=float)
        if ghosted.shape != (self.grid.n_cells + 2,):
            raise ValueError("ghosted_values must contain one ghost cell on each side")
        if not np.all(np.isfinite(ghosted)):
            raise ValueError("ghosted_values must be finite")
        if not np.array_equal(ghosted[1:-1], field):
            raise ValueError("The valid portion of ghosted_values must equal values")
        return self.rusanov_flux(ghosted[:-1], ghosted[1:])

    def step(self, values: ArrayLike, dt: float) -> NDArray[np.float64]:
        """Advance one conservative forward-Euler step."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        limit = self.stable_timestep(field)
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, limit)
        if dt > limit + tolerance:
            raise ValueError("dt exceeds the Burgers CFL stability limit")
        fluxes = self.interface_fluxes(field)
        return field - (dt / self.grid.dx) * (fluxes[1:] - fluxes[:-1])

    def step_with_ghost_cells(
        self,
        values: ArrayLike,
        ghosted_values: ArrayLike,
        dt: float,
    ) -> NDArray[np.float64]:
        """Advance one step using externally supplied boundary states."""

        field = np.asarray(values, dtype=float)
        ghosted = np.asarray(ghosted_values, dtype=float)
        fluxes = self.interface_fluxes_with_ghost_cells(field, ghosted)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        maximum_speed = float(np.max(np.abs(ghosted)))
        limit = np.inf if maximum_speed == 0.0 else self.cfl * self.grid.dx / maximum_speed
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, limit)
        if dt > limit + tolerance:
            raise ValueError("dt exceeds the Burgers CFL stability limit")
        return field - (dt / self.grid.dx) * (fluxes[1:] - fluxes[:-1])

    def solve(self, initial_values: ArrayLike, final_time: float) -> BurgersResult:
        """Integrate with a timestep recomputed from the evolving wave speed."""

        values = np.array(initial_values, dtype=float, copy=True)
        self.grid.validate_field(values)
        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and non-negative")
        if final_time == 0.0 or np.max(np.abs(values)) == 0.0:
            return BurgersResult(values, float(final_time), 0)

        time = 0.0
        n_steps = 0
        time_tolerance = 16.0 * np.spacing(final_time)
        while final_time - time > time_tolerance:
            dt = min(self.stable_timestep(values), final_time - time)
            values = self.step(values, dt)
            time += dt
            n_steps += 1
        return BurgersResult(values, float(final_time), n_steps)
