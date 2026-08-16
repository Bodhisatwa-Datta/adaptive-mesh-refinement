"""First-order finite-volume solver for constant-coefficient linear advection."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amr.grid.grid1d import UniformGrid1D
from amr.numerics.boundary_conditions import fill_periodic_ghost_cells


@dataclass(frozen=True, slots=True)
class AdvectionResult:
    """Final state and integration metadata."""

    values: NDArray[np.float64]
    time: float
    n_steps: int


@dataclass(frozen=True, slots=True)
class LinearAdvection1D:
    """Solve ``u_t + a u_x = 0`` using an upwind finite-volume flux."""

    grid: UniformGrid1D
    velocity: float
    cfl: float = 0.8

    def __post_init__(self) -> None:
        if not np.isfinite(self.velocity):
            raise ValueError("velocity must be finite")
        if not np.isfinite(self.cfl) or not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must lie in (0, 1] for first-order upwinding")

    @property
    def stable_timestep(self) -> float:
        """Largest timestep allowed by the configured CFL number."""

        if self.velocity == 0.0:
            return np.inf
        return self.cfl * self.grid.dx / abs(self.velocity)

    def spatial_operator(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return the conservative semi-discrete finite-volume operator."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        ghosted = fill_periodic_ghost_cells(field)

        if self.velocity >= 0.0:
            fluxes = self.velocity * ghosted[:-1]
        else:
            fluxes = self.velocity * ghosted[1:]
        return -(fluxes[1:] - fluxes[:-1]) / self.grid.dx

    def step(self, values: ArrayLike, dt: float) -> NDArray[np.float64]:
        """Advance one forward-Euler step, rejecting CFL-unstable timesteps."""

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, self.stable_timestep)
        if dt > self.stable_timestep + tolerance:
            raise ValueError("dt exceeds the configured CFL stability limit")
        field = np.asarray(values, dtype=float)
        return field + dt * self.spatial_operator(field)

    def solve(self, initial_values: ArrayLike, final_time: float) -> AdvectionResult:
        """Integrate from time zero to ``final_time`` with an exact final step."""

        values = np.array(initial_values, dtype=float, copy=True)
        self.grid.validate_field(values)
        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and non-negative")
        if final_time == 0.0 or self.velocity == 0.0:
            return AdvectionResult(values=values, time=float(final_time), n_steps=0)

        time = 0.0
        n_steps = 0
        while time < final_time:
            dt = min(self.stable_timestep, final_time - time)
            values = self.step(values, dt)
            time += dt
            n_steps += 1
        return AdvectionResult(values=values, time=float(final_time), n_steps=n_steps)

