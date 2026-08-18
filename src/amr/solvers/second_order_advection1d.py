"""Second-order finite-volume solver for constant-coefficient advection."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amr.grid.grid1d import UniformGrid1D
from amr.numerics.reconstruction import monotonized_central_slopes
from amr.solvers.advection1d import AdvectionResult


@dataclass(frozen=True, slots=True)
class SecondOrderLinearAdvection1D:
    """Solve linear advection with MC-limited MUSCL and SSP-RK2."""

    grid: UniformGrid1D
    velocity: float
    cfl: float = 0.8

    def __post_init__(self) -> None:
        if not np.isfinite(self.velocity):
            raise ValueError("velocity must be finite")
        if not np.isfinite(self.cfl) or not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must lie in (0, 1] for limited MUSCL advection")

    @property
    def stable_timestep(self) -> float:
        if self.velocity == 0.0:
            return np.inf
        return self.cfl * self.grid.dx / abs(self.velocity)

    def limited_slopes(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return monotonized-central slopes in cell-average units."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        return monotonized_central_slopes(field)

    def interface_fluxes(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return periodic MUSCL fluxes at all ``N+1`` cell interfaces."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        slopes = self.limited_slopes(field)
        left_states = field + 0.5 * slopes
        right_states = np.roll(field - 0.5 * slopes, -1)
        interface = self.velocity * (
            left_states if self.velocity >= 0.0 else right_states
        )
        return np.concatenate(([interface[-1]], interface))

    def spatial_operator(self, values: ArrayLike) -> NDArray[np.float64]:
        fluxes = self.interface_fluxes(values)
        return -(fluxes[1:] - fluxes[:-1]) / self.grid.dx

    def _validate_timestep(self, dt: float) -> None:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, self.stable_timestep)
        if dt > self.stable_timestep + tolerance:
            raise ValueError("dt exceeds the configured CFL stability limit")

    def step(self, values: ArrayLike, dt: float) -> NDArray[np.float64]:
        """Advance one SSP-RK2 step."""

        self._validate_timestep(dt)
        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        stage_one = field + dt * self.spatial_operator(field)
        return 0.5 * field + 0.5 * (
            stage_one + dt * self.spatial_operator(stage_one)
        )

    def solve(self, initial_values: ArrayLike, final_time: float) -> AdvectionResult:
        """Integrate to an exact final time with CFL-controlled SSP-RK2 steps."""

        values = np.array(initial_values, dtype=float, copy=True)
        self.grid.validate_field(values)
        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and non-negative")
        if final_time == 0.0 or self.velocity == 0.0:
            return AdvectionResult(values, float(final_time), 0)

        time = 0.0
        steps = 0
        tolerance = 16.0 * np.spacing(final_time)
        while final_time - time > tolerance:
            dt = min(self.stable_timestep, final_time - time)
            values = self.step(values, dt)
            time += dt
            steps += 1
        return AdvectionResult(values, float(final_time), steps)
