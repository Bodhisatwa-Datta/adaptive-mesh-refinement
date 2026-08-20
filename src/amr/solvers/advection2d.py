"""First-order finite-volume solver for periodic linear advection in 2D."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amr.grid.grid2d import UniformGrid2D
from amr.numerics.boundary_conditions import fill_periodic_ghost_cells_2d


@dataclass(frozen=True, slots=True)
class AdvectionResult2D:
    """Final two-dimensional state and integration metadata."""

    values: NDArray[np.float64]
    time: float
    n_steps: int


@dataclass(frozen=True, slots=True)
class LinearAdvection2D:
    """Solve ``u_t + a u_x + b u_y = 0`` with donor-cell upwinding."""

    grid: UniformGrid2D
    velocity_x: float
    velocity_y: float
    cfl: float = 0.8

    def __post_init__(self) -> None:
        if not np.all(np.isfinite((self.velocity_x, self.velocity_y))):
            raise ValueError("velocity components must be finite")
        if not np.isfinite(self.cfl) or not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must lie in (0, 1] for donor-cell upwinding")

    @property
    def stable_timestep(self) -> float:
        """Largest timestep satisfying the unsplit multidimensional CFL limit."""

        spectral_radius = (
            abs(self.velocity_x) / self.grid.dx
            + abs(self.velocity_y) / self.grid.dy
        )
        if spectral_radius == 0.0:
            return np.inf
        return self.cfl / spectral_radius

    def spatial_operator(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return the conservative periodic donor-cell spatial operator."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        return self.spatial_operator_with_ghost_cells(
            field, fill_periodic_ghost_cells_2d(field)
        )

    def interface_fluxes(
        self, values: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return periodic x- and y-interface flux arrays."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        return self.interface_fluxes_with_ghost_cells(
            field, fill_periodic_ghost_cells_2d(field)
        )

    def interface_fluxes_with_ghost_cells(
        self, values: ArrayLike, ghosted_values: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return interface fluxes using one caller-provided ghost layer."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        ghosted = np.asarray(ghosted_values, dtype=float)
        expected_shape = (self.grid.ny + 2, self.grid.nx + 2)
        if ghosted.shape != expected_shape:
            raise ValueError(f"ghosted_values must have shape {expected_shape}")
        if not np.all(np.isfinite(ghosted)):
            raise ValueError("ghosted_values must be finite")
        if not np.array_equal(ghosted[1:-1, 1:-1], field):
            raise ValueError("The valid portion of ghosted_values must equal values")

        if self.velocity_x >= 0.0:
            flux_x = self.velocity_x * ghosted[1:-1, :-1]
        else:
            flux_x = self.velocity_x * ghosted[1:-1, 1:]
        if self.velocity_y >= 0.0:
            flux_y = self.velocity_y * ghosted[:-1, 1:-1]
        else:
            flux_y = self.velocity_y * ghosted[1:, 1:-1]
        return flux_x, flux_y

    def spatial_operator_with_ghost_cells(
        self, values: ArrayLike, ghosted_values: ArrayLike
    ) -> NDArray[np.float64]:
        """Return the conservative operator using caller-provided ghosts."""

        flux_x, flux_y = self.interface_fluxes_with_ghost_cells(
            values, ghosted_values
        )
        return -(
            (flux_x[:, 1:] - flux_x[:, :-1]) / self.grid.dx
            + (flux_y[1:, :] - flux_y[:-1, :]) / self.grid.dy
        )

    def step(self, values: ArrayLike, dt: float) -> NDArray[np.float64]:
        """Advance one forward-Euler step, rejecting CFL-unstable timesteps."""

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, self.stable_timestep)
        if dt > self.stable_timestep + tolerance:
            raise ValueError("dt exceeds the configured multidimensional CFL limit")
        field = np.asarray(values, dtype=float)
        return field + dt * self.spatial_operator(field)

    def step_with_ghost_cells(
        self, values: ArrayLike, ghosted_values: ArrayLike, dt: float
    ) -> NDArray[np.float64]:
        """Advance one step using externally filled boundary ghost cells."""

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, self.stable_timestep)
        if dt > self.stable_timestep + tolerance:
            raise ValueError("dt exceeds the configured multidimensional CFL limit")
        field = np.asarray(values, dtype=float)
        return field + dt * self.spatial_operator_with_ghost_cells(
            field, ghosted_values
        )

    def solve(self, initial_values: ArrayLike, final_time: float) -> AdvectionResult2D:
        """Integrate from zero to ``final_time`` with an exact final step."""

        values = np.array(initial_values, dtype=float, copy=True)
        self.grid.validate_field(values)
        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and non-negative")
        if final_time == 0.0 or (
            self.velocity_x == 0.0 and self.velocity_y == 0.0
        ):
            return AdvectionResult2D(values, float(final_time), 0)

        time = 0.0
        n_steps = 0
        time_tolerance = 16.0 * np.spacing(final_time)
        while final_time - time > time_tolerance:
            dt = min(self.stable_timestep, final_time - time)
            values = self.step(values, dt)
            time += dt
            n_steps += 1
        return AdvectionResult2D(values, float(final_time), n_steps)
