"""Explicit finite-volume solver for periodic diffusion in two dimensions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amr.grid.grid2d import UniformGrid2D
from amr.numerics.boundary_conditions import fill_periodic_ghost_cells_2d


@dataclass(frozen=True, slots=True)
class DiffusionResult2D:
    """Final two-dimensional diffusion state and integration metadata."""

    values: NDArray[np.float64]
    time: float
    n_steps: int


@dataclass(frozen=True, slots=True)
class ExplicitDiffusion2D:
    """Solve ``u_t = D (u_xx + u_yy)`` with centred periodic fluxes."""

    grid: UniformGrid2D
    diffusivity: float
    stability_factor: float = 0.8

    def __post_init__(self) -> None:
        if not np.isfinite(self.diffusivity) or self.diffusivity < 0.0:
            raise ValueError("diffusivity must be non-negative and finite")
        if (
            not np.isfinite(self.stability_factor)
            or not 0.0 < self.stability_factor <= 1.0
        ):
            raise ValueError("stability_factor must lie in (0, 1]")

    @property
    def stable_timestep(self) -> float:
        """Largest timestep satisfying the 2D explicit diffusion limit."""

        if self.diffusivity == 0.0:
            return np.inf
        inverse_spacing_squared = 1.0 / self.grid.dx**2 + 1.0 / self.grid.dy**2
        return self.stability_factor / (
            2.0 * self.diffusivity * inverse_spacing_squared
        )

    def spatial_operator(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return the conservative centred periodic diffusion operator."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        return self.spatial_operator_with_ghost_cells(
            field, fill_periodic_ghost_cells_2d(field)
        )

    def interface_fluxes(
        self, values: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return periodic diffusive flux densities at x and y interfaces."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        return self.interface_fluxes_with_ghost_cells(
            field, fill_periodic_ghost_cells_2d(field)
        )

    def interface_fluxes_with_ghost_cells(
        self, values: ArrayLike, ghosted_values: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return diffusive interface fluxes using one external ghost layer."""

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
        flux_x = -self.diffusivity * (
            ghosted[1:-1, 1:] - ghosted[1:-1, :-1]
        ) / self.grid.dx
        flux_y = -self.diffusivity * (
            ghosted[1:, 1:-1] - ghosted[:-1, 1:-1]
        ) / self.grid.dy
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

    def fourier_amplification_factor(
        self, mode_x: int, mode_y: int, dt: float
    ) -> float:
        """Return the exact one-step gain of a discrete 2D Fourier mode."""

        for name, mode, count in (
            ("mode_x", mode_x, self.grid.nx),
            ("mode_y", mode_y, self.grid.ny),
        ):
            if isinstance(mode, bool) or not isinstance(mode, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= mode < count:
                raise ValueError(f"{name} must lie in [0, {count})")
        self._validate_timestep(dt)
        eigenvalue = (
            np.sin(np.pi * mode_x / self.grid.nx) ** 2 / self.grid.dx**2
            + np.sin(np.pi * mode_y / self.grid.ny) ** 2 / self.grid.dy**2
        )
        return float(1.0 - 4.0 * self.diffusivity * dt * eigenvalue)

    def step(self, values: ArrayLike, dt: float) -> NDArray[np.float64]:
        """Advance one conservative forward-Euler diffusion step."""

        self._validate_timestep(dt)
        field = np.asarray(values, dtype=float)
        return field + dt * self.spatial_operator(field)

    def step_with_ghost_cells(
        self, values: ArrayLike, ghosted_values: ArrayLike, dt: float
    ) -> NDArray[np.float64]:
        """Advance one step using externally filled boundary ghost cells."""

        self._validate_timestep(dt)
        field = np.asarray(values, dtype=float)
        return field + dt * self.spatial_operator_with_ghost_cells(
            field, ghosted_values
        )

    def _validate_timestep(self, dt: float) -> None:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, self.stable_timestep)
        if dt > self.stable_timestep + tolerance:
            raise ValueError("dt exceeds the explicit 2D diffusion stability limit")

    def solve(self, initial_values: ArrayLike, final_time: float) -> DiffusionResult2D:
        """Integrate to an exact final time using the parabolic stability limit."""

        values = np.array(initial_values, dtype=float, copy=True)
        self.grid.validate_field(values)
        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and non-negative")
        if final_time == 0.0 or self.diffusivity == 0.0:
            return DiffusionResult2D(values, float(final_time), 0)

        time = 0.0
        n_steps = 0
        tolerance = 16.0 * np.spacing(final_time)
        while final_time - time > tolerance:
            dt = min(self.stable_timestep, final_time - time)
            values = self.step(values, dt)
            time += dt
            n_steps += 1
        return DiffusionResult2D(values, float(final_time), n_steps)
