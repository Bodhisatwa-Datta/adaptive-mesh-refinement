"""Explicit finite-volume solver for one-dimensional diffusion."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amr.grid.grid1d import UniformGrid1D
from amr.numerics.boundary_conditions import fill_periodic_ghost_cells


@dataclass(frozen=True, slots=True)
class DiffusionResult:
    """Final diffusion state and integration metadata."""

    values: NDArray[np.float64]
    time: float
    n_steps: int


@dataclass(frozen=True, slots=True)
class ExplicitDiffusion1D:
    """Solve ``u_t = D u_xx`` using centred diffusive fluxes."""

    grid: UniformGrid1D
    diffusivity: float
    stability_factor: float = 0.8

    def __post_init__(self) -> None:
        if not np.isfinite(self.diffusivity) or self.diffusivity < 0.0:
            raise ValueError("diffusivity must be non-negative and finite")
        if not np.isfinite(self.stability_factor) or not 0.0 < self.stability_factor <= 1.0:
            raise ValueError("stability_factor must lie in (0, 1]")

    @property
    def stable_timestep(self) -> float:
        """Return ``factor * dx^2/(2D)`` for the explicit centred scheme."""

        if self.diffusivity == 0.0:
            return np.inf
        return self.stability_factor * self.grid.dx**2 / (2.0 * self.diffusivity)

    def interface_fluxes(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return periodic diffusive fluxes ``-D du/dx`` at all interfaces."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        return self.interface_fluxes_with_ghost_cells(
            field,
            fill_periodic_ghost_cells(field),
        )

    def interface_fluxes_with_ghost_cells(
        self,
        values: ArrayLike,
        ghosted_values: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return diffusive interface fluxes using supplied ghost cells."""

        field = np.asarray(values, dtype=float)
        self.grid.validate_field(field)
        ghosted = np.asarray(ghosted_values, dtype=float)
        if ghosted.shape != (self.grid.n_cells + 2,):
            raise ValueError("ghosted_values must contain one ghost cell on each side")
        if not np.all(np.isfinite(ghosted)):
            raise ValueError("ghosted_values must be finite")
        if not np.array_equal(ghosted[1:-1], field):
            raise ValueError("The valid portion of ghosted_values must equal values")
        return -self.diffusivity * (ghosted[1:] - ghosted[:-1]) / self.grid.dx

    def step(self, values: ArrayLike, dt: float) -> NDArray[np.float64]:
        """Advance one conservative forward-Euler diffusion step."""

        field = np.asarray(values, dtype=float)
        fluxes = self.interface_fluxes(field)
        self._validate_timestep(dt)
        return field - (dt / self.grid.dx) * (fluxes[1:] - fluxes[:-1])

    def step_with_ghost_cells(
        self,
        values: ArrayLike,
        ghosted_values: ArrayLike,
        dt: float,
    ) -> NDArray[np.float64]:
        """Advance one step using externally filled boundary values."""

        field = np.asarray(values, dtype=float)
        fluxes = self.interface_fluxes_with_ghost_cells(field, ghosted_values)
        self._validate_timestep(dt)
        return field - (dt / self.grid.dx) * (fluxes[1:] - fluxes[:-1])

    def _validate_timestep(self, dt: float) -> None:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, self.stable_timestep)
        if dt > self.stable_timestep + tolerance:
            raise ValueError("dt exceeds the explicit diffusion stability limit")

    def solve(self, initial_values: ArrayLike, final_time: float) -> DiffusionResult:
        """Integrate to an exact final time using the parabolic stability limit."""

        values = np.array(initial_values, dtype=float, copy=True)
        self.grid.validate_field(values)
        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and non-negative")
        if final_time == 0.0 or self.diffusivity == 0.0:
            return DiffusionResult(values, float(final_time), 0)

        time = 0.0
        steps = 0
        tolerance = 16.0 * np.spacing(final_time)
        while final_time - time > tolerance:
            dt = min(self.stable_timestep, final_time - time)
            values = self.step(values, dt)
            time += dt
            steps += 1
        return DiffusionResult(values, float(final_time), steps)

