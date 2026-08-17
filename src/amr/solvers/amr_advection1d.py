"""Synchronized one-level AMR integration for linear advection."""

from dataclasses import dataclass

import numpy as np

from amr.diagnostics.conservation import composite_mass
from amr.grid.hierarchy import AMRHierarchy1D
from amr.numerics.boundary_conditions import fill_coarse_fine_ghost_cells
from amr.solvers.advection1d import LinearAdvection1D


@dataclass(frozen=True, slots=True)
class AMRAdvectionResult:
    """Integration metadata and measured composite mass change."""

    time: float
    n_steps: int
    initial_mass: float
    final_mass: float

    @property
    def mass_error(self) -> float:
        """Signed change in composite finite-volume mass."""

        return self.final_mass - self.initial_mass


@dataclass(slots=True)
class AMRLinearAdvection1D:
    """Advance a static, one-level hierarchy with one global timestep.

    Fine ghosts are filled at the old time, every level takes the same step,
    then fine values are restricted onto covered coarse cells. Refluxing is not
    applied, so composite conservation is measured but not guaranteed.
    """

    hierarchy: AMRHierarchy1D
    velocity: float
    cfl: float = 0.8

    def __post_init__(self) -> None:
        if not np.isfinite(self.velocity):
            raise ValueError("velocity must be finite")
        if not np.isfinite(self.cfl) or not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must lie in (0, 1]")
        if any(patch.level > 1 for patch in self.hierarchy.patches):
            raise NotImplementedError("The synchronized AMR solver currently supports one fine level")

    @property
    def stable_timestep(self) -> float:
        """Global CFL timestep determined by the smallest active spacing."""

        if self.velocity == 0.0:
            return np.inf
        finest_dx = min(patch.grid.dx for patch in self.hierarchy.patches)
        return self.cfl * finest_dx / abs(self.velocity)

    def step(self, dt: float) -> None:
        """Advance all patches once and synchronize fine data onto the root."""

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, self.stable_timestep)
        if dt > self.stable_timestep + tolerance:
            raise ValueError("dt exceeds the finest-level CFL stability limit")

        children = tuple(self.hierarchy.root.children)
        ghosted_children = [fill_coarse_fine_ghost_cells(child) for child in children]

        root_solver = LinearAdvection1D(self.hierarchy.root.grid, self.velocity, self.cfl)
        next_root = root_solver.step(self.hierarchy.root.values, dt)
        next_children = []
        for child, ghosted in zip(children, ghosted_children):
            child_solver = LinearAdvection1D(child.grid, self.velocity, self.cfl)
            next_children.append(
                child_solver.step_with_ghost_cells(child.values, ghosted, dt)
            )

        self.hierarchy.root.set_values(next_root)
        for child, values in zip(children, next_children):
            child.set_values(values)
        for child in children:
            self.hierarchy.restrict_patch(child)

    def solve(self, final_time: float) -> AMRAdvectionResult:
        """Mutate the hierarchy from time zero to an exact requested final time."""

        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and non-negative")
        initial_mass = composite_mass(self.hierarchy)
        if final_time == 0.0 or self.velocity == 0.0:
            return AMRAdvectionResult(final_time, 0, initial_mass, initial_mass)

        time = 0.0
        n_steps = 0
        time_tolerance = 16.0 * np.spacing(final_time)
        while final_time - time > time_tolerance:
            dt = min(self.stable_timestep, final_time - time)
            self.step(dt)
            time += dt
            n_steps += 1
        return AMRAdvectionResult(
            time=float(final_time),
            n_steps=n_steps,
            initial_mass=initial_mass,
            final_mass=composite_mass(self.hierarchy),
        )
