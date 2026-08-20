"""Synchronized one-level rectangular AMR integration for 2D advection."""

from dataclasses import dataclass

import numpy as np

from amr.diagnostics.conservation import composite_mass_2d
from amr.grid.hierarchy2d import AMRHierarchy2D
from amr.grid.patch2d import Patch2D
from amr.numerics.boundary_conditions import fill_coarse_fine_ghost_cells_2d
from amr.numerics.reflux2d import apply_reflux_2d
from amr.refinement.regrid2d import (
    GradientRegridConfig2D,
    regrid_from_gradient_2d,
)
from amr.solvers.advection2d import LinearAdvection2D


@dataclass(frozen=True, slots=True)
class RegridEvent2D:
    """One rectangular hierarchy change during time integration."""

    time: float
    old_boxes: tuple[tuple[int, int, int, int], ...]
    new_boxes: tuple[tuple[int, int, int, int], ...]
    mass_change: float


@dataclass(frozen=True, slots=True)
class AMRAdvectionResult2D:
    """Final metadata and measured conservation for a 2D AMR integration."""

    time: float
    n_steps: int
    initial_mass: float
    final_mass: float
    cell_updates: int
    peak_active_cells: int
    peak_stored_cells: int
    fine_steps: int
    regrid_events: tuple[RegridEvent2D, ...] = ()

    @property
    def mass_error(self) -> float:
        """Signed change in composite finite-volume mass."""

        return self.final_mass - self.initial_mass


@dataclass(slots=True)
class AMRLinearAdvection2D:
    """Advance a static or dynamically regridded one-level 2D hierarchy."""

    hierarchy: AMRHierarchy2D
    velocity_x: float
    velocity_y: float
    cfl: float = 0.8
    reflux: bool = False
    subcycling: bool = False
    regrid_config: GradientRegridConfig2D | None = None
    regrid_interval: int = 1

    def __post_init__(self) -> None:
        if not np.all(np.isfinite((self.velocity_x, self.velocity_y))):
            raise ValueError("velocity components must be finite")
        if not np.isfinite(self.cfl) or not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must lie in (0, 1]")
        if any(patch.level > 1 for patch in self.hierarchy.patches):
            raise NotImplementedError("The 2D AMR solver supports one fine level")
        if isinstance(self.regrid_interval, bool) or not isinstance(
            self.regrid_interval, (int, np.integer)
        ):
            raise TypeError("regrid_interval must be an integer")
        if self.regrid_interval < 1:
            raise ValueError("regrid_interval must be positive")

    @property
    def stable_timestep(self) -> float:
        """Return the global timestep limited by the finest stored spacing."""

        timestep_patches = (
            (self.hierarchy.root,) if self.subcycling else self.hierarchy.patches
        )
        spectral_radius = max(
            abs(self.velocity_x) / patch.grid.dx
            + abs(self.velocity_y) / patch.grid.dy
            for patch in timestep_patches
        )
        if spectral_radius == 0.0:
            return np.inf
        return self.cfl / spectral_radius

    def step(self, dt: float) -> None:
        """Advance every patch once, restrict, and optionally reflux."""

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, self.stable_timestep)
        if dt > self.stable_timestep + tolerance:
            raise ValueError("dt exceeds the finest-level CFL stability limit")
        if self.subcycling:
            self._subcycled_step(dt)
            return

        root = self.hierarchy.root
        children = tuple(root.children)
        ghosted_children = [
            fill_coarse_fine_ghost_cells_2d(child) for child in children
        ]
        root_solver = LinearAdvection2D(
            root.grid, self.velocity_x, self.velocity_y, self.cfl
        )
        coarse_integrated_fluxes = None
        if self.reflux and children:
            root_flux_x, root_flux_y = root_solver.interface_fluxes(root.values)
            coarse_integrated_fluxes = (dt * root_flux_x, dt * root_flux_y)
        next_root = root_solver.step(root.values, dt)

        next_children = []
        fine_integrated_fluxes = []
        for child, ghosted in zip(children, ghosted_children):
            child_solver = LinearAdvection2D(
                child.grid, self.velocity_x, self.velocity_y, self.cfl
            )
            if self.reflux:
                flux_x, flux_y = child_solver.interface_fluxes_with_ghost_cells(
                    child.values, ghosted
                )
                fine_integrated_fluxes.append((dt * flux_x, dt * flux_y))
            next_children.append(
                child_solver.step_with_ghost_cells(child.values, ghosted, dt)
            )

        root.set_values(next_root)
        for child, values in zip(children, next_children):
            child.set_values(values)
        for child in children:
            self.hierarchy.restrict_patch(child)
        if coarse_integrated_fluxes is not None:
            self._apply_reflux(
                children, coarse_integrated_fluxes, fine_integrated_fluxes
            )

    def _subcycled_step(self, coarse_dt: float) -> None:
        """Take ``r`` fine steps during one provisional root-grid step."""

        root = self.hierarchy.root
        children = tuple(root.children)
        root_solver = LinearAdvection2D(
            root.grid, self.velocity_x, self.velocity_y, self.cfl
        )
        old_root = root.values.copy()
        next_root = root_solver.step(old_root, coarse_dt)
        if not children:
            root.set_values(next_root)
            return

        ratio = self.hierarchy.refinement_ratio
        fine_dt = coarse_dt / ratio
        coarse_integrated_fluxes = None
        if self.reflux:
            coarse_x, coarse_y = root_solver.interface_fluxes(old_root)
            coarse_integrated_fluxes = (
                coarse_dt * coarse_x,
                coarse_dt * coarse_y,
            )
        fine_integrated_fluxes = [
            (
                np.zeros((child.grid.ny, child.grid.nx + 1)),
                np.zeros((child.grid.ny + 1, child.grid.nx)),
            )
            for child in children
        ]

        for substep in range(ratio):
            time_fraction = substep / ratio
            interpolated_parent = (
                (1.0 - time_fraction) * old_root + time_fraction * next_root
            )
            ghosted_children = [
                fill_coarse_fine_ghost_cells_2d(
                    child, parent_values=interpolated_parent
                )
                for child in children
            ]
            next_children = []
            for index, (child, ghosted) in enumerate(
                zip(children, ghosted_children)
            ):
                child_solver = LinearAdvection2D(
                    child.grid, self.velocity_x, self.velocity_y, self.cfl
                )
                if self.reflux:
                    flux_x, flux_y = child_solver.interface_fluxes_with_ghost_cells(
                        child.values, ghosted
                    )
                    fine_integrated_fluxes[index][0][:] += fine_dt * flux_x
                    fine_integrated_fluxes[index][1][:] += fine_dt * flux_y
                next_children.append(
                    child_solver.step_with_ghost_cells(
                        child.values, ghosted, fine_dt
                    )
                )
            for child, values in zip(children, next_children):
                child.set_values(values)

        root.set_values(next_root)
        for child in children:
            self.hierarchy.restrict_patch(child)
        if coarse_integrated_fluxes is not None:
            self._apply_reflux(
                children, coarse_integrated_fluxes, fine_integrated_fluxes
            )

    def _apply_reflux(
        self,
        children: tuple[Patch2D, ...],
        coarse_fluxes: tuple[np.ndarray, np.ndarray],
        fine_fluxes: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Replace coarse interface fluxes with face-averaged fine fluxes."""

        apply_reflux_2d(self.hierarchy, children, coarse_fluxes, fine_fluxes)

    def solve(self, final_time: float) -> AMRAdvectionResult2D:
        """Mutate the hierarchy through an exact requested final time."""

        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and non-negative")
        initial_mass = composite_mass_2d(self.hierarchy)
        if final_time == 0.0 or (
            self.velocity_x == 0.0 and self.velocity_y == 0.0
        ):
            return AMRAdvectionResult2D(
                final_time,
                0,
                initial_mass,
                initial_mass,
                0,
                self.hierarchy.n_active_cells,
                self.hierarchy.n_stored_cells,
                0,
            )

        time = 0.0
        n_steps = 0
        cell_updates = 0
        fine_steps = 0
        peak_active_cells = self.hierarchy.n_active_cells
        peak_stored_cells = self.hierarchy.n_stored_cells
        events: list[RegridEvent2D] = []
        tolerance = 16.0 * np.spacing(final_time)
        while final_time - time > tolerance:
            dt = min(self.stable_timestep, final_time - time)
            children = tuple(self.hierarchy.root.children)
            if self.subcycling:
                cell_updates += self.hierarchy.root.n_valid_cells + sum(
                    child.n_valid_cells * self.hierarchy.refinement_ratio
                    for child in children
                )
                if children:
                    fine_steps += self.hierarchy.refinement_ratio
            else:
                cell_updates += self.hierarchy.n_stored_cells
                if children:
                    fine_steps += 1
            self.step(dt)
            time += dt
            n_steps += 1
            if (
                self.regrid_config is not None
                and n_steps % self.regrid_interval == 0
            ):
                report = regrid_from_gradient_2d(
                    self.hierarchy, self.regrid_config
                )
                if report.changed:
                    events.append(
                        RegridEvent2D(
                            float(min(time, final_time)),
                            report.old_boxes,
                            report.new_boxes,
                            report.mass_change,
                        )
                    )
                peak_active_cells = max(
                    peak_active_cells, self.hierarchy.n_active_cells
                )
                peak_stored_cells = max(
                    peak_stored_cells, self.hierarchy.n_stored_cells
                )
        return AMRAdvectionResult2D(
            float(final_time),
            n_steps,
            initial_mass,
            composite_mass_2d(self.hierarchy),
            cell_updates,
            peak_active_cells,
            peak_stored_cells,
            fine_steps,
            tuple(events),
        )
