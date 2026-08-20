"""One-level rectangular AMR integration for explicit 2D diffusion."""

from dataclasses import dataclass

import numpy as np

from amr.diagnostics.conservation import composite_mass_2d
from amr.grid.hierarchy2d import AMRHierarchy2D
from amr.numerics.boundary_conditions import fill_coarse_fine_ghost_cells_2d
from amr.numerics.reflux2d import apply_reflux_2d
from amr.refinement.regrid2d import (
    GradientRegridConfig2D,
    regrid_from_gradient_2d,
)
from amr.solvers.diffusion2d import ExplicitDiffusion2D


@dataclass(frozen=True, slots=True)
class DiffusionRegridEvent2D:
    """One rectangular hierarchy change during diffusion integration."""

    time: float
    old_boxes: tuple[tuple[int, int, int, int], ...]
    new_boxes: tuple[tuple[int, int, int, int], ...]
    mass_change: float


@dataclass(frozen=True, slots=True)
class AMRDiffusionResult2D:
    """Integration metadata and conservation for 2D AMR diffusion."""

    time: float
    n_steps: int
    initial_mass: float
    final_mass: float
    cell_updates: int
    peak_active_cells: int
    peak_stored_cells: int
    fine_steps: int
    regrid_events: tuple[DiffusionRegridEvent2D, ...] = ()

    @property
    def mass_error(self) -> float:
        return self.final_mass - self.initial_mass


@dataclass(slots=True)
class AMRExplicitDiffusion2D:
    """Advance static or dynamic rectangular diffusion with optional subcycling."""

    hierarchy: AMRHierarchy2D
    diffusivity: float
    stability_factor: float = 0.8
    reflux: bool = False
    subcycling: bool = False
    regrid_config: GradientRegridConfig2D | None = None
    regrid_interval: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.diffusivity) or self.diffusivity < 0.0:
            raise ValueError("diffusivity must be non-negative and finite")
        if (
            not np.isfinite(self.stability_factor)
            or not 0.0 < self.stability_factor <= 1.0
        ):
            raise ValueError("stability_factor must lie in (0, 1]")
        if any(patch.level > 1 for patch in self.hierarchy.patches):
            raise NotImplementedError("The 2D AMR diffusion solver supports one fine level")
        if isinstance(self.regrid_interval, bool) or not isinstance(
            self.regrid_interval, (int, np.integer)
        ):
            raise TypeError("regrid_interval must be an integer")
        if self.regrid_interval < 1:
            raise ValueError("regrid_interval must be positive")

    @property
    def stable_timestep(self) -> float:
        """Return the root or finest explicit parabolic timestep."""

        patches = (self.hierarchy.root,) if self.subcycling else self.hierarchy.patches
        return min(
            ExplicitDiffusion2D(
                patch.grid, self.diffusivity, self.stability_factor
            ).stable_timestep
            for patch in patches
        )

    def step(self, dt: float) -> None:
        """Advance one root step and synchronize the hierarchy."""

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, self.stable_timestep)
        if dt > self.stable_timestep + tolerance:
            raise ValueError("dt exceeds the configured diffusion stability limit")
        if self.subcycling:
            self._subcycled_step(dt)
        else:
            self._synchronized_step(dt)

    def _synchronized_step(self, dt: float) -> None:
        root = self.hierarchy.root
        children = tuple(root.children)
        ghosts = [
            fill_coarse_fine_ghost_cells_2d(
                child, parent_interpolation="bilinear"
            )
            for child in children
        ]
        root_solver = ExplicitDiffusion2D(
            root.grid, self.diffusivity, self.stability_factor
        )
        coarse_fluxes = None
        if self.reflux and children:
            flux_x, flux_y = root_solver.interface_fluxes(root.values)
            coarse_fluxes = (dt * flux_x, dt * flux_y)
        next_root = root_solver.step(root.values, dt)
        next_children = []
        fine_fluxes = []
        for child, ghosted in zip(children, ghosts):
            solver = ExplicitDiffusion2D(
                child.grid, self.diffusivity, self.stability_factor
            )
            if self.reflux:
                flux_x, flux_y = solver.interface_fluxes_with_ghost_cells(
                    child.values, ghosted
                )
                fine_fluxes.append((dt * flux_x, dt * flux_y))
            next_children.append(
                solver.step_with_ghost_cells(child.values, ghosted, dt)
            )
        root.set_values(next_root)
        for child, values in zip(children, next_children):
            child.set_values(values)
        for child in children:
            self.hierarchy.restrict_patch(child)
        if coarse_fluxes is not None:
            apply_reflux_2d(self.hierarchy, children, coarse_fluxes, fine_fluxes)

    def _subcycled_step(self, coarse_dt: float) -> None:
        root = self.hierarchy.root
        children = tuple(root.children)
        root_solver = ExplicitDiffusion2D(
            root.grid, self.diffusivity, self.stability_factor
        )
        old_root = root.values.copy()
        next_root = root_solver.step(old_root, coarse_dt)
        if not children:
            root.set_values(next_root)
            return

        ratio = self.hierarchy.refinement_ratio
        fine_substeps = ratio**2
        fine_dt = coarse_dt / fine_substeps
        coarse_fluxes = None
        if self.reflux:
            flux_x, flux_y = root_solver.interface_fluxes(old_root)
            coarse_fluxes = (coarse_dt * flux_x, coarse_dt * flux_y)
        fine_fluxes = [
            (
                np.zeros((child.grid.ny, child.grid.nx + 1)),
                np.zeros((child.grid.ny + 1, child.grid.nx)),
            )
            for child in children
        ]
        for substep in range(fine_substeps):
            fraction = substep / fine_substeps
            parent_values = (1.0 - fraction) * old_root + fraction * next_root
            ghosts = [
                fill_coarse_fine_ghost_cells_2d(
                    child,
                    parent_values=parent_values,
                    parent_interpolation="bilinear",
                )
                for child in children
            ]
            next_children = []
            for index, (child, ghosted) in enumerate(zip(children, ghosts)):
                solver = ExplicitDiffusion2D(
                    child.grid, self.diffusivity, self.stability_factor
                )
                if self.reflux:
                    flux_x, flux_y = solver.interface_fluxes_with_ghost_cells(
                        child.values, ghosted
                    )
                    fine_fluxes[index][0][:] += fine_dt * flux_x
                    fine_fluxes[index][1][:] += fine_dt * flux_y
                next_children.append(
                    solver.step_with_ghost_cells(child.values, ghosted, fine_dt)
                )
            for child, values in zip(children, next_children):
                child.set_values(values)

        root.set_values(next_root)
        for child in children:
            self.hierarchy.restrict_patch(child)
        if coarse_fluxes is not None:
            apply_reflux_2d(self.hierarchy, children, coarse_fluxes, fine_fluxes)

    def solve(self, final_time: float) -> AMRDiffusionResult2D:
        """Mutate the hierarchy through an exact requested final time."""

        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and non-negative")
        initial_mass = composite_mass_2d(self.hierarchy)
        if final_time == 0.0 or self.diffusivity == 0.0:
            return AMRDiffusionResult2D(
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
        peak_active = self.hierarchy.n_active_cells
        peak_stored = self.hierarchy.n_stored_cells
        events: list[DiffusionRegridEvent2D] = []
        tolerance = 16.0 * np.spacing(final_time)
        while final_time - time > tolerance:
            dt = min(self.stable_timestep, final_time - time)
            children = tuple(self.hierarchy.root.children)
            if self.subcycling:
                substeps = self.hierarchy.refinement_ratio**2
                cell_updates += self.hierarchy.root.n_valid_cells + sum(
                    child.n_valid_cells * substeps for child in children
                )
                if children:
                    fine_steps += substeps
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
                        DiffusionRegridEvent2D(
                            float(min(time, final_time)),
                            report.old_boxes,
                            report.new_boxes,
                            report.mass_change,
                        )
                    )
                peak_active = max(peak_active, self.hierarchy.n_active_cells)
                peak_stored = max(peak_stored, self.hierarchy.n_stored_cells)
        return AMRDiffusionResult2D(
            float(final_time),
            n_steps,
            initial_mass,
            composite_mass_2d(self.hierarchy),
            cell_updates,
            peak_active,
            peak_stored,
            fine_steps,
            tuple(events),
        )
