"""One-level explicit AMR solver for one-dimensional diffusion."""

from dataclasses import dataclass

import numpy as np

from amr.diagnostics.conservation import composite_mass
from amr.grid.hierarchy import AMRHierarchy1D
from amr.grid.patch import Patch1D
from amr.numerics.boundary_conditions import fill_coarse_fine_ghost_cells
from amr.refinement.regrid import GradientRegridConfig, regrid_from_gradient
from amr.solvers.amr_advection1d import RegridEvent
from amr.solvers.diffusion1d import ExplicitDiffusion1D


@dataclass(frozen=True, slots=True)
class AMRDiffusionResult:
    """Integration, conservation, regridding, and work diagnostics."""

    time: float
    n_steps: int
    fine_steps: int
    initial_mass: float
    final_mass: float
    cell_updates: int
    peak_active_cells: int
    peak_stored_cells: int
    regrid_events: tuple[RegridEvent, ...]

    @property
    def mass_error(self) -> float:
        return self.final_mass - self.initial_mass


@dataclass(slots=True)
class AMRExplicitDiffusion1D:
    """Advance one AMR level with parabolic subcycling and optional refluxing."""

    hierarchy: AMRHierarchy1D
    diffusivity: float
    stability_factor: float = 0.8
    regrid_config: GradientRegridConfig | None = None
    regrid_interval: int = 1
    subcycling: bool = True
    reflux: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.diffusivity) or self.diffusivity < 0.0:
            raise ValueError("diffusivity must be non-negative and finite")
        if not np.isfinite(self.stability_factor) or not 0.0 < self.stability_factor <= 1.0:
            raise ValueError("stability_factor must lie in (0, 1]")
        if any(patch.level > 1 for patch in self.hierarchy.patches):
            raise NotImplementedError("The diffusion AMR solver currently supports one fine level")
        if isinstance(self.regrid_interval, bool) or not isinstance(
            self.regrid_interval, (int, np.integer)
        ):
            raise TypeError("regrid_interval must be an integer")
        if self.regrid_interval < 1:
            raise ValueError("regrid_interval must be positive")

    @property
    def stable_timestep(self) -> float:
        if self.diffusivity == 0.0:
            return np.inf
        spacing = (
            self.hierarchy.root.grid.dx
            if self.subcycling
            else min(patch.grid.dx for patch in self.hierarchy.patches)
        )
        return self.stability_factor * spacing**2 / (2.0 * self.diffusivity)

    def step(self, coarse_dt: float) -> None:
        if not np.isfinite(coarse_dt) or coarse_dt <= 0.0:
            raise ValueError("coarse_dt must be positive and finite")
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, self.stable_timestep)
        if coarse_dt > self.stable_timestep + tolerance:
            raise ValueError("coarse_dt exceeds the AMR diffusion stability limit")
        if self.subcycling:
            self._subcycled_step(coarse_dt)
        else:
            self._synchronized_step(coarse_dt)

    def _ghosts(
        self,
        children: tuple[Patch1D, ...],
        parent_values: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        return [
            fill_coarse_fine_ghost_cells(
                child,
                parent_values=parent_values,
                parent_interpolation="linear",
            )
            for child in children
        ]

    def _synchronized_step(self, dt: float) -> None:
        root = self.hierarchy.root
        children = tuple(root.children)
        ghosts = self._ghosts(children)
        root_solver = ExplicitDiffusion1D(root.grid, self.diffusivity, self.stability_factor)
        coarse_integrals = (
            dt * root_solver.interface_fluxes(root.values)
            if self.reflux and children
            else None
        )
        fine_integrals = []
        next_root = root_solver.step(root.values, dt)
        next_children = []
        for child, ghosted in zip(children, ghosts):
            solver = ExplicitDiffusion1D(child.grid, self.diffusivity, self.stability_factor)
            if self.reflux:
                fluxes = solver.interface_fluxes_with_ghost_cells(child.values, ghosted)
                fine_integrals.append((dt * fluxes[0], dt * fluxes[-1]))
            next_children.append(solver.step_with_ghost_cells(child.values, ghosted, dt))
        root.set_values(next_root)
        for child, values in zip(children, next_children):
            child.set_values(values)
            self.hierarchy.restrict_patch(child)
        if coarse_integrals is not None:
            self._apply_reflux(children, coarse_integrals, fine_integrals)

    def _subcycled_step(self, coarse_dt: float) -> None:
        root = self.hierarchy.root
        children = tuple(root.children)
        root_solver = ExplicitDiffusion1D(root.grid, self.diffusivity, self.stability_factor)
        old_root = root.values.copy()
        next_root = root_solver.step(old_root, coarse_dt)
        if not children:
            root.set_values(next_root)
            return

        ratio = self.hierarchy.refinement_ratio
        substeps = ratio**2
        fine_dt = coarse_dt / substeps
        coarse_integrals = (
            coarse_dt * root_solver.interface_fluxes(old_root) if self.reflux else None
        )
        fine_integrals = np.zeros((len(children), 2), dtype=float)
        for substep in range(substeps):
            fraction = substep / substeps
            parent_values = (1.0 - fraction) * old_root + fraction * next_root
            ghosts = self._ghosts(children, parent_values)
            next_children = []
            for index, (child, ghosted) in enumerate(zip(children, ghosts)):
                solver = ExplicitDiffusion1D(
                    child.grid,
                    self.diffusivity,
                    self.stability_factor,
                )
                if self.reflux:
                    fluxes = solver.interface_fluxes_with_ghost_cells(child.values, ghosted)
                    fine_integrals[index, 0] += fine_dt * fluxes[0]
                    fine_integrals[index, 1] += fine_dt * fluxes[-1]
                next_children.append(solver.step_with_ghost_cells(child.values, ghosted, fine_dt))
            for child, values in zip(children, next_children):
                child.set_values(values)

        root.set_values(next_root)
        for child in children:
            self.hierarchy.restrict_patch(child)
        if coarse_integrals is not None:
            self._apply_reflux(
                children,
                coarse_integrals,
                [tuple(fluxes) for fluxes in fine_integrals],
            )

    def _apply_reflux(
        self,
        children: tuple[Patch1D, ...],
        coarse_integrals: np.ndarray,
        fine_integrals: list[tuple[float, float]],
    ) -> None:
        root = self.hierarchy.root
        covered = np.zeros(root.n_valid_cells, dtype=bool)
        for child in children:
            if child.parent_range is None:
                raise RuntimeError("Hierarchy invariant violated: child has no parent range")
            start, stop = child.parent_range
            covered[start:stop] = True
        for child, (fine_left, fine_right) in zip(children, fine_integrals):
            if child.parent_range is None:
                raise RuntimeError("Hierarchy invariant violated: child has no parent range")
            start, stop = child.parent_range
            left = (start - 1) % root.n_valid_cells
            right = stop % root.n_valid_cells
            if not covered[left]:
                root.values[left] += (coarse_integrals[start] - fine_left) / root.grid.dx
            if not covered[right]:
                root.values[right] += (fine_right - coarse_integrals[stop]) / root.grid.dx

    def solve(self, final_time: float) -> AMRDiffusionResult:
        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and non-negative")
        initial_mass = composite_mass(self.hierarchy)
        time = 0.0
        steps = 0
        fine_steps = 0
        updates = 0
        peak_active = self.hierarchy.n_active_cells
        peak_stored = self.hierarchy.n_stored_cells
        events = []
        if final_time > 0.0 and self.diffusivity > 0.0:
            tolerance = 16.0 * np.spacing(final_time)
            while final_time - time > tolerance:
                dt = min(self.stable_timestep, final_time - time)
                children = tuple(self.hierarchy.root.children)
                if self.subcycling:
                    substeps = self.hierarchy.refinement_ratio**2
                    updates += self.hierarchy.root.n_valid_cells + sum(
                        child.n_valid_cells * substeps for child in children
                    )
                    if children:
                        fine_steps += substeps
                else:
                    updates += self.hierarchy.n_stored_cells
                    if children:
                        fine_steps += 1
                self.step(dt)
                time += dt
                steps += 1
                if self.regrid_config is not None and steps % self.regrid_interval == 0:
                    report = regrid_from_gradient(self.hierarchy, self.regrid_config)
                    if report.changed:
                        events.append(
                            RegridEvent(
                                float(min(time, final_time)),
                                report.old_regions,
                                report.new_regions,
                                report.mass_change,
                            )
                        )
                    peak_active = max(peak_active, self.hierarchy.n_active_cells)
                    peak_stored = max(peak_stored, self.hierarchy.n_stored_cells)
        return AMRDiffusionResult(
            time=float(final_time),
            n_steps=steps,
            fine_steps=fine_steps,
            initial_mass=initial_mass,
            final_mass=composite_mass(self.hierarchy),
            cell_updates=updates,
            peak_active_cells=peak_active,
            peak_stored_cells=peak_stored,
            regrid_events=tuple(events),
        )

