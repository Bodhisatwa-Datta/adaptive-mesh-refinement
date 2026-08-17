"""One-level AMR integration for linear advection."""

from dataclasses import dataclass

import numpy as np

from amr.diagnostics.conservation import composite_mass
from amr.grid.hierarchy import AMRHierarchy1D
from amr.grid.patch import Patch1D
from amr.numerics.boundary_conditions import fill_coarse_fine_ghost_cells
from amr.refinement.regrid import GradientRegridConfig, regrid_from_gradient
from amr.solvers.advection1d import LinearAdvection1D


@dataclass(frozen=True, slots=True)
class RegridEvent:
    """A hierarchy change recorded during time integration."""

    time: float
    old_regions: tuple[tuple[int, int], ...]
    new_regions: tuple[tuple[int, int], ...]
    mass_change: float


@dataclass(frozen=True, slots=True)
class AMRAdvectionResult:
    """Integration metadata and measured composite mass change."""

    time: float
    n_steps: int
    initial_mass: float
    final_mass: float
    cell_updates: int = 0
    peak_active_cells: int = 0
    peak_stored_cells: int = 0
    fine_steps: int = 0
    regrid_events: tuple[RegridEvent, ...] = ()

    @property
    def mass_error(self) -> float:
        """Signed change in composite finite-volume mass."""

        return self.final_mass - self.initial_mass


@dataclass(slots=True)
class AMRLinearAdvection1D:
    """Advance a static or dynamic one-level hierarchy.

    Levels may use a shared timestep or temporal subcycling. Fine values are
    restricted after each coarse step, and optional refluxing corrects uncovered
    root cells using time-integrated coarse/fine interface flux differences.
    """

    hierarchy: AMRHierarchy1D
    velocity: float
    cfl: float = 0.8
    regrid_config: GradientRegridConfig | None = None
    regrid_interval: int = 1
    subcycling: bool = False
    reflux: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.velocity):
            raise ValueError("velocity must be finite")
        if not np.isfinite(self.cfl) or not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must lie in (0, 1]")
        if any(patch.level > 1 for patch in self.hierarchy.patches):
            raise NotImplementedError("The synchronized AMR solver currently supports one fine level")
        if isinstance(self.regrid_interval, bool) or not isinstance(
            self.regrid_interval, (int, np.integer)
        ):
            raise TypeError("regrid_interval must be an integer")
        if self.regrid_interval < 1:
            raise ValueError("regrid_interval must be positive")

    @property
    def stable_timestep(self) -> float:
        """CFL timestep for a global fine step or a subcycled root step."""

        if self.velocity == 0.0:
            return np.inf
        timestep_dx = (
            self.hierarchy.root.grid.dx
            if self.subcycling
            else min(patch.grid.dx for patch in self.hierarchy.patches)
        )
        return self.cfl * timestep_dx / abs(self.velocity)

    def step(self, dt: float) -> None:
        """Advance all patches once and synchronize fine data onto the root."""

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, self.stable_timestep)
        if dt > self.stable_timestep + tolerance:
            raise ValueError("dt exceeds the finest-level CFL stability limit")

        if self.subcycling:
            self._subcycled_step(dt)
        else:
            self._synchronized_step(dt)

    def _synchronized_step(self, dt: float) -> None:
        """Advance every level once with the same timestep."""

        children = tuple(self.hierarchy.root.children)
        ghosted_children = [fill_coarse_fine_ghost_cells(child) for child in children]

        root_solver = LinearAdvection1D(self.hierarchy.root.grid, self.velocity, self.cfl)
        coarse_integrated_fluxes = (
            dt * root_solver.interface_fluxes(self.hierarchy.root.values)
            if self.reflux and children
            else None
        )
        fine_integrated_fluxes = []
        next_root = root_solver.step(self.hierarchy.root.values, dt)
        next_children = []
        for child, ghosted in zip(children, ghosted_children):
            child_solver = LinearAdvection1D(child.grid, self.velocity, self.cfl)
            if self.reflux:
                fluxes = child_solver.interface_fluxes_with_ghost_cells(
                    child.values, ghosted
                )
                fine_integrated_fluxes.append((dt * fluxes[0], dt * fluxes[-1]))
            next_children.append(
                child_solver.step_with_ghost_cells(child.values, ghosted, dt)
            )

        self.hierarchy.root.set_values(next_root)
        for child, values in zip(children, next_children):
            child.set_values(values)
        for child in children:
            self.hierarchy.restrict_patch(child)
        if coarse_integrated_fluxes is not None:
            self._apply_reflux(
                children,
                coarse_integrated_fluxes,
                fine_integrated_fluxes,
            )

    def _subcycled_step(self, coarse_dt: float) -> None:
        """Advance level one ``r`` times during one provisional root step."""

        root = self.hierarchy.root
        children = tuple(root.children)
        root_solver = LinearAdvection1D(root.grid, self.velocity, self.cfl)
        old_root = root.values.copy()
        next_root = root_solver.step(old_root, coarse_dt)
        if not children:
            root.set_values(next_root)
            return

        ratio = self.hierarchy.refinement_ratio
        fine_dt = coarse_dt / ratio
        coarse_integrated_fluxes = (
            coarse_dt * root_solver.interface_fluxes(old_root) if self.reflux else None
        )
        fine_integrated_fluxes = np.zeros((len(children), 2), dtype=float)
        for substep in range(ratio):
            time_fraction = substep / ratio
            interpolated_parent = (
                (1.0 - time_fraction) * old_root + time_fraction * next_root
            )
            ghosted_children = [
                fill_coarse_fine_ghost_cells(child, parent_values=interpolated_parent)
                for child in children
            ]
            next_children = []
            for child_index, (child, ghosted) in enumerate(
                zip(children, ghosted_children)
            ):
                child_solver = LinearAdvection1D(child.grid, self.velocity, self.cfl)
                if self.reflux:
                    fluxes = child_solver.interface_fluxes_with_ghost_cells(
                        child.values, ghosted
                    )
                    fine_integrated_fluxes[child_index, 0] += fine_dt * fluxes[0]
                    fine_integrated_fluxes[child_index, 1] += fine_dt * fluxes[-1]
                next_children.append(
                    child_solver.step_with_ghost_cells(child.values, ghosted, fine_dt)
                )
            for child, values in zip(children, next_children):
                child.set_values(values)

        root.set_values(next_root)
        for child in children:
            self.hierarchy.restrict_patch(child)
        if coarse_integrated_fluxes is not None:
            self._apply_reflux(
                children,
                coarse_integrated_fluxes,
                [tuple(fluxes) for fluxes in fine_integrated_fluxes],
            )

    def _apply_reflux(
        self,
        children: tuple[Patch1D, ...],
        coarse_integrated_fluxes: np.ndarray,
        fine_integrated_fluxes: list[tuple[float, float]],
    ) -> None:
        """Correct uncovered root cells using coarse/fine flux-register differences."""

        root = self.hierarchy.root
        covered = np.zeros(root.n_valid_cells, dtype=bool)
        for child in children:
            if child.parent_range is None:
                raise RuntimeError("Hierarchy invariant violated: child has no parent range")
            start, stop = child.parent_range
            covered[start:stop] = True

        for child, (fine_left, fine_right) in zip(children, fine_integrated_fluxes):
            if child.parent_range is None:
                raise RuntimeError("Hierarchy invariant violated: child has no parent range")
            start, stop = child.parent_range
            left_cell = (start - 1) % root.n_valid_cells
            right_cell = stop % root.n_valid_cells
            if not covered[left_cell]:
                root.values[left_cell] += (
                    coarse_integrated_fluxes[start] - fine_left
                ) / root.grid.dx
            if not covered[right_cell]:
                root.values[right_cell] += (
                    fine_right - coarse_integrated_fluxes[stop]
                ) / root.grid.dx

    def solve(self, final_time: float) -> AMRAdvectionResult:
        """Mutate the hierarchy from time zero to an exact requested final time."""

        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and non-negative")
        initial_mass = composite_mass(self.hierarchy)
        if final_time == 0.0 or self.velocity == 0.0:
            return AMRAdvectionResult(
                time=final_time,
                n_steps=0,
                initial_mass=initial_mass,
                final_mass=initial_mass,
                peak_active_cells=self.hierarchy.n_active_cells,
                peak_stored_cells=self.hierarchy.n_stored_cells,
            )

        time = 0.0
        n_steps = 0
        cell_updates = 0
        fine_steps = 0
        peak_active_cells = self.hierarchy.n_active_cells
        peak_stored_cells = self.hierarchy.n_stored_cells
        events: list[RegridEvent] = []
        time_tolerance = 16.0 * np.spacing(final_time)
        while final_time - time > time_tolerance:
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
            if self.regrid_config is not None and n_steps % self.regrid_interval == 0:
                report = regrid_from_gradient(self.hierarchy, self.regrid_config)
                if report.changed:
                    events.append(
                        RegridEvent(
                            time=float(min(time, final_time)),
                            old_regions=report.old_regions,
                            new_regions=report.new_regions,
                            mass_change=report.mass_change,
                        )
                    )
                peak_active_cells = max(peak_active_cells, self.hierarchy.n_active_cells)
                peak_stored_cells = max(peak_stored_cells, self.hierarchy.n_stored_cells)
        return AMRAdvectionResult(
            time=float(final_time),
            n_steps=n_steps,
            initial_mass=initial_mass,
            final_mass=composite_mass(self.hierarchy),
            cell_updates=cell_updates,
            peak_active_cells=peak_active_cells,
            peak_stored_cells=peak_stored_cells,
            fine_steps=fine_steps,
            regrid_events=tuple(events),
        )
