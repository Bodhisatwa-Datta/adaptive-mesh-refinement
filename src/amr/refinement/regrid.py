"""Conservative replacement of level-one patches from refinement flags."""

from dataclasses import dataclass

import numpy as np

from amr.diagnostics.conservation import composite_mass
from amr.grid.hierarchy import AMRHierarchy1D
from amr.grid.patch import Patch1D
from amr.refinement.criteria import (
    buffer_flags,
    gradient_indicator,
    regions_from_flags,
)
from amr.refinement.prolongation import (
    prolong_conservative_linear,
    prolong_piecewise_constant,
)


@dataclass(frozen=True, slots=True)
class GradientRegridConfig:
    """Parameters for gradient-based refinement with derefinement hysteresis."""

    refine_threshold: float
    derefine_threshold: float
    n_buffer: int = 2
    merge_gap: int = 0
    normalized: bool = False
    epsilon: float = 1.0e-12
    periodic: bool = True
    prolongation: str = "piecewise_constant"

    def __post_init__(self) -> None:
        if not np.isfinite(self.refine_threshold) or self.refine_threshold < 0.0:
            raise ValueError("refine_threshold must be non-negative and finite")
        if not np.isfinite(self.derefine_threshold) or self.derefine_threshold < 0.0:
            raise ValueError("derefine_threshold must be non-negative and finite")
        if self.derefine_threshold > self.refine_threshold:
            raise ValueError("derefine_threshold cannot exceed refine_threshold")
        if isinstance(self.n_buffer, bool) or not isinstance(self.n_buffer, (int, np.integer)):
            raise TypeError("n_buffer must be an integer")
        if self.n_buffer < 0:
            raise ValueError("n_buffer must be non-negative")
        if isinstance(self.merge_gap, bool) or not isinstance(self.merge_gap, (int, np.integer)):
            raise TypeError("merge_gap must be an integer")
        if self.merge_gap < 0:
            raise ValueError("merge_gap must be non-negative")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive and finite")
        if self.prolongation not in {"piecewise_constant", "conservative_linear"}:
            raise ValueError(
                "prolongation must be 'piecewise_constant' or 'conservative_linear'"
            )


@dataclass(frozen=True, slots=True)
class RegridReport:
    """Measured result of one regridding decision."""

    changed: bool
    old_regions: tuple[tuple[int, int], ...]
    new_regions: tuple[tuple[int, int], ...]
    mass_before: float
    mass_after: float

    @property
    def mass_change(self) -> float:
        """Signed composite mass change caused only by regridding."""

        return self.mass_after - self.mass_before


def level_one_regions(hierarchy: AMRHierarchy1D) -> tuple[tuple[int, int], ...]:
    """Return sorted root-cell ranges covered by level-one patches."""

    regions = []
    for child in hierarchy.root.children:
        if child.parent_range is None:
            raise RuntimeError("Hierarchy invariant violated: child has no parent range")
        if child.children:
            raise NotImplementedError("Level-one replacement requires leaf child patches")
        regions.append(child.parent_range)
    return tuple(sorted(regions))


def _validate_regions(
    regions: list[tuple[int, int]] | tuple[tuple[int, int], ...], n_parent: int
) -> tuple[tuple[int, int], ...]:
    validated = tuple(sorted((int(start), int(stop)) for start, stop in regions))
    previous_stop = 0
    for index, (start, stop) in enumerate(validated):
        if not 0 <= start < stop <= n_parent:
            raise ValueError("Every regrid region must be a non-empty range inside the root")
        if index > 0 and start < previous_stop:
            raise ValueError("Regrid regions cannot overlap")
        previous_stop = stop
    return validated


def replace_level_one_patches(
    hierarchy: AMRHierarchy1D,
    regions: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    prolongation: str = "piecewise_constant",
) -> tuple[Patch1D, ...]:
    """Conservatively replace all root children with ``regions``.

    Old fine values are retained wherever old and new patches overlap. Before
    replacement, all old patches are restricted so newly refined cells can be
    initialized conservatively from synchronized root averages.
    """

    requested = _validate_regions(regions, hierarchy.root.n_valid_cells)
    old_children = tuple(hierarchy.root.children)
    if any(child.children for child in old_children):
        raise NotImplementedError("Cannot replace level-one patches with deeper children")
    old_data = []
    for child in old_children:
        if child.parent_range is None:
            raise RuntimeError("Hierarchy invariant violated: child has no parent range")
        old_data.append((child.parent_range, child.values.copy()))
        hierarchy.restrict_patch(child)

    hierarchy.root.children.clear()
    ratio = hierarchy.refinement_ratio
    if prolongation == "piecewise_constant":
        prolonged_root = prolong_piecewise_constant(hierarchy.root.values, ratio)
    elif prolongation == "conservative_linear":
        prolonged_root = prolong_conservative_linear(hierarchy.root.values, ratio)
    else:
        raise ValueError("Unknown prolongation method")
    new_children = []
    for start, stop in requested:
        values = prolonged_root[start * ratio : stop * ratio].copy()
        for (old_start, old_stop), old_values in old_data:
            overlap_start = max(start, old_start)
            overlap_stop = min(stop, old_stop)
            if overlap_start >= overlap_stop:
                continue
            new_slice = slice((overlap_start - start) * ratio, (overlap_stop - start) * ratio)
            old_slice = slice(
                (overlap_start - old_start) * ratio,
                (overlap_stop - old_start) * ratio,
            )
            values[new_slice] = old_values[old_slice]
        new_children.append(hierarchy.add_patch(hierarchy.root, start, stop, values=values))
    return tuple(new_children)


def regrid_from_gradient(
    hierarchy: AMRHierarchy1D,
    config: GradientRegridConfig,
) -> RegridReport:
    """Rebuild level one from root gradients while preserving composite mass."""

    old_regions = level_one_regions(hierarchy)
    indicator = gradient_indicator(
        hierarchy.root.values,
        hierarchy.root.grid.dx,
        normalized=config.normalized,
        epsilon=config.epsilon,
        periodic=config.periodic,
    )
    currently_refined = np.zeros(hierarchy.root.n_valid_cells, dtype=bool)
    for start, stop in old_regions:
        currently_refined[start:stop] = True

    newly_flagged = indicator > config.refine_threshold
    retained = currently_refined & (indicator > config.derefine_threshold)
    desired = buffer_flags(
        newly_flagged | retained,
        config.n_buffer,
        periodic=config.periodic,
    )
    new_regions = tuple(regions_from_flags(desired, merge_gap=config.merge_gap))
    mass_before = composite_mass(hierarchy)
    if new_regions == old_regions:
        return RegridReport(False, old_regions, new_regions, mass_before, mass_before)

    replace_level_one_patches(
        hierarchy,
        new_regions,
        prolongation=config.prolongation,
    )
    mass_after = composite_mass(hierarchy)
    return RegridReport(True, old_regions, new_regions, mass_before, mass_after)
