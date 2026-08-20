"""Conservative replacement of rectangular level-one patches."""

from dataclasses import dataclass

import numpy as np

from amr.diagnostics.conservation import composite_mass_2d
from amr.grid.hierarchy2d import AMRHierarchy2D
from amr.grid.patch2d import Patch2D
from amr.refinement.criteria import (
    boxes_from_flags_2d,
    buffer_flags_2d,
    gradient_indicator_2d,
)
from amr.refinement.prolongation import (
    prolong_conservative_quadratic_2d,
    prolong_piecewise_constant_2d,
)

IndexBox2D = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class GradientRegridConfig2D:
    """Parameters for gradient-based rectangular regridding with hysteresis."""

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
        if isinstance(self.n_buffer, bool) or not isinstance(
            self.n_buffer, (int, np.integer)
        ):
            raise TypeError("n_buffer must be an integer")
        if self.n_buffer < 0:
            raise ValueError("n_buffer must be non-negative")
        if isinstance(self.merge_gap, bool) or not isinstance(
            self.merge_gap, (int, np.integer)
        ):
            raise TypeError("merge_gap must be an integer")
        if self.merge_gap < 0:
            raise ValueError("merge_gap must be non-negative")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive and finite")
        if self.prolongation not in {
            "piecewise_constant",
            "conservative_quadratic",
        }:
            raise ValueError(
                "prolongation must be 'piecewise_constant' or "
                "'conservative_quadratic'"
            )


@dataclass(frozen=True, slots=True)
class RegridReport2D:
    """Measured result of one rectangular regridding decision."""

    changed: bool
    old_boxes: tuple[IndexBox2D, ...]
    new_boxes: tuple[IndexBox2D, ...]
    mass_before: float
    mass_after: float

    @property
    def mass_change(self) -> float:
        """Signed composite mass change caused only by patch replacement."""

        return self.mass_after - self.mass_before


def level_one_boxes_2d(hierarchy: AMRHierarchy2D) -> tuple[IndexBox2D, ...]:
    """Return sorted root-cell boxes covered by level-one patches."""

    boxes = []
    for child in hierarchy.root.children:
        if child.parent_range is None:
            raise RuntimeError("Hierarchy invariant violated: missing parent range")
        if child.children:
            raise NotImplementedError("Level-one replacement requires leaf patches")
        (y_start, y_stop), (x_start, x_stop) = child.parent_range
        boxes.append((x_start, x_stop, y_start, y_stop))
    return tuple(sorted(boxes))


def replace_level_one_patches_2d(
    hierarchy: AMRHierarchy2D,
    boxes: list[IndexBox2D] | tuple[IndexBox2D, ...],
    *,
    prolongation: str = "piecewise_constant",
) -> tuple[Patch2D, ...]:
    """Replace root children conservatively while retaining overlap data."""

    requested = _validate_boxes(boxes, hierarchy.root.grid.nx, hierarchy.root.grid.ny)
    old_children = tuple(hierarchy.root.children)
    if any(child.children for child in old_children):
        raise NotImplementedError("Cannot replace level-one patches with deeper children")
    old_data: list[tuple[IndexBox2D, np.ndarray]] = []
    for child in old_children:
        if child.parent_range is None:
            raise RuntimeError("Hierarchy invariant violated: missing parent range")
        (old_y0, old_y1), (old_x0, old_x1) = child.parent_range
        old_data.append(((old_x0, old_x1, old_y0, old_y1), child.values.copy()))
        hierarchy.restrict_patch(child)

    hierarchy.root.children.clear()
    ratio = hierarchy.refinement_ratio
    if prolongation == "piecewise_constant":
        prolonged_root = prolong_piecewise_constant_2d(hierarchy.root.values, ratio)
    elif prolongation == "conservative_quadratic":
        prolonged_root = prolong_conservative_quadratic_2d(
            hierarchy.root.values, ratio, periodic=True
        )
    else:
        raise ValueError("Unknown 2D prolongation method")
    new_children = []
    for x_start, x_stop, y_start, y_stop in requested:
        values = prolonged_root[
            y_start * ratio : y_stop * ratio,
            x_start * ratio : x_stop * ratio,
        ].copy()
        for (old_x0, old_x1, old_y0, old_y1), old_values in old_data:
            overlap_x0 = max(x_start, old_x0)
            overlap_x1 = min(x_stop, old_x1)
            overlap_y0 = max(y_start, old_y0)
            overlap_y1 = min(y_stop, old_y1)
            if overlap_x0 >= overlap_x1 or overlap_y0 >= overlap_y1:
                continue
            new_y = slice(
                (overlap_y0 - y_start) * ratio,
                (overlap_y1 - y_start) * ratio,
            )
            new_x = slice(
                (overlap_x0 - x_start) * ratio,
                (overlap_x1 - x_start) * ratio,
            )
            old_y = slice(
                (overlap_y0 - old_y0) * ratio,
                (overlap_y1 - old_y0) * ratio,
            )
            old_x = slice(
                (overlap_x0 - old_x0) * ratio,
                (overlap_x1 - old_x0) * ratio,
            )
            values[new_y, new_x] = old_values[old_y, old_x]
        new_children.append(
            hierarchy.add_patch(
                hierarchy.root,
                x_start,
                x_stop,
                y_start,
                y_stop,
                values=values,
            )
        )
    return tuple(new_children)


def regrid_from_gradient_2d(
    hierarchy: AMRHierarchy2D,
    config: GradientRegridConfig2D,
) -> RegridReport2D:
    """Rebuild level one from buffered 2D gradients while preserving mass."""

    old_boxes = level_one_boxes_2d(hierarchy)
    indicator = gradient_indicator_2d(
        hierarchy.root.values,
        hierarchy.root.grid.dx,
        hierarchy.root.grid.dy,
        normalized=config.normalized,
        epsilon=config.epsilon,
        periodic=config.periodic,
    )
    currently_refined = np.zeros(hierarchy.root.grid.shape, dtype=bool)
    for x_start, x_stop, y_start, y_stop in old_boxes:
        currently_refined[y_start:y_stop, x_start:x_stop] = True
    newly_flagged = indicator > config.refine_threshold
    retained = currently_refined & (indicator > config.derefine_threshold)
    desired = buffer_flags_2d(
        newly_flagged | retained, config.n_buffer, periodic=config.periodic
    )
    new_boxes = tuple(boxes_from_flags_2d(desired, merge_gap=config.merge_gap))
    mass_before = composite_mass_2d(hierarchy)
    if new_boxes == old_boxes:
        return RegridReport2D(
            False, old_boxes, new_boxes, mass_before, mass_before
        )
    replace_level_one_patches_2d(
        hierarchy, new_boxes, prolongation=config.prolongation
    )
    mass_after = composite_mass_2d(hierarchy)
    return RegridReport2D(True, old_boxes, new_boxes, mass_before, mass_after)


def _validate_boxes(
    boxes: list[IndexBox2D] | tuple[IndexBox2D, ...], nx: int, ny: int
) -> tuple[IndexBox2D, ...]:
    validated = tuple(sorted(tuple(int(value) for value in box) for box in boxes))
    for x_start, x_stop, y_start, y_stop in validated:
        if not (0 <= x_start < x_stop <= nx and 0 <= y_start < y_stop <= ny):
            raise ValueError("Every regrid box must be non-empty and inside the root")
    for index, first in enumerate(validated):
        for second in validated[index + 1 :]:
            overlap_x = max(first[0], second[0]) < min(first[1], second[1])
            overlap_y = max(first[2], second[2]) < min(first[3], second[3])
            if overlap_x and overlap_y:
                raise ValueError("Regrid boxes cannot overlap")
    return validated
