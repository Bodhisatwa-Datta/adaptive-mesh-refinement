"""Rectangular two-dimensional AMR hierarchy with conservative transfers."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from amr.grid.grid2d import UniformGrid2D
from amr.grid.patch2d import Patch2D
from amr.refinement.prolongation import prolong_piecewise_constant_2d
from amr.refinement.restriction import restrict_cell_averages_2d


@dataclass(slots=True)
class AMRHierarchy2D:
    """Tree of nested, non-overlapping rectangular patches."""

    base_grid: UniformGrid2D
    base_values: InitVar[ArrayLike]
    refinement_ratio: int = 2
    root: Patch2D = field(init=False)

    def __post_init__(self, base_values: ArrayLike) -> None:
        if isinstance(self.refinement_ratio, bool) or not isinstance(
            self.refinement_ratio, (int, np.integer)
        ):
            raise TypeError("refinement_ratio must be an integer")
        if self.refinement_ratio < 2:
            raise ValueError("refinement_ratio must be at least 2")
        self.root = Patch2D(self.base_grid, level=0, values=np.asarray(base_values))

    def add_patch(
        self,
        parent: Patch2D,
        x_start: int,
        x_stop: int,
        y_start: int,
        y_stop: int,
        values: ArrayLike | None = None,
    ) -> Patch2D:
        """Add a child covering a half-open rectangular parent-cell box."""

        if parent not in self.patches:
            raise ValueError("parent does not belong to this hierarchy")
        if not (
            0 <= x_start < x_stop <= parent.grid.nx
            and 0 <= y_start < y_stop <= parent.grid.ny
        ):
            raise ValueError("Requested parent range lies outside the parent patch")
        requested = ((y_start, y_stop), (x_start, x_stop))
        for sibling in parent.children:
            if sibling.parent_range is None:
                raise RuntimeError("Hierarchy invariant violated: child has no parent range")
            if _boxes_overlap(requested, sibling.parent_range):
                raise ValueError("Sibling patches cannot overlap")

        child_grid = UniformGrid2D(
            float(parent.grid.x_edges[x_start]),
            float(parent.grid.x_edges[x_stop]),
            (x_stop - x_start) * self.refinement_ratio,
            float(parent.grid.y_edges[y_start]),
            float(parent.grid.y_edges[y_stop]),
            (y_stop - y_start) * self.refinement_ratio,
        )
        if values is None:
            coarse_region = parent.values[y_start:y_stop, x_start:x_stop]
            child_values = prolong_piecewise_constant_2d(
                coarse_region, self.refinement_ratio
            )
        else:
            child_values = np.asarray(values, dtype=float)
        child = Patch2D(
            child_grid,
            parent.level + 1,
            child_values,
            parent=parent,
            parent_range=requested,
            refinement_ratio=self.refinement_ratio,
        )
        parent.children.append(child)
        return child

    def restrict_patch(self, patch: Patch2D) -> None:
        """Average a child onto the rectangular region covered in its parent."""

        if patch.parent is None or patch not in self.patches:
            raise ValueError("Can only restrict a child belonging to this hierarchy")
        if patch.parent_range is None:
            raise RuntimeError("Hierarchy invariant violated: child has no parent range")
        (y_start, y_stop), (x_start, x_stop) = patch.parent_range
        patch.parent.values[y_start:y_stop, x_start:x_stop] = (
            restrict_cell_averages_2d(patch.values, patch.refinement_ratio)
        )

    def remove_patch(self, patch: Patch2D, restrict: bool = True) -> None:
        """Remove a leaf patch, optionally restricting it before derefinement."""

        if patch.parent is None or patch not in self.patches:
            raise ValueError("Can only remove a child belonging to this hierarchy")
        if patch.children:
            raise ValueError("Cannot remove a patch that still has children")
        if restrict:
            self.restrict_patch(patch)
        patch.parent.children.remove(patch)

    @property
    def patches(self) -> tuple[Patch2D, ...]:
        """All patches in deterministic depth-first order."""

        result: list[Patch2D] = []

        def visit(patch: Patch2D) -> None:
            result.append(patch)
            for child in patch.children:
                visit(child)

        visit(self.root)
        return tuple(result)

    def patches_at_level(self, level: int) -> tuple[Patch2D, ...]:
        """Return all patches at a requested refinement level."""

        return tuple(patch for patch in self.patches if patch.level == level)

    @property
    def n_stored_cells(self) -> int:
        """Total stored cells, including covered cells."""

        return sum(patch.n_valid_cells for patch in self.patches)

    @property
    def n_active_cells(self) -> int:
        """Leaf-cell count, excluding parent cells covered by children."""

        def count(patch: Patch2D) -> int:
            covered = 0
            for child in patch.children:
                if child.parent_range is None:
                    raise RuntimeError(
                        "Hierarchy invariant violated: child has no parent range"
                    )
                (y_start, y_stop), (x_start, x_stop) = child.parent_range
                covered += (y_stop - y_start) * (x_stop - x_start)
            return patch.n_valid_cells - covered + sum(
                count(child) for child in patch.children
            )

        return count(self.root)


def _boxes_overlap(
    first: tuple[tuple[int, int], tuple[int, int]],
    second: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    (first_y0, first_y1), (first_x0, first_x1) = first
    (second_y0, second_y1), (second_x0, second_x1) = second
    overlap_y = max(first_y0, second_y0) < min(first_y1, second_y1)
    overlap_x = max(first_x0, second_x0) < min(first_x1, second_x1)
    return overlap_y and overlap_x
