"""Minimal one-dimensional AMR hierarchy with conservative data transfer."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from amr.grid.grid1d import UniformGrid1D
from amr.grid.patch import Patch1D
from amr.refinement.prolongation import prolong_piecewise_constant
from amr.refinement.restriction import restrict_cell_averages


@dataclass(slots=True)
class AMRHierarchy1D:
    """Tree of non-overlapping child patches with one ratio per level jump."""

    base_grid: UniformGrid1D
    base_values: InitVar[ArrayLike]
    refinement_ratio: int = 2
    root: Patch1D = field(init=False)

    def __post_init__(self, base_values: ArrayLike) -> None:
        if isinstance(self.refinement_ratio, bool) or not isinstance(
            self.refinement_ratio, (int, np.integer)
        ):
            raise TypeError("refinement_ratio must be an integer")
        if self.refinement_ratio < 2:
            raise ValueError("refinement_ratio must be at least 2")
        self.root = Patch1D(self.base_grid, level=0, values=np.asarray(base_values))

    def add_patch(
        self,
        parent: Patch1D,
        parent_start: int,
        parent_stop: int,
        values: ArrayLike | None = None,
    ) -> Patch1D:
        """Create a child covering ``[parent_start, parent_stop)`` in its parent.

        Without explicit values, conservative piecewise-constant prolongation is
        used. Sibling patches may touch but may not overlap.
        """

        if parent not in self.patches:
            raise ValueError("parent does not belong to this hierarchy")
        if not 0 <= parent_start < parent_stop <= parent.grid.n_cells:
            raise ValueError("Requested parent range lies outside the parent patch")
        for sibling in parent.children:
            if sibling.parent_range is None:
                raise RuntimeError("Hierarchy invariant violated: child has no parent range")
            sibling_start, sibling_stop = sibling.parent_range
            if max(parent_start, sibling_start) < min(parent_stop, sibling_stop):
                raise ValueError("Sibling patches cannot overlap")

        parent_edges = parent.grid.cell_edges
        n_fine = (parent_stop - parent_start) * self.refinement_ratio
        child_grid = UniformGrid1D(
            float(parent_edges[parent_start]),
            float(parent_edges[parent_stop]),
            n_fine,
        )
        if values is None:
            child_values = prolong_piecewise_constant(
                parent.values[parent_start:parent_stop], self.refinement_ratio
            )
        else:
            child_values = np.asarray(values, dtype=float)

        child = Patch1D(
            grid=child_grid,
            level=parent.level + 1,
            values=child_values,
            parent=parent,
            parent_range=(parent_start, parent_stop),
            refinement_ratio=self.refinement_ratio,
        )
        parent.children.append(child)
        return child

    def restrict_patch(self, patch: Patch1D) -> None:
        """Average a child onto the covered cells of its parent in place."""

        if patch.parent is None or patch not in self.patches:
            raise ValueError("Can only restrict a child belonging to this hierarchy")
        if patch.parent_range is None:
            raise RuntimeError("Hierarchy invariant violated: child has no parent range")
        start, stop = patch.parent_range
        patch.parent.values[start:stop] = restrict_cell_averages(
            patch.values, patch.refinement_ratio
        )

    def remove_patch(self, patch: Patch1D, restrict: bool = True) -> None:
        """Remove a leaf child, optionally restricting it before derefinement."""

        if patch.parent is None or patch not in self.patches:
            raise ValueError("Can only remove a child belonging to this hierarchy")
        if patch.children:
            raise ValueError("Cannot remove a patch that still has children")
        if restrict:
            self.restrict_patch(patch)
        patch.parent.children.remove(patch)

    @property
    def patches(self) -> tuple[Patch1D, ...]:
        """All patches in deterministic depth-first order."""

        result: list[Patch1D] = []

        def visit(patch: Patch1D) -> None:
            result.append(patch)
            for child in patch.children:
                visit(child)

        visit(self.root)
        return tuple(result)

    def patches_at_level(self, level: int) -> tuple[Patch1D, ...]:
        """Return all patches at a requested refinement level."""

        return tuple(patch for patch in self.patches if patch.level == level)

    @property
    def n_stored_cells(self) -> int:
        """Total cells stored across every level, including covered cells."""

        return sum(patch.n_valid_cells for patch in self.patches)

    @property
    def n_active_cells(self) -> int:
        """Leaf-cell count, excluding parent cells covered by finer patches."""

        def count(patch: Patch1D) -> int:
            covered = 0
            for child in patch.children:
                if child.parent_range is None:
                    raise RuntimeError("Hierarchy invariant violated: child has no parent range")
                start, stop = child.parent_range
                covered += stop - start
            return patch.n_valid_cells - covered + sum(count(child) for child in patch.children)

        return count(self.root)
