"""One-dimensional finite-volume patches and their parent/child metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amr.grid.grid1d import UniformGrid1D


@dataclass(slots=True, eq=False)
class Patch1D:
    """A valid rectangular segment of one refinement level.

    ``parent_range`` is a half-open range in the parent's local cell indexing.
    Values contain valid cells only; ghost cells are a later milestone.
    """

    grid: UniformGrid1D
    level: int
    values: NDArray[np.float64]
    parent: Patch1D | None = field(default=None, repr=False)
    parent_range: tuple[int, int] | None = None
    refinement_ratio: int = 1
    children: list[Patch1D] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.level, bool) or not isinstance(self.level, (int, np.integer)):
            raise TypeError("level must be an integer")
        if self.level < 0:
            raise ValueError("level must be non-negative")
        self.values = np.array(self.values, dtype=float, copy=True)
        self.grid.validate_field(self.values)

        if self.parent is None:
            if self.level != 0 or self.parent_range is not None or self.refinement_ratio != 1:
                raise ValueError("A root patch must have level 0, no parent range, and ratio 1")
            return

        if self.level != self.parent.level + 1:
            raise ValueError("A child patch must be exactly one level finer than its parent")
        if self.parent_range is None:
            raise ValueError("A child patch requires a parent cell range")
        start, stop = self.parent_range
        if not 0 <= start < stop <= self.parent.grid.n_cells:
            raise ValueError("parent_range must be a non-empty range inside the parent")
        if isinstance(self.refinement_ratio, bool) or not isinstance(
            self.refinement_ratio, (int, np.integer)
        ):
            raise TypeError("refinement_ratio must be an integer")
        if self.refinement_ratio < 2:
            raise ValueError("A child refinement ratio must be at least 2")
        if self.grid.n_cells != (stop - start) * self.refinement_ratio:
            raise ValueError("Child cell count is inconsistent with its parent range and ratio")

        parent_edges = self.parent.grid.cell_edges
        if not np.isclose(self.grid.x_min, parent_edges[start]) or not np.isclose(
            self.grid.x_max, parent_edges[stop]
        ):
            raise ValueError("Child physical bounds must coincide with parent cell edges")

    @property
    def n_valid_cells(self) -> int:
        """Number of valid finite-volume cells in the patch."""

        return self.grid.n_cells

    @property
    def physical_bounds(self) -> tuple[float, float]:
        """Patch extent as ``(x_min, x_max)``."""

        return self.grid.x_min, self.grid.x_max

    @property
    def parent_start(self) -> int | None:
        """First covered parent cell, or ``None`` for the root patch."""

        return None if self.parent_range is None else self.parent_range[0]

    @property
    def parent_stop(self) -> int | None:
        """Exclusive end of the covered parent range, or ``None`` for the root."""

        return None if self.parent_range is None else self.parent_range[1]

    def set_values(self, values: ArrayLike) -> None:
        """Replace valid-cell values after shape and finiteness validation."""

        array = np.asarray(values, dtype=float)
        self.grid.validate_field(array)
        self.values = np.array(array, copy=True)

