"""Rectangular two-dimensional finite-volume patches."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amr.grid.grid2d import UniformGrid2D

IndexRange2D = tuple[tuple[int, int], tuple[int, int]]


@dataclass(slots=True, eq=False)
class Patch2D:
    """A rectangular patch and its parent/child metadata.

    ``parent_range`` is ``((y_start, y_stop), (x_start, x_stop))`` in the
    parent's local array indexing. Values contain valid cells only.
    """

    grid: UniformGrid2D
    level: int
    values: NDArray[np.float64]
    parent: Patch2D | None = field(default=None, repr=False)
    parent_range: IndexRange2D | None = None
    refinement_ratio: int = 1
    children: list[Patch2D] = field(default_factory=list, repr=False)

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
        (y_start, y_stop), (x_start, x_stop) = self.parent_range
        if not (
            0 <= y_start < y_stop <= self.parent.grid.ny
            and 0 <= x_start < x_stop <= self.parent.grid.nx
        ):
            raise ValueError("parent_range must describe a non-empty box inside the parent")
        if isinstance(self.refinement_ratio, bool) or not isinstance(
            self.refinement_ratio, (int, np.integer)
        ):
            raise TypeError("refinement_ratio must be an integer")
        if self.refinement_ratio < 2:
            raise ValueError("A child refinement ratio must be at least 2")
        expected_shape = (
            (y_stop - y_start) * self.refinement_ratio,
            (x_stop - x_start) * self.refinement_ratio,
        )
        if self.grid.shape != expected_shape:
            raise ValueError("Child shape is inconsistent with its parent range and ratio")

        parent = self.parent.grid
        aligned = (
            np.isclose(self.grid.x_min, parent.x_edges[x_start])
            and np.isclose(self.grid.x_max, parent.x_edges[x_stop])
            and np.isclose(self.grid.y_min, parent.y_edges[y_start])
            and np.isclose(self.grid.y_max, parent.y_edges[y_stop])
        )
        if not aligned:
            raise ValueError("Child physical bounds must coincide with parent cell edges")

    @property
    def n_valid_cells(self) -> int:
        """Number of valid finite-volume cells in the patch."""

        return self.grid.nx * self.grid.ny

    @property
    def physical_bounds(self) -> tuple[float, float, float, float]:
        """Patch extent as ``(x_min, x_max, y_min, y_max)``."""

        return self.grid.x_min, self.grid.x_max, self.grid.y_min, self.grid.y_max

    def set_values(self, values: ArrayLike) -> None:
        """Replace valid-cell values after shape and finiteness validation."""

        array = np.asarray(values, dtype=float)
        self.grid.validate_field(array)
        self.values = np.array(array, copy=True)
