"""Boundary-condition utilities independent of a particular PDE solver."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amr.grid.patch import Patch1D


def fill_periodic_ghost_cells(values: ArrayLike, n_ghost: int = 1) -> NDArray[np.float64]:
    """Return a copy of a 1D field padded by periodic ghost cells."""

    field = np.asarray(values, dtype=float)
    if field.ndim != 1:
        raise ValueError("Periodic ghost filling expects a one-dimensional field")
    if isinstance(n_ghost, bool) or not isinstance(n_ghost, (int, np.integer)):
        raise TypeError("n_ghost must be an integer")
    if n_ghost < 1:
        raise ValueError("n_ghost must be positive")
    if n_ghost > field.size:
        raise ValueError("n_ghost cannot exceed the number of physical cells")
    return np.concatenate((field[-n_ghost:], field, field[:n_ghost]))


def fill_coarse_fine_ghost_cells(
    patch: Patch1D,
    n_ghost: int = 1,
    *,
    periodic: bool = True,
    parent_values: ArrayLike | None = None,
    parent_interpolation: str = "piecewise_constant",
) -> NDArray[np.float64]:
    """Pad a level-one patch using fine neighbours before its coarse parent.

    Ghost-cell centres covered by any sibling patch use that sibling's fine
    data. Remaining centres use piecewise-constant or linear interpolation
    from the parent. Periodic wrapping is relative to the root domain. Physical
    boundary conditions other than periodic are deliberately not inferred here.
    """

    if patch.parent is None:
        raise ValueError("Coarse-fine ghost filling requires a child patch")
    if patch.parent.parent is not None:
        raise NotImplementedError("Coarse-fine ghost filling currently supports level one")
    if patch not in patch.parent.children:
        raise ValueError("patch is not attached to its parent")
    if isinstance(n_ghost, bool) or not isinstance(n_ghost, (int, np.integer)):
        raise TypeError("n_ghost must be an integer")
    if n_ghost < 1:
        raise ValueError("n_ghost must be positive")
    if n_ghost > patch.n_valid_cells:
        raise ValueError("n_ghost cannot exceed the number of valid patch cells")

    parent = patch.parent
    coarse_values = (
        parent.values if parent_values is None else np.asarray(parent_values, dtype=float)
    )
    parent.grid.validate_field(coarse_values)
    if parent_interpolation not in {"piecewise_constant", "linear"}:
        raise ValueError("parent_interpolation must be 'piecewise_constant' or 'linear'")
    left_centres = patch.grid.x_min - (
        np.arange(n_ghost, 0, -1, dtype=float) - 0.5
    ) * patch.grid.dx
    right_centres = patch.grid.x_max + (
        np.arange(n_ghost, dtype=float) + 0.5
    ) * patch.grid.dx

    def sample(coordinate: float) -> float:
        x = coordinate
        if periodic:
            x = parent.grid.x_min + np.mod(x - parent.grid.x_min, parent.grid.length)
        elif x < parent.grid.x_min or x >= parent.grid.x_max:
            raise ValueError("A non-periodic physical boundary value is required")

        # Prefer same-level data wherever a fine patch covers the ghost centre.
        for sibling in parent.children:
            if sibling.grid.x_min <= x < sibling.grid.x_max:
                index = int(np.floor((x - sibling.grid.x_min) / sibling.grid.dx))
                index = min(max(index, 0), sibling.grid.n_cells - 1)
                return float(sibling.values[index])

        if parent_interpolation == "piecewise_constant":
            index = int(np.floor((x - parent.grid.x_min) / parent.grid.dx))
            index = min(max(index, 0), parent.grid.n_cells - 1)
            return float(coarse_values[index])

        fractional_index = (
            x - (parent.grid.x_min + 0.5 * parent.grid.dx)
        ) / parent.grid.dx
        lower_unwrapped = int(np.floor(fractional_index))
        fraction = fractional_index - lower_unwrapped
        lower = lower_unwrapped % parent.grid.n_cells
        upper = (lower + 1) % parent.grid.n_cells
        return float((1.0 - fraction) * coarse_values[lower] + fraction * coarse_values[upper])

    left = np.fromiter((sample(x) for x in left_centres), dtype=float, count=n_ghost)
    right = np.fromiter((sample(x) for x in right_centres), dtype=float, count=n_ghost)
    return np.concatenate((left, patch.values, right))
