"""Boundary-condition utilities independent of a particular PDE solver."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amr.grid.patch import Patch1D
from amr.grid.patch2d import Patch2D


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


def fill_periodic_ghost_cells_2d(
    values: ArrayLike, n_ghost: int = 1
) -> NDArray[np.float64]:
    """Pad a 2D field periodically in both coordinate directions."""

    field = np.asarray(values, dtype=float)
    if field.ndim != 2 or field.size == 0:
        raise ValueError("Periodic ghost filling expects a non-empty 2D field")
    if isinstance(n_ghost, bool) or not isinstance(n_ghost, (int, np.integer)):
        raise TypeError("n_ghost must be an integer")
    if n_ghost < 1:
        raise ValueError("n_ghost must be positive")
    if n_ghost > min(field.shape):
        raise ValueError("n_ghost cannot exceed either physical field dimension")
    return np.pad(field, n_ghost, mode="wrap")


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


def fill_coarse_fine_ghost_cells_2d(
    patch: Patch2D,
    n_ghost: int = 1,
    *,
    periodic: bool = True,
    parent_values: ArrayLike | None = None,
    parent_interpolation: str = "piecewise_constant",
) -> NDArray[np.float64]:
    """Pad a level-one rectangular patch using fine neighbors then its parent."""

    if patch.parent is None:
        raise ValueError("Coarse-fine ghost filling requires a child patch")
    if patch.parent.parent is not None:
        raise NotImplementedError("2D coarse-fine ghost filling supports level one")
    if patch not in patch.parent.children:
        raise ValueError("patch is not attached to its parent")
    if isinstance(n_ghost, bool) or not isinstance(n_ghost, (int, np.integer)):
        raise TypeError("n_ghost must be an integer")
    if n_ghost < 1:
        raise ValueError("n_ghost must be positive")
    if n_ghost > min(patch.grid.shape):
        raise ValueError("n_ghost cannot exceed either valid patch dimension")

    parent = patch.parent
    coarse_values = (
        parent.values if parent_values is None else np.asarray(parent_values, dtype=float)
    )
    parent.grid.validate_field(coarse_values)
    if parent_interpolation not in {"piecewise_constant", "bilinear"}:
        raise ValueError(
            "parent_interpolation must be 'piecewise_constant' or 'bilinear'"
        )
    x_coordinates = patch.grid.x_min + (
        np.arange(-n_ghost, patch.grid.nx + n_ghost, dtype=float) + 0.5
    ) * patch.grid.dx
    y_coordinates = patch.grid.y_min + (
        np.arange(-n_ghost, patch.grid.ny + n_ghost, dtype=float) + 0.5
    ) * patch.grid.dy
    ghosted = np.empty(
        (patch.grid.ny + 2 * n_ghost, patch.grid.nx + 2 * n_ghost),
        dtype=float,
    )
    ghosted[n_ghost:-n_ghost, n_ghost:-n_ghost] = patch.values

    def sample(x_coordinate: float, y_coordinate: float) -> float:
        x = x_coordinate
        y = y_coordinate
        if periodic:
            x_length = parent.grid.x_max - parent.grid.x_min
            y_length = parent.grid.y_max - parent.grid.y_min
            x = parent.grid.x_min + np.mod(x - parent.grid.x_min, x_length)
            y = parent.grid.y_min + np.mod(y - parent.grid.y_min, y_length)
        elif (
            x < parent.grid.x_min
            or x >= parent.grid.x_max
            or y < parent.grid.y_min
            or y >= parent.grid.y_max
        ):
            raise ValueError("A non-periodic physical boundary value is required")

        for sibling in parent.children:
            if (
                sibling.grid.x_min <= x < sibling.grid.x_max
                and sibling.grid.y_min <= y < sibling.grid.y_max
            ):
                fine_x = int(np.floor((x - sibling.grid.x_min) / sibling.grid.dx))
                fine_y = int(np.floor((y - sibling.grid.y_min) / sibling.grid.dy))
                fine_x = min(max(fine_x, 0), sibling.grid.nx - 1)
                fine_y = min(max(fine_y, 0), sibling.grid.ny - 1)
                return float(sibling.values[fine_y, fine_x])

        if parent_interpolation == "piecewise_constant":
            coarse_x = int(np.floor((x - parent.grid.x_min) / parent.grid.dx))
            coarse_y = int(np.floor((y - parent.grid.y_min) / parent.grid.dy))
            coarse_x = min(max(coarse_x, 0), parent.grid.nx - 1)
            coarse_y = min(max(coarse_y, 0), parent.grid.ny - 1)
            return float(coarse_values[coarse_y, coarse_x])

        fractional_x = (
            x - (parent.grid.x_min + 0.5 * parent.grid.dx)
        ) / parent.grid.dx
        fractional_y = (
            y - (parent.grid.y_min + 0.5 * parent.grid.dy)
        ) / parent.grid.dy
        lower_x_unwrapped = int(np.floor(fractional_x))
        lower_y_unwrapped = int(np.floor(fractional_y))
        weight_x = fractional_x - lower_x_unwrapped
        weight_y = fractional_y - lower_y_unwrapped
        lower_x = lower_x_unwrapped % parent.grid.nx
        upper_x = (lower_x + 1) % parent.grid.nx
        lower_y = lower_y_unwrapped % parent.grid.ny
        upper_y = (lower_y + 1) % parent.grid.ny
        lower_value = (
            (1.0 - weight_x) * coarse_values[lower_y, lower_x]
            + weight_x * coarse_values[lower_y, upper_x]
        )
        upper_value = (
            (1.0 - weight_x) * coarse_values[upper_y, lower_x]
            + weight_x * coarse_values[upper_y, upper_x]
        )
        return float((1.0 - weight_y) * lower_value + weight_y * upper_value)

    for ghost_y, y_coordinate in enumerate(y_coordinates):
        for ghost_x, x_coordinate in enumerate(x_coordinates):
            interior_y = n_ghost <= ghost_y < n_ghost + patch.grid.ny
            interior_x = n_ghost <= ghost_x < n_ghost + patch.grid.nx
            if not (interior_y and interior_x):
                ghosted[ghost_y, ghost_x] = sample(x_coordinate, y_coordinate)
    return ghosted
