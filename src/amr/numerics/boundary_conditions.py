"""Boundary-condition utilities independent of a particular PDE solver."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


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

