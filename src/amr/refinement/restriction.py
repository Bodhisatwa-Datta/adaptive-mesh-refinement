"""Conservative averaging from fine to coarse finite-volume cells."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def restrict_cell_averages(
    fine_values: ArrayLike, refinement_ratio: int = 2
) -> NDArray[np.float64]:
    """Average each consecutive group of ``r`` fine cells onto its parent."""

    fine = np.asarray(fine_values, dtype=float)
    if fine.ndim != 1 or fine.size == 0:
        raise ValueError("fine_values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(fine)):
        raise ValueError("fine_values must be finite")
    if isinstance(refinement_ratio, bool) or not isinstance(
        refinement_ratio, (int, np.integer)
    ):
        raise TypeError("refinement_ratio must be an integer")
    if refinement_ratio < 2:
        raise ValueError("refinement_ratio must be at least 2")
    if fine.size % refinement_ratio != 0:
        raise ValueError("Fine cell count must be divisible by refinement_ratio")
    return np.mean(fine.reshape(-1, refinement_ratio), axis=1)


def restrict_cell_averages_2d(
    fine_values: ArrayLike, refinement_ratio: int = 2
) -> NDArray[np.float64]:
    """Average each ``r x r`` fine block onto its two-dimensional parent."""

    fine = np.asarray(fine_values, dtype=float)
    if fine.ndim != 2 or fine.size == 0:
        raise ValueError("fine_values must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(fine)):
        raise ValueError("fine_values must be finite")
    if isinstance(refinement_ratio, bool) or not isinstance(
        refinement_ratio, (int, np.integer)
    ):
        raise TypeError("refinement_ratio must be an integer")
    if refinement_ratio < 2:
        raise ValueError("refinement_ratio must be at least 2")
    ny, nx = fine.shape
    if ny % refinement_ratio != 0 or nx % refinement_ratio != 0:
        raise ValueError("Both fine dimensions must be divisible by refinement_ratio")
    blocked = fine.reshape(
        ny // refinement_ratio,
        refinement_ratio,
        nx // refinement_ratio,
        refinement_ratio,
    )
    return np.mean(blocked, axis=(1, 3))
