"""Conservative interpolation from coarse to fine finite-volume cells."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def prolong_piecewise_constant(
    coarse_values: ArrayLike, refinement_ratio: int = 2
) -> NDArray[np.float64]:
    """Split every coarse average into equal fine-cell averages.

    If coarse cell ``i`` has average ``U_i``, all ``r`` fine children receive
    ``U_i``. Their arithmetic mean is therefore exactly the parent average.
    """

    coarse = np.asarray(coarse_values, dtype=float)
    if coarse.ndim != 1 or coarse.size == 0:
        raise ValueError("coarse_values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(coarse)):
        raise ValueError("coarse_values must be finite")
    if isinstance(refinement_ratio, bool) or not isinstance(
        refinement_ratio, (int, np.integer)
    ):
        raise TypeError("refinement_ratio must be an integer")
    if refinement_ratio < 2:
        raise ValueError("refinement_ratio must be at least 2")
    return np.repeat(coarse, refinement_ratio)

