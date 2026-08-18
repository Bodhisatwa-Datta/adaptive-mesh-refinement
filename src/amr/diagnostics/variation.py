"""Discrete variation diagnostics for one-dimensional fields."""

import numpy as np
from numpy.typing import ArrayLike


def total_variation(values: ArrayLike, *, periodic: bool = True) -> float:
    """Return the discrete total variation of a finite one-dimensional field."""
    field = np.asarray(values, dtype=float)
    if field.ndim != 1 or field.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(field)):
        raise ValueError("values must be finite")

    variation = float(np.sum(np.abs(np.diff(field))))
    if periodic and field.size > 1:
        variation += float(abs(field[0] - field[-1]))
    return variation
