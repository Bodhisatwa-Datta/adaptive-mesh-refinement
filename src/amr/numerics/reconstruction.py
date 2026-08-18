"""Reusable slope limiters for finite-volume reconstruction."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def minmod(*arguments: ArrayLike) -> NDArray[np.float64]:
    """Return the elementwise minimum-modulus value for equal-shaped arrays."""

    if not arguments:
        raise ValueError("minmod requires at least one argument")
    arrays = [np.asarray(argument, dtype=float) for argument in arguments]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("all minmod arguments must have equal shape")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("minmod arguments must be finite")
    stacked = np.stack(arrays)
    same_sign = np.all(stacked > 0.0, axis=0) | np.all(stacked < 0.0, axis=0)
    return np.where(
        same_sign,
        np.sign(stacked[0]) * np.min(np.abs(stacked), axis=0),
        0.0,
    )


def monotonized_central_slopes(
    values: ArrayLike,
    *,
    periodic: bool = True,
) -> NDArray[np.float64]:
    """Return dimensionless monotonized-central slopes for 1D cell data."""

    field = np.asarray(values, dtype=float)
    if field.ndim != 1 or field.size < 2:
        raise ValueError("values must contain at least two one-dimensional cells")
    if not np.all(np.isfinite(field)):
        raise ValueError("values must be finite")
    if periodic:
        backward = field - np.roll(field, 1)
        forward = np.roll(field, -1) - field
    else:
        backward = np.empty_like(field)
        forward = np.empty_like(field)
        backward[1:] = field[1:] - field[:-1]
        forward[:-1] = field[1:] - field[:-1]
        backward[0] = forward[0]
        forward[-1] = backward[-1]
    centred = 0.5 * (backward + forward)
    return minmod(2.0 * backward, centred, 2.0 * forward)
