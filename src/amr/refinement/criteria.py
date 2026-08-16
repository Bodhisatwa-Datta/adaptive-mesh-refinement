"""Gradient indicators and deterministic conversion of flags into regions."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _field(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 3:
        raise ValueError("Gradient indicators require at least three 1D cells")
    if not np.all(np.isfinite(array)):
        raise ValueError("Field values must be finite")
    return array


def gradient_indicator(
    values: ArrayLike,
    dx: float,
    *,
    normalized: bool = False,
    epsilon: float = 1.0e-12,
    periodic: bool = True,
) -> NDArray[np.float64]:
    """Return a centred absolute or normalized gradient indicator.

    The absolute form is ``|U[i+1]-U[i-1]|/(2 dx)``. The normalized form is
    ``|U[i+1]-U[i-1]|/(|U[i]|+epsilon)`` and is dimensionless. Non-periodic
    edge values use one-sided first differences.
    """

    field = _field(values)
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be positive and finite")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")

    jump = np.empty_like(field)
    jump[1:-1] = np.abs(field[2:] - field[:-2])
    if periodic:
        jump[0] = abs(field[1] - field[-1])
        jump[-1] = abs(field[0] - field[-2])
    else:
        jump[0] = 2.0 * abs(field[1] - field[0])
        jump[-1] = 2.0 * abs(field[-1] - field[-2])

    if normalized:
        return jump / (np.abs(field) + epsilon)
    return jump / (2.0 * dx)


def buffer_flags(
    flags: ArrayLike, n_buffer: int, *, periodic: bool = True
) -> NDArray[np.bool_]:
    """Expand every flagged cell by ``n_buffer`` cells on both sides."""

    result = np.asarray(flags, dtype=bool)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("flags must be a non-empty one-dimensional array")
    if isinstance(n_buffer, bool) or not isinstance(n_buffer, (int, np.integer)):
        raise TypeError("n_buffer must be an integer")
    if n_buffer < 0:
        raise ValueError("n_buffer must be non-negative")
    expanded = result.copy()
    flagged = np.flatnonzero(result)
    for index in flagged:
        offsets = np.arange(index - n_buffer, index + n_buffer + 1)
        if periodic:
            expanded[np.mod(offsets, result.size)] = True
        else:
            expanded[np.clip(offsets, 0, result.size - 1)] = True
    return expanded


def flag_gradient(
    values: ArrayLike,
    dx: float,
    threshold: float,
    *,
    normalized: bool = False,
    epsilon: float = 1.0e-12,
    n_buffer: int = 0,
    periodic: bool = True,
) -> NDArray[np.bool_]:
    """Flag cells whose gradient indicator strictly exceeds ``threshold``."""

    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be non-negative and finite")
    indicator = gradient_indicator(
        values,
        dx,
        normalized=normalized,
        epsilon=epsilon,
        periodic=periodic,
    )
    return buffer_flags(indicator > threshold, n_buffer, periodic=periodic)


def regions_from_flags(flags: ArrayLike, merge_gap: int = 0) -> list[tuple[int, int]]:
    """Convert flags to merged half-open index ranges.

    Periodic end regions remain separate because a single patch cannot cross the
    chosen domain origin. ``merge_gap`` joins regions separated by at most that
    many unflagged cells.
    """

    flagged = np.asarray(flags, dtype=bool)
    if flagged.ndim != 1 or flagged.size == 0:
        raise ValueError("flags must be a non-empty one-dimensional array")
    if isinstance(merge_gap, bool) or not isinstance(merge_gap, (int, np.integer)):
        raise TypeError("merge_gap must be an integer")
    if merge_gap < 0:
        raise ValueError("merge_gap must be non-negative")

    padded = np.concatenate(([False], flagged, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    regions: list[tuple[int, int]] = []
    for start, stop in zip(starts, stops):
        if regions and start - regions[-1][1] <= merge_gap:
            regions[-1] = (regions[-1][0], int(stop))
        else:
            regions.append((int(start), int(stop)))
    return regions

