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


def gradient_indicator_2d(
    values: ArrayLike,
    dx: float,
    dy: float,
    *,
    normalized: bool = False,
    epsilon: float = 1.0e-12,
    periodic: bool = True,
) -> NDArray[np.float64]:
    """Return the magnitude of a centred two-dimensional gradient indicator."""

    field = np.asarray(values, dtype=float)
    if field.ndim != 2 or min(field.shape) < 3:
        raise ValueError("2D gradient indicators require at least three cells per axis")
    if not np.all(np.isfinite(field)):
        raise ValueError("Field values must be finite")
    if not np.isfinite(dx) or dx <= 0.0 or not np.isfinite(dy) or dy <= 0.0:
        raise ValueError("dx and dy must be positive and finite")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")

    jump_x = np.empty_like(field)
    jump_y = np.empty_like(field)
    jump_x[:, 1:-1] = field[:, 2:] - field[:, :-2]
    jump_y[1:-1, :] = field[2:, :] - field[:-2, :]
    if periodic:
        jump_x[:, 0] = field[:, 1] - field[:, -1]
        jump_x[:, -1] = field[:, 0] - field[:, -2]
        jump_y[0, :] = field[1, :] - field[-1, :]
        jump_y[-1, :] = field[0, :] - field[-2, :]
    else:
        jump_x[:, 0] = 2.0 * (field[:, 1] - field[:, 0])
        jump_x[:, -1] = 2.0 * (field[:, -1] - field[:, -2])
        jump_y[0, :] = 2.0 * (field[1, :] - field[0, :])
        jump_y[-1, :] = 2.0 * (field[-1, :] - field[-2, :])

    if normalized:
        return np.hypot(jump_x, jump_y) / (np.abs(field) + epsilon)
    return np.hypot(jump_x / (2.0 * dx), jump_y / (2.0 * dy))


def buffer_flags_2d(
    flags: ArrayLike, n_buffer: int, *, periodic: bool = True
) -> NDArray[np.bool_]:
    """Expand 2D flags by a square ``n_buffer``-cell neighborhood."""

    original = np.asarray(flags, dtype=bool)
    if original.ndim != 2 or original.size == 0:
        raise ValueError("flags must be a non-empty two-dimensional array")
    if isinstance(n_buffer, bool) or not isinstance(n_buffer, (int, np.integer)):
        raise TypeError("n_buffer must be an integer")
    if n_buffer < 0:
        raise ValueError("n_buffer must be non-negative")
    expanded = original.copy()
    ny, nx = original.shape
    for y, x in np.argwhere(original):
        for offset_y in range(-n_buffer, n_buffer + 1):
            for offset_x in range(-n_buffer, n_buffer + 1):
                target_y = int(y + offset_y)
                target_x = int(x + offset_x)
                if periodic:
                    expanded[target_y % ny, target_x % nx] = True
                elif 0 <= target_y < ny and 0 <= target_x < nx:
                    expanded[target_y, target_x] = True
    return expanded


def flag_gradient_2d(
    values: ArrayLike,
    dx: float,
    dy: float,
    threshold: float,
    *,
    normalized: bool = False,
    epsilon: float = 1.0e-12,
    n_buffer: int = 0,
    periodic: bool = True,
) -> NDArray[np.bool_]:
    """Flag cells whose 2D gradient magnitude exceeds ``threshold``."""

    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be non-negative and finite")
    indicator = gradient_indicator_2d(
        values,
        dx,
        dy,
        normalized=normalized,
        epsilon=epsilon,
        periodic=periodic,
    )
    return buffer_flags_2d(indicator > threshold, n_buffer, periodic=periodic)


def bounding_box_from_flags_2d(
    flags: ArrayLike,
) -> tuple[int, int, int, int] | None:
    """Return ``(x_start, x_stop, y_start, y_stop)`` around all true cells."""

    flagged = np.asarray(flags, dtype=bool)
    if flagged.ndim != 2 or flagged.size == 0:
        raise ValueError("flags must be a non-empty two-dimensional array")
    coordinates = np.argwhere(flagged)
    if coordinates.size == 0:
        return None
    y_start, x_start = np.min(coordinates, axis=0)
    y_stop, x_stop = np.max(coordinates, axis=0) + 1
    return int(x_start), int(x_stop), int(y_start), int(y_stop)


def boxes_from_flags_2d(
    flags: ArrayLike, merge_gap: int = 0
) -> list[tuple[int, int, int, int]]:
    """Convert 8-connected flag components into merged, non-overlapping boxes.

    Boxes use ``(x_start, x_stop, y_start, y_stop)``. Connectivity does not
    wrap across periodic boundaries because rectangular patches cannot cross
    the selected domain origin; edge components therefore remain separate.
    """

    flagged = np.asarray(flags, dtype=bool)
    if flagged.ndim != 2 or flagged.size == 0:
        raise ValueError("flags must be a non-empty two-dimensional array")
    if isinstance(merge_gap, bool) or not isinstance(merge_gap, (int, np.integer)):
        raise TypeError("merge_gap must be an integer")
    if merge_gap < 0:
        raise ValueError("merge_gap must be non-negative")

    visited = np.zeros_like(flagged)
    ny, nx = flagged.shape
    boxes: list[tuple[int, int, int, int]] = []
    for start_y, start_x in np.argwhere(flagged):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        component: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    if offset_y == 0 and offset_x == 0:
                        continue
                    neighbor_y = y + offset_y
                    neighbor_x = x + offset_x
                    if (
                        0 <= neighbor_y < ny
                        and 0 <= neighbor_x < nx
                        and flagged[neighbor_y, neighbor_x]
                        and not visited[neighbor_y, neighbor_x]
                    ):
                        visited[neighbor_y, neighbor_x] = True
                        stack.append((neighbor_y, neighbor_x))
        coordinates = np.asarray(component)
        y_start, x_start = np.min(coordinates, axis=0)
        y_stop, x_stop = np.max(coordinates, axis=0) + 1
        boxes.append((int(x_start), int(x_stop), int(y_start), int(y_stop)))

    boxes.sort()
    changed = True
    while changed:
        changed = False
        for first_index, first in enumerate(boxes):
            for second_index in range(first_index + 1, len(boxes)):
                second = boxes[second_index]
                gap_x = max(0, max(first[0], second[0]) - min(first[1], second[1]))
                gap_y = max(0, max(first[2], second[2]) - min(first[3], second[3]))
                if gap_x <= merge_gap and gap_y <= merge_gap:
                    boxes[first_index] = (
                        min(first[0], second[0]),
                        max(first[1], second[1]),
                        min(first[2], second[2]),
                        max(first[3], second[3]),
                    )
                    del boxes[second_index]
                    boxes.sort()
                    changed = True
                    break
            if changed:
                break
    return boxes
