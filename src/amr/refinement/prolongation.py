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


def _minmod(*arguments: NDArray[np.float64]) -> NDArray[np.float64]:
    stacked = np.stack(arguments)
    same_sign = np.all(stacked > 0.0, axis=0) | np.all(stacked < 0.0, axis=0)
    magnitude = np.min(np.abs(stacked), axis=0)
    return np.where(same_sign, np.sign(stacked[0]) * magnitude, 0.0)


def prolong_conservative_linear(
    coarse_values: ArrayLike,
    refinement_ratio: int = 2,
    *,
    periodic: bool = True,
    limit_slopes: bool = True,
) -> NDArray[np.float64]:
    """Conservatively prolong cell averages with piecewise-linear reconstruction.

    Slopes use the monotonized-central limiter by default. Fine subcell offsets
    sum to zero, so every group of ``r`` children has exactly the parent average.
    """

    coarse = np.asarray(coarse_values, dtype=float)
    if coarse.ndim != 1 or coarse.size < 3:
        raise ValueError("coarse_values must contain at least three one-dimensional cells")
    if not np.all(np.isfinite(coarse)):
        raise ValueError("coarse_values must be finite")
    if isinstance(refinement_ratio, bool) or not isinstance(
        refinement_ratio, (int, np.integer)
    ):
        raise TypeError("refinement_ratio must be an integer")
    if refinement_ratio < 2:
        raise ValueError("refinement_ratio must be at least 2")

    if periodic:
        backward = coarse - np.roll(coarse, 1)
        forward = np.roll(coarse, -1) - coarse
    else:
        backward = np.empty_like(coarse)
        forward = np.empty_like(coarse)
        backward[1:] = coarse[1:] - coarse[:-1]
        forward[:-1] = coarse[1:] - coarse[:-1]
        backward[0] = forward[0]
        forward[-1] = backward[-1]
    centred = 0.5 * (backward + forward)
    slopes = _minmod(2.0 * backward, centred, 2.0 * forward) if limit_slopes else centred
    offsets = (np.arange(refinement_ratio, dtype=float) + 0.5) / refinement_ratio - 0.5
    return (coarse[:, None] + slopes[:, None] * offsets[None, :]).reshape(-1)
