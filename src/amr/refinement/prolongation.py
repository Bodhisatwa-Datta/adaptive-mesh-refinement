"""Conservative interpolation from coarse to fine finite-volume cells."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amr.numerics.reconstruction import monotonized_central_slopes


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

    if limit_slopes:
        slopes = monotonized_central_slopes(coarse, periodic=periodic)
    elif periodic:
        slopes = 0.5 * (np.roll(coarse, -1) - np.roll(coarse, 1))
    else:
        slopes = np.gradient(coarse)
    offsets = (np.arange(refinement_ratio, dtype=float) + 0.5) / refinement_ratio - 0.5
    return (coarse[:, None] + slopes[:, None] * offsets[None, :]).reshape(-1)


def prolong_conservative_quadratic(
    coarse_values: ArrayLike,
    refinement_ratio: int = 2,
    *,
    periodic: bool = True,
) -> NDArray[np.float64]:
    """Conservatively prolong smooth cell averages with a quadratic.

    The reconstruction matches three neighbouring coarse-cell averages and is
    integrated over each fine child. It is not monotonicity preserving.
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
        left = np.roll(coarse, 1)
        right = np.roll(coarse, -1)
    else:
        left_boundary = 3.0 * coarse[0] - 3.0 * coarse[1] + coarse[2]
        right_boundary = coarse[-3] - 3.0 * coarse[-2] + 3.0 * coarse[-1]
        left = np.concatenate(([left_boundary], coarse[:-1]))
        right = np.concatenate((coarse[1:], [right_boundary]))

    slopes = 0.5 * (right - left)
    curvature = 0.5 * (right - 2.0 * coarse + left)
    offsets = (np.arange(refinement_ratio, dtype=float) + 0.5) / refinement_ratio - 0.5
    average_correction = (1.0 / refinement_ratio**2 - 1.0) / 12.0
    fine = (
        coarse[:, None]
        + slopes[:, None] * offsets[None, :]
        + curvature[:, None] * (offsets[None, :] ** 2 + average_correction)
    )
    return fine.reshape(-1)
