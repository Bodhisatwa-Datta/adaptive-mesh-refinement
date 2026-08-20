"""Periodic profiles and exact translations for two-dimensional advection."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def periodic_gaussian_2d(
    x: ArrayLike,
    y: ArrayLike,
    *,
    centre: tuple[float, float] = (0.5, 0.5),
    width: tuple[float, float] = (0.08, 0.08),
    x_bounds: tuple[float, float] = (0.0, 1.0),
    y_bounds: tuple[float, float] = (0.0, 1.0),
) -> NDArray[np.float64]:
    """Evaluate a periodic anisotropic Gaussian using shortest distances."""

    if width[0] <= 0.0 or width[1] <= 0.0:
        raise ValueError("Gaussian widths must be positive")
    lx = x_bounds[1] - x_bounds[0]
    ly = y_bounds[1] - y_bounds[0]
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("Each upper domain bound must exceed its lower bound")
    dx = np.mod(np.asarray(x, dtype=float) - centre[0] + 0.5 * lx, lx) - 0.5 * lx
    dy = np.mod(np.asarray(y, dtype=float) - centre[1] + 0.5 * ly, ly) - 0.5 * ly
    return np.exp(-0.5 * ((dx / width[0]) ** 2 + (dy / width[1]) ** 2))


def translated_gaussian_2d(
    x: ArrayLike,
    y: ArrayLike,
    time: float,
    velocity: tuple[float, float],
    **profile_parameters: object,
) -> NDArray[np.float64]:
    """Evaluate the exact periodic translation of ``periodic_gaussian_2d``."""

    return periodic_gaussian_2d(
        np.asarray(x, dtype=float) - velocity[0] * time,
        np.asarray(y, dtype=float) - velocity[1] * time,
        **profile_parameters,
    )
