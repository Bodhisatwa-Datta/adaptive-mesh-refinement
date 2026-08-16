"""Periodic profiles and exact solutions for constant linear advection."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def periodic_coordinate(x: ArrayLike, x_min: float, x_max: float) -> NDArray[np.float64]:
    """Wrap coordinates into the half-open interval ``[x_min, x_max)``."""

    return x_min + np.mod(np.asarray(x, dtype=float) - x_min, x_max - x_min)


def gaussian(
    x: ArrayLike,
    centre: float = 0.5,
    width: float = 0.07,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> NDArray[np.float64]:
    """Periodic Gaussian defined using the shortest distance to ``centre``."""

    if width <= 0.0:
        raise ValueError("width must be positive")
    length = x_max - x_min
    displacement = np.mod(np.asarray(x, dtype=float) - centre + 0.5 * length, length) - 0.5 * length
    return np.exp(-0.5 * (displacement / width) ** 2)


def square_pulse(
    x: ArrayLike,
    centre: float = 0.5,
    width: float = 0.2,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> NDArray[np.float64]:
    """Unit-height periodic square pulse."""

    length = x_max - x_min
    displacement = np.mod(np.asarray(x, dtype=float) - centre + 0.5 * length, length) - 0.5 * length
    return (np.abs(displacement) <= 0.5 * width).astype(float)


def sinusoid(x: ArrayLike, x_min: float = 0.0, x_max: float = 1.0) -> NDArray[np.float64]:
    """One-period smooth sinusoidal profile."""

    return np.sin(2.0 * np.pi * (np.asarray(x, dtype=float) - x_min) / (x_max - x_min))


def translated_profile(
    x: ArrayLike,
    time: float,
    velocity: float,
    profile: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> NDArray[np.float64]:
    """Evaluate the exact periodic translation ``u(x,t)=u_0(x-a t)``."""

    departure_points = periodic_coordinate(np.asarray(x) - velocity * time, x_min, x_max)
    return np.asarray(profile(departure_points), dtype=float)

