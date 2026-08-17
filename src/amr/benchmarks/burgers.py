"""Smooth pre-shock analytical solution for periodic Burgers' equation."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def smooth_periodic_profile(
    x: ArrayLike,
    mean: float = 0.5,
    amplitude: float = 0.2,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> NDArray[np.float64]:
    """Return ``mean + amplitude*sin(2*pi*(x-x_min)/L)``."""

    coordinate = np.asarray(x, dtype=float)
    return mean + amplitude * np.sin(2.0 * np.pi * (coordinate - x_min) / (x_max - x_min))


def smooth_periodic_derivative(
    x: ArrayLike,
    amplitude: float = 0.2,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> NDArray[np.float64]:
    """Spatial derivative of :func:`smooth_periodic_profile`."""

    coordinate = np.asarray(x, dtype=float)
    length = x_max - x_min
    return amplitude * (2.0 * np.pi / length) * np.cos(
        2.0 * np.pi * (coordinate - x_min) / length
    )


def shock_formation_time(amplitude: float = 0.2, length: float = 1.0) -> float:
    """Return the first shock time for the sinusoidal profile."""

    if amplitude <= 0.0 or length <= 0.0:
        raise ValueError("amplitude and length must be positive")
    return length / (2.0 * np.pi * amplitude)


def exact_smooth_solution(
    x: ArrayLike,
    time: float,
    *,
    mean: float = 0.5,
    amplitude: float = 0.2,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> NDArray[np.float64]:
    """Evaluate the characteristic solution before shock formation.

    The characteristic foot ``xi`` solves ``x = xi + u0(xi)*time``. Newton's
    method is well-conditioned while ``time`` is below the first shock time.
    """

    if not np.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and non-negative")
    shock_time = shock_formation_time(amplitude, x_max - x_min)
    if time >= shock_time:
        raise ValueError("The smooth characteristic solution is only valid before shock formation")

    coordinate = np.asarray(x, dtype=float)
    foot = coordinate - time * smooth_periodic_profile(
        coordinate, mean, amplitude, x_min, x_max
    )
    for _ in range(12):
        value = smooth_periodic_profile(foot, mean, amplitude, x_min, x_max)
        derivative = smooth_periodic_derivative(foot, amplitude, x_min, x_max)
        correction = (foot + time * value - coordinate) / (1.0 + time * derivative)
        foot -= correction
        if np.max(np.abs(correction)) < 2.0e-14:
            break
    return smooth_periodic_profile(foot, mean, amplitude, x_min, x_max)

