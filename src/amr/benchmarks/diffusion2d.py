"""Analytical finite-volume Fourier solution for periodic 2D diffusion."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amr.benchmarks.diffusion import periodic_gaussian_diffusion_cell_averages


def periodic_gaussian_diffusion_2d_cell_averages(
    x_edges: ArrayLike,
    y_edges: ArrayLike,
    time: float,
    diffusivity: float,
    *,
    centre: tuple[float, float] = (0.5, 0.5),
    initial_width: tuple[float, float] = (0.06, 0.06),
    image_count: int = 4,
) -> NDArray[np.float64]:
    """Return exact cell averages of a separable periodic Gaussian."""

    x = np.asarray(x_edges, dtype=float)
    y = np.asarray(y_edges, dtype=float)
    average_x = periodic_gaussian_diffusion_cell_averages(
        x,
        time,
        diffusivity,
        centre=centre[0],
        initial_width=initial_width[0],
        x_min=float(x[0]),
        x_max=float(x[-1]),
        image_count=image_count,
    )
    average_y = periodic_gaussian_diffusion_cell_averages(
        y,
        time,
        diffusivity,
        centre=centre[1],
        initial_width=initial_width[1],
        x_min=float(y[0]),
        x_max=float(y[-1]),
        image_count=image_count,
    )
    return np.outer(average_y, average_x)


def periodic_fourier_diffusion_2d_cell_averages(
    x_edges: ArrayLike,
    y_edges: ArrayLike,
    time: float,
    diffusivity: float,
    *,
    mean: float = 0.5,
    amplitude: float = 0.2,
    modes: tuple[int, int] = (1, 2),
    phases: tuple[float, float] = (0.0, 0.0),
) -> NDArray[np.float64]:
    """Return exact averages of a separable, decaying sine-mode product."""

    x = _validate_edges(x_edges, "x_edges")
    y = _validate_edges(y_edges, "y_edges")
    if not np.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and non-negative")
    if not np.isfinite(diffusivity) or diffusivity < 0.0:
        raise ValueError("diffusivity must be non-negative and finite")
    if len(modes) != 2 or len(phases) != 2:
        raise ValueError("modes and phases must each contain two entries")
    if not np.all(np.isfinite((mean, amplitude, *phases))):
        raise ValueError("mean, amplitude, and phases must be finite")
    for mode in modes:
        if isinstance(mode, bool) or not isinstance(mode, (int, np.integer)):
            raise TypeError("modes must contain integers")
        if mode < 1:
            raise ValueError("modes must be positive")

    length_x = x[-1] - x[0]
    length_y = y[-1] - y[0]
    wave_x = 2.0 * np.pi * modes[0] / length_x
    wave_y = 2.0 * np.pi * modes[1] / length_y
    average_x = _sine_cell_averages(x, wave_x, phases[0])
    average_y = _sine_cell_averages(y, wave_y, phases[1])
    decay = np.exp(-diffusivity * (wave_x**2 + wave_y**2) * time)
    return mean + amplitude * decay * np.outer(average_y, average_x)


def _validate_edges(edges: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(edges, dtype=float)
    if array.ndim != 1 or array.size < 3:
        raise ValueError(f"{name} must be one-dimensional with at least three edges")
    if not np.all(np.isfinite(array)) or np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return array


def _sine_cell_averages(
    edges: NDArray[np.float64], wave_number: float, phase: float
) -> NDArray[np.float64]:
    angles = wave_number * (edges - edges[0]) + phase
    return (np.cos(angles[:-1]) - np.cos(angles[1:])) / (
        wave_number * np.diff(edges)
    )
