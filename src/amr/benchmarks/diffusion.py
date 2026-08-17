"""Analytical periodic Gaussian solution of the diffusion equation."""

from math import erf

import numpy as np
from numpy.typing import ArrayLike, NDArray


def periodic_gaussian_diffusion(
    x: ArrayLike,
    time: float,
    diffusivity: float,
    *,
    centre: float = 0.5,
    initial_width: float = 0.06,
    x_min: float = 0.0,
    x_max: float = 1.0,
    image_count: int = 4,
) -> NDArray[np.float64]:
    """Return a periodic image-sum Gaussian after diffusion time ``time``.

    The initial image-sum has unit central amplitude. Each image broadens from
    ``sigma_0`` to ``sqrt(sigma_0^2 + 2Dt)`` with the amplitude scaled by
    ``sigma_0/sigma(t)``, preserving total mass.
    """

    if not np.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and non-negative")
    if not np.isfinite(diffusivity) or diffusivity < 0.0:
        raise ValueError("diffusivity must be non-negative and finite")
    if not np.isfinite(initial_width) or initial_width <= 0.0:
        raise ValueError("initial_width must be positive and finite")
    if x_max <= x_min:
        raise ValueError("x_max must exceed x_min")
    if isinstance(image_count, bool) or not isinstance(image_count, (int, np.integer)):
        raise TypeError("image_count must be an integer")
    if image_count < 1:
        raise ValueError("image_count must be positive")

    coordinate = np.asarray(x, dtype=float)
    length = x_max - x_min
    width = np.sqrt(initial_width**2 + 2.0 * diffusivity * time)
    result = np.zeros_like(coordinate)
    for image in range(-image_count, image_count + 1):
        displacement = coordinate - centre + image * length
        result += np.exp(-0.5 * (displacement / width) ** 2)
    return (initial_width / width) * result


def periodic_gaussian_diffusion_cell_averages(
    cell_edges: ArrayLike,
    time: float,
    diffusivity: float,
    *,
    centre: float = 0.5,
    initial_width: float = 0.06,
    x_min: float = 0.0,
    x_max: float = 1.0,
    image_count: int = 4,
) -> NDArray[np.float64]:
    """Integrate the analytical periodic Gaussian over finite-volume cells."""

    edges = np.asarray(cell_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("cell_edges must be a one-dimensional array with at least two edges")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("cell_edges must be finite and strictly increasing")

    # Reuse the point-solution validation for all physical parameters.
    periodic_gaussian_diffusion(
        edges[:1],
        time,
        diffusivity,
        centre=centre,
        initial_width=initial_width,
        x_min=x_min,
        x_max=x_max,
        image_count=image_count,
    )
    width = np.sqrt(initial_width**2 + 2.0 * diffusivity * time)
    length = x_max - x_min
    integral = np.zeros(edges.size - 1, dtype=float)
    scale = np.sqrt(2.0) * width
    for image in range(-image_count, image_count + 1):
        shifted = edges - centre + image * length
        edge_erf = np.fromiter(
            (erf(float(value / scale)) for value in shifted),
            dtype=float,
            count=edges.size,
        )
        integral += initial_width * np.sqrt(np.pi / 2.0) * np.diff(edge_erf)
    return integral / np.diff(edges)
