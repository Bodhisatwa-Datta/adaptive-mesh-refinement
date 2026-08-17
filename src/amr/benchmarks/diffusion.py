"""Analytical periodic Gaussian solution of the diffusion equation."""

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

