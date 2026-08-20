"""Initial conditions and exact solutions used by validation benchmarks."""

from amr.benchmarks.advection import gaussian, sinusoid, square_pulse, translated_profile
from amr.benchmarks.advection2d import periodic_gaussian_2d, translated_gaussian_2d
from amr.benchmarks.burgers import exact_smooth_solution, smooth_periodic_profile
from amr.benchmarks.diffusion import (
    periodic_gaussian_diffusion,
    periodic_gaussian_diffusion_cell_averages,
    periodic_sine_diffusion_cell_averages,
)
from amr.benchmarks.diffusion2d import (
    periodic_fourier_diffusion_2d_cell_averages,
    periodic_gaussian_diffusion_2d_cell_averages,
)

__all__ = [
    "exact_smooth_solution",
    "gaussian",
    "periodic_gaussian_diffusion",
    "periodic_gaussian_diffusion_cell_averages",
    "periodic_sine_diffusion_cell_averages",
    "periodic_gaussian_2d",
    "periodic_fourier_diffusion_2d_cell_averages",
    "periodic_gaussian_diffusion_2d_cell_averages",
    "sinusoid",
    "smooth_periodic_profile",
    "square_pulse",
    "translated_profile",
    "translated_gaussian_2d",
]
