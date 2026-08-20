"""Accuracy and conservation diagnostics."""

from amr.diagnostics.conservation import (
    composite_mass,
    composite_mass_2d,
    total_mass,
    total_mass_2d,
)
from amr.diagnostics.errors import (
    ErrorNorms,
    composite_cell_average_error_norms,
    composite_cell_average_error_norms_2d,
    composite_error_norms,
    composite_error_norms_2d,
    error_norms,
)
from amr.diagnostics.variation import total_variation

__all__ = [
    "ErrorNorms",
    "composite_error_norms",
    "composite_error_norms_2d",
    "composite_cell_average_error_norms",
    "composite_cell_average_error_norms_2d",
    "composite_mass",
    "composite_mass_2d",
    "error_norms",
    "total_mass",
    "total_mass_2d",
    "total_variation",
]
