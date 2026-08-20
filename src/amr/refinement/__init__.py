"""Refinement indicators, flag handling, and conservative level transfers."""

from amr.refinement.criteria import (
    buffer_flags,
    buffer_flags_2d,
    bounding_box_from_flags_2d,
    boxes_from_flags_2d,
    flag_gradient,
    flag_gradient_2d,
    gradient_indicator,
    gradient_indicator_2d,
    regions_from_flags,
)
from amr.refinement.prolongation import (
    prolong_conservative_linear,
    prolong_conservative_quadratic,
    prolong_conservative_quadratic_2d,
    prolong_piecewise_constant,
    prolong_piecewise_constant_2d,
)
from amr.refinement.restriction import restrict_cell_averages, restrict_cell_averages_2d

__all__ = [
    "buffer_flags",
    "buffer_flags_2d",
    "bounding_box_from_flags_2d",
    "boxes_from_flags_2d",
    "flag_gradient",
    "flag_gradient_2d",
    "gradient_indicator",
    "gradient_indicator_2d",
    "prolong_piecewise_constant",
    "prolong_piecewise_constant_2d",
    "prolong_conservative_linear",
    "prolong_conservative_quadratic",
    "prolong_conservative_quadratic_2d",
    "regions_from_flags",
    "restrict_cell_averages",
    "restrict_cell_averages_2d",
]
