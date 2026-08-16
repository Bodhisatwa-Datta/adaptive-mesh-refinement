"""Conservation diagnostics for finite-volume fields."""

import numpy as np
from numpy.typing import ArrayLike

from amr.grid.grid1d import UniformGrid1D


def total_mass(values: ArrayLike, grid: UniformGrid1D) -> float:
    """Return the finite-volume integral ``sum(values) * dx``."""

    field = np.asarray(values, dtype=float)
    grid.validate_field(field)
    return float(np.sum(field) * grid.dx)

