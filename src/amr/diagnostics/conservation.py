"""Conservation diagnostics for finite-volume fields."""

import numpy as np
from numpy.typing import ArrayLike

from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.grid.patch import Patch1D


def total_mass(values: ArrayLike, grid: UniformGrid1D) -> float:
    """Return the finite-volume integral ``sum(values) * dx``."""

    field = np.asarray(values, dtype=float)
    grid.validate_field(field)
    return float(np.sum(field) * grid.dx)


def composite_mass(hierarchy: AMRHierarchy1D) -> float:
    """Integrate the AMR composite solution, counting only leaf cells."""

    def patch_mass(patch: Patch1D) -> float:
        covered = np.zeros(patch.n_valid_cells, dtype=bool)
        for child in patch.children:
            if child.parent_range is None:
                raise RuntimeError("Hierarchy invariant violated: child has no parent range")
            start, stop = child.parent_range
            covered[start:stop] = True
        mass = float(np.sum(patch.values[~covered]) * patch.grid.dx)
        return mass + sum(patch_mass(child) for child in patch.children)

    return patch_mass(hierarchy.root)
