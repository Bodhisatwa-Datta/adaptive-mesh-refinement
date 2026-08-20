"""PDE-independent flux-register correction for one-level 2D hierarchies."""

import numpy as np

from amr.grid.hierarchy2d import AMRHierarchy2D
from amr.grid.patch2d import Patch2D


def apply_reflux_2d(
    hierarchy: AMRHierarchy2D,
    children: tuple[Patch2D, ...],
    coarse_fluxes: tuple[np.ndarray, np.ndarray],
    fine_fluxes: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Replace coarse interface fluxes with face-averaged fine fluxes."""

    root = hierarchy.root
    covered = np.zeros(root.grid.shape, dtype=bool)
    for child in children:
        if child.parent_range is None:
            raise RuntimeError("Hierarchy invariant violated: missing parent range")
        (y_start, y_stop), (x_start, x_stop) = child.parent_range
        covered[y_start:y_stop, x_start:x_stop] = True

    ratio = hierarchy.refinement_ratio
    coarse_x, coarse_y = coarse_fluxes
    for child, (fine_x, fine_y) in zip(children, fine_fluxes):
        if child.parent_range is None:
            raise RuntimeError("Hierarchy invariant violated: missing parent range")
        (y_start, y_stop), (x_start, x_stop) = child.parent_range
        fine_left = fine_x[:, 0].reshape(-1, ratio).mean(axis=1)
        fine_right = fine_x[:, -1].reshape(-1, ratio).mean(axis=1)
        fine_bottom = fine_y[0, :].reshape(-1, ratio).mean(axis=1)
        fine_top = fine_y[-1, :].reshape(-1, ratio).mean(axis=1)

        left_x = (x_start - 1) % root.grid.nx
        right_x = x_stop % root.grid.nx
        for local_y, parent_y in enumerate(range(y_start, y_stop)):
            if not covered[parent_y, left_x]:
                root.values[parent_y, left_x] += (
                    coarse_x[parent_y, x_start] - fine_left[local_y]
                ) / root.grid.dx
            if not covered[parent_y, right_x]:
                root.values[parent_y, right_x] += (
                    fine_right[local_y] - coarse_x[parent_y, x_stop]
                ) / root.grid.dx

        bottom_y = (y_start - 1) % root.grid.ny
        top_y = y_stop % root.grid.ny
        for local_x, parent_x in enumerate(range(x_start, x_stop)):
            if not covered[bottom_y, parent_x]:
                root.values[bottom_y, parent_x] += (
                    coarse_y[y_start, parent_x] - fine_bottom[local_x]
                ) / root.grid.dy
            if not covered[top_y, parent_x]:
                root.values[top_y, parent_x] += (
                    fine_top[local_x] - coarse_y[y_stop, parent_x]
                ) / root.grid.dy
