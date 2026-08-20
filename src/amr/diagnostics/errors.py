"""Discrete error norms."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from amr.grid.hierarchy import AMRHierarchy1D
from amr.grid.hierarchy2d import AMRHierarchy2D
from amr.grid.patch import Patch1D
from amr.grid.patch2d import Patch2D


@dataclass(frozen=True, slots=True)
class ErrorNorms:
    """Mean L1, root-mean-square L2, and maximum norms."""

    l1: float
    l2: float
    linf: float


def error_norms(numerical: ArrayLike, exact: ArrayLike) -> ErrorNorms:
    """Calculate discrete error norms for arrays of equal, non-zero shape."""

    numerical_array = np.asarray(numerical, dtype=float)
    exact_array = np.asarray(exact, dtype=float)
    if numerical_array.shape != exact_array.shape:
        raise ValueError("numerical and exact arrays must have the same shape")
    if numerical_array.size == 0:
        raise ValueError("error norms require at least one value")
    difference = np.abs(numerical_array - exact_array)
    return ErrorNorms(
        l1=float(np.mean(difference)),
        l2=float(np.sqrt(np.mean(difference**2))),
        linf=float(np.max(difference)),
    )


def composite_error_norms(
    hierarchy: AMRHierarchy1D,
    exact: Callable[[np.ndarray], np.ndarray],
) -> ErrorNorms:
    """Calculate physical-space error norms over AMR leaf cells."""

    l1_integral = 0.0
    l2_integral = 0.0
    linf = 0.0

    def accumulate(patch: Patch1D) -> None:
        nonlocal l1_integral, l2_integral, linf
        covered = np.zeros(patch.n_valid_cells, dtype=bool)
        for child in patch.children:
            if child.parent_range is None:
                raise RuntimeError("Hierarchy invariant violated: child has no parent range")
            start, stop = child.parent_range
            covered[start:stop] = True
        if np.any(~covered):
            difference = np.abs(
                patch.values[~covered] - np.asarray(exact(patch.grid.cell_centres[~covered]))
            )
            l1_integral += float(np.sum(difference) * patch.grid.dx)
            l2_integral += float(np.sum(difference**2) * patch.grid.dx)
            linf = max(linf, float(np.max(difference)))
        for child in patch.children:
            accumulate(child)

    accumulate(hierarchy.root)
    length = hierarchy.root.grid.length
    return ErrorNorms(
        l1=l1_integral / length,
        l2=float(np.sqrt(l2_integral / length)),
        linf=linf,
    )


def composite_cell_average_error_norms(
    hierarchy: AMRHierarchy1D,
    exact_cell_averages: Callable[[np.ndarray], np.ndarray],
) -> ErrorNorms:
    """Calculate composite norms against exact finite-volume cell averages."""

    l1_integral = 0.0
    l2_integral = 0.0
    linf = 0.0

    def accumulate(patch: Patch1D) -> None:
        nonlocal l1_integral, l2_integral, linf
        exact = np.asarray(exact_cell_averages(patch.grid.cell_edges), dtype=float)
        patch.grid.validate_field(exact)
        covered = np.zeros(patch.n_valid_cells, dtype=bool)
        for child in patch.children:
            if child.parent_range is None:
                raise RuntimeError("Hierarchy invariant violated: child has no parent range")
            start, stop = child.parent_range
            covered[start:stop] = True
        if np.any(~covered):
            difference = np.abs(patch.values[~covered] - exact[~covered])
            l1_integral += float(np.sum(difference) * patch.grid.dx)
            l2_integral += float(np.sum(difference**2) * patch.grid.dx)
            linf = max(linf, float(np.max(difference)))
        for child in patch.children:
            accumulate(child)

    accumulate(hierarchy.root)
    length = hierarchy.root.grid.length
    return ErrorNorms(
        l1=l1_integral / length,
        l2=float(np.sqrt(l2_integral / length)),
        linf=linf,
    )


def composite_error_norms_2d(
    hierarchy: AMRHierarchy2D,
    exact: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> ErrorNorms:
    """Calculate physical-space error norms over rectangular 2D leaf cells."""

    l1_integral = 0.0
    l2_integral = 0.0
    linf = 0.0

    def accumulate(patch: Patch2D) -> None:
        nonlocal l1_integral, l2_integral, linf
        covered = np.zeros(patch.grid.shape, dtype=bool)
        for child in patch.children:
            if child.parent_range is None:
                raise RuntimeError("Hierarchy invariant violated: missing parent range")
            (y_start, y_stop), (x_start, x_stop) = child.parent_range
            covered[y_start:y_stop, x_start:x_stop] = True
        if np.any(~covered):
            x, y = patch.grid.cell_centres
            difference = np.abs(
                patch.values[~covered] - np.asarray(exact(x, y))[~covered]
            )
            area = patch.grid.cell_area
            l1_integral += float(np.sum(difference) * area)
            l2_integral += float(np.sum(difference**2) * area)
            linf = max(linf, float(np.max(difference)))
        for child in patch.children:
            accumulate(child)

    accumulate(hierarchy.root)
    root = hierarchy.root.grid
    domain_area = (root.x_max - root.x_min) * (root.y_max - root.y_min)
    return ErrorNorms(
        l1=l1_integral / domain_area,
        l2=float(np.sqrt(l2_integral / domain_area)),
        linf=linf,
    )


def composite_cell_average_error_norms_2d(
    hierarchy: AMRHierarchy2D,
    exact_cell_averages: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> ErrorNorms:
    """Calculate 2D composite norms against analytical cell averages."""

    l1_integral = 0.0
    l2_integral = 0.0
    linf = 0.0

    def accumulate(patch: Patch2D) -> None:
        nonlocal l1_integral, l2_integral, linf
        exact = np.asarray(
            exact_cell_averages(patch.grid.x_edges, patch.grid.y_edges),
            dtype=float,
        )
        patch.grid.validate_field(exact)
        covered = np.zeros(patch.grid.shape, dtype=bool)
        for child in patch.children:
            if child.parent_range is None:
                raise RuntimeError("Hierarchy invariant violated: missing parent range")
            (y_start, y_stop), (x_start, x_stop) = child.parent_range
            covered[y_start:y_stop, x_start:x_stop] = True
        if np.any(~covered):
            difference = np.abs(patch.values[~covered] - exact[~covered])
            area = patch.grid.cell_area
            l1_integral += float(np.sum(difference) * area)
            l2_integral += float(np.sum(difference**2) * area)
            linf = max(linf, float(np.max(difference)))
        for child in patch.children:
            accumulate(child)

    accumulate(hierarchy.root)
    root = hierarchy.root.grid
    domain_area = (root.x_max - root.x_min) * (root.y_max - root.y_min)
    return ErrorNorms(
        l1=l1_integral / domain_area,
        l2=float(np.sqrt(l2_integral / domain_area)),
        linf=linf,
    )
