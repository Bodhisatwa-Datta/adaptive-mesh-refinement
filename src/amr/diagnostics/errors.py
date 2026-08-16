"""Discrete error norms."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


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

