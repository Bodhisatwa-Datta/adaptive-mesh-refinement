"""One-dimensional uniform, cell-centred grid."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class UniformGrid1D:
    """Uniform finite-volume grid on the half-open domain ``[x_min, x_max)``."""

    x_min: float
    x_max: float
    n_cells: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.x_min) or not np.isfinite(self.x_max):
            raise ValueError("Domain bounds must be finite")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if isinstance(self.n_cells, bool) or not isinstance(self.n_cells, (int, np.integer)):
            raise TypeError("n_cells must be an integer")
        if self.n_cells < 2:
            raise ValueError("n_cells must be at least 2")

    @property
    def length(self) -> float:
        """Physical domain length."""

        return self.x_max - self.x_min

    @property
    def dx(self) -> float:
        """Cell width."""

        return self.length / self.n_cells

    @property
    def cell_centres(self) -> NDArray[np.float64]:
        """Coordinates of all cell centres."""

        return self.x_min + (np.arange(self.n_cells, dtype=float) + 0.5) * self.dx

    @property
    def cell_edges(self) -> NDArray[np.float64]:
        """Coordinates of all cell edges."""

        return np.linspace(self.x_min, self.x_max, self.n_cells + 1)

    def validate_field(self, values: NDArray[np.floating]) -> None:
        """Raise if ``values`` is not a finite scalar field on this grid."""

        if values.shape != (self.n_cells,):
            raise ValueError(f"Expected field shape {(self.n_cells,)}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("Field values must be finite")

