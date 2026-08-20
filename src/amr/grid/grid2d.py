"""Two-dimensional uniform, cell-centred Cartesian grid."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class UniformGrid2D:
    """Uniform grid on ``[x_min, x_max) x [y_min, y_max)``.

    Scalar fields use NumPy shape ``(ny, nx)``: axis zero is the y direction
    and axis one is the x direction.
    """

    x_min: float
    x_max: float
    nx: int
    y_min: float
    y_max: float
    ny: int

    def __post_init__(self) -> None:
        bounds = (self.x_min, self.x_max, self.y_min, self.y_max)
        if not np.all(np.isfinite(bounds)):
            raise ValueError("Domain bounds must be finite")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Each upper domain bound must exceed its lower bound")
        for name, count in (("nx", self.nx), ("ny", self.ny)):
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if count < 2:
                raise ValueError(f"{name} must be at least 2")

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape of a scalar field on this grid."""

        return (self.ny, self.nx)

    @property
    def dx(self) -> float:
        """Cell width in the x direction."""

        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        """Cell width in the y direction."""

        return (self.y_max - self.y_min) / self.ny

    @property
    def cell_area(self) -> float:
        """Area of one cell."""

        return self.dx * self.dy

    @property
    def x_centres(self) -> NDArray[np.float64]:
        """One-dimensional x coordinates of cell centres."""

        return self.x_min + (np.arange(self.nx, dtype=float) + 0.5) * self.dx

    @property
    def x_edges(self) -> NDArray[np.float64]:
        """One-dimensional x coordinates of cell edges."""

        return np.linspace(self.x_min, self.x_max, self.nx + 1)

    @property
    def y_centres(self) -> NDArray[np.float64]:
        """One-dimensional y coordinates of cell centres."""

        return self.y_min + (np.arange(self.ny, dtype=float) + 0.5) * self.dy

    @property
    def y_edges(self) -> NDArray[np.float64]:
        """One-dimensional y coordinates of cell edges."""

        return np.linspace(self.y_min, self.y_max, self.ny + 1)

    @property
    def cell_centres(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return x and y centre-coordinate arrays with shape ``(ny, nx)``."""

        return np.meshgrid(self.x_centres, self.y_centres, indexing="xy")

    def validate_field(self, values: NDArray[np.floating]) -> None:
        """Raise if ``values`` is not a finite scalar field on this grid."""

        if values.shape != self.shape:
            raise ValueError(f"Expected field shape {self.shape}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("Field values must be finite")
