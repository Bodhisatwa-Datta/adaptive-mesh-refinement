"""PDE solvers."""

from amr.solvers.advection1d import AdvectionResult, LinearAdvection1D
from amr.solvers.amr_advection1d import AMRAdvectionResult, AMRLinearAdvection1D

__all__ = [
    "AMRAdvectionResult",
    "AMRLinearAdvection1D",
    "AdvectionResult",
    "LinearAdvection1D",
]
