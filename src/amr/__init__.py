"""Numerical foundations for an adaptive mesh refinement PDE framework."""

from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.grid.patch import Patch1D
from amr.solvers.advection1d import AdvectionResult, LinearAdvection1D

__all__ = [
    "AMRHierarchy1D",
    "AdvectionResult",
    "LinearAdvection1D",
    "Patch1D",
    "UniformGrid1D",
]
