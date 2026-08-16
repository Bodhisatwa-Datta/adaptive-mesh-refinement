"""Numerical foundations for an adaptive mesh refinement PDE framework."""

from amr.grid.grid1d import UniformGrid1D
from amr.solvers.advection1d import AdvectionResult, LinearAdvection1D

__all__ = ["AdvectionResult", "LinearAdvection1D", "UniformGrid1D"]

