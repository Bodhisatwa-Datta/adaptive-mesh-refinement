"""Numerical foundations for an adaptive mesh refinement PDE framework."""

from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.grid.patch import Patch1D
from amr.solvers.advection1d import AdvectionResult, LinearAdvection1D
from amr.solvers.amr_advection1d import (
    AMRAdvectionResult,
    AMRLinearAdvection1D,
    RegridEvent,
)
from amr.solvers.burgers1d import BurgersResult, InviscidBurgers1D
from amr.solvers.amr_burgers1d import AMRBurgersResult, AMRInviscidBurgers1D
from amr.solvers.diffusion1d import DiffusionResult, ExplicitDiffusion1D
from amr.solvers.amr_diffusion1d import AMRDiffusionResult, AMRExplicitDiffusion1D

__all__ = [
    "AMRHierarchy1D",
    "AMRAdvectionResult",
    "AMRLinearAdvection1D",
    "AMRBurgersResult",
    "AMRInviscidBurgers1D",
    "AMRDiffusionResult",
    "AMRExplicitDiffusion1D",
    "AdvectionResult",
    "BurgersResult",
    "DiffusionResult",
    "ExplicitDiffusion1D",
    "InviscidBurgers1D",
    "LinearAdvection1D",
    "Patch1D",
    "RegridEvent",
    "UniformGrid1D",
]
