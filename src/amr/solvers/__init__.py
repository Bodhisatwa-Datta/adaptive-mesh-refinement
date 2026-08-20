"""PDE solvers."""

from amr.solvers.advection1d import AdvectionResult, LinearAdvection1D
from amr.solvers.advection2d import AdvectionResult2D, LinearAdvection2D
from amr.solvers.amr_advection2d import (
    AMRAdvectionResult2D,
    AMRLinearAdvection2D,
    RegridEvent2D,
)
from amr.solvers.amr_advection1d import (
    AMRAdvectionResult,
    AMRLinearAdvection1D,
    RegridEvent,
)
from amr.solvers.burgers1d import BurgersResult, InviscidBurgers1D
from amr.solvers.amr_burgers1d import AMRBurgersResult, AMRInviscidBurgers1D
from amr.solvers.diffusion1d import DiffusionResult, ExplicitDiffusion1D
from amr.solvers.diffusion2d import DiffusionResult2D, ExplicitDiffusion2D
from amr.solvers.amr_diffusion2d import (
    AMRDiffusionResult2D,
    AMRExplicitDiffusion2D,
    DiffusionRegridEvent2D,
)
from amr.solvers.amr_diffusion1d import AMRDiffusionResult, AMRExplicitDiffusion1D
from amr.solvers.second_order_advection1d import SecondOrderLinearAdvection1D
from amr.solvers.second_order_burgers1d import SecondOrderInviscidBurgers1D

__all__ = [
    "AMRAdvectionResult",
    "AMRAdvectionResult2D",
    "AMRLinearAdvection1D",
    "AMRLinearAdvection2D",
    "AMRBurgersResult",
    "AMRInviscidBurgers1D",
    "AMRDiffusionResult",
    "AMRDiffusionResult2D",
    "AMRExplicitDiffusion1D",
    "AMRExplicitDiffusion2D",
    "AdvectionResult",
    "AdvectionResult2D",
    "BurgersResult",
    "DiffusionResult",
    "DiffusionResult2D",
    "DiffusionRegridEvent2D",
    "ExplicitDiffusion1D",
    "ExplicitDiffusion2D",
    "InviscidBurgers1D",
    "LinearAdvection1D",
    "LinearAdvection2D",
    "RegridEvent",
    "RegridEvent2D",
    "SecondOrderLinearAdvection1D",
    "SecondOrderInviscidBurgers1D",
]
