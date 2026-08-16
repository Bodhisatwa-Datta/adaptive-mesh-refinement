"""Mesh-hierarchy visualisation utilities."""

from __future__ import annotations

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from amr.grid.hierarchy import AMRHierarchy1D


def plot_hierarchy_1d(hierarchy: AMRHierarchy1D, ax: Axes | None = None) -> Axes:
    """Draw valid-cell edges for every patch, separated by refinement level."""

    if ax is None:
        _, ax = plt.subplots()
    for patch in hierarchy.patches:
        level = patch.level
        edges = patch.grid.cell_edges
        ax.hlines(level, patch.grid.x_min, patch.grid.x_max, color=f"C{level}", linewidth=2)
        ax.vlines(edges, level - 0.16, level + 0.16, color=f"C{level}", linewidth=0.7)
    max_level = max(patch.level for patch in hierarchy.patches)
    ax.set(
        xlabel="x",
        ylabel="Refinement level",
        yticks=range(max_level + 1),
        ylim=(-0.4, max_level + 0.4),
    )
    ax.grid(axis="x", alpha=0.2)
    return ax

