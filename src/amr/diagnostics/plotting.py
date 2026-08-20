"""Mesh-hierarchy visualisation utilities."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from amr.grid.hierarchy import AMRHierarchy1D
from amr.grid.hierarchy2d import AMRHierarchy2D


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


def plot_hierarchy_2d(hierarchy: AMRHierarchy2D, ax: Axes | None = None) -> Axes:
    """Overlay every rectangular 2D patch boundary, colored by level."""

    if ax is None:
        _, ax = plt.subplots()
    for patch in hierarchy.patches:
        rectangle = Rectangle(
            (patch.grid.x_min, patch.grid.y_min),
            patch.grid.x_max - patch.grid.x_min,
            patch.grid.y_max - patch.grid.y_min,
            fill=False,
            edgecolor=f"C{patch.level}",
            linewidth=1.5 + 0.5 * patch.level,
            label=f"Level {patch.level}",
        )
        ax.add_patch(rectangle)
    root = hierarchy.root.grid
    ax.set(
        xlim=(root.x_min, root.x_max),
        ylim=(root.y_min, root.y_max),
        xlabel="x",
        ylabel="y",
        aspect="equal",
    )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right")
    return ax
