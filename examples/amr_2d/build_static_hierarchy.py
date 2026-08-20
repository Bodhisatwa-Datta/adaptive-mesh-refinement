"""Build and visualize a gradient-selected rectangular 2D hierarchy."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amr.benchmarks.advection2d import periodic_gaussian_2d
from amr.diagnostics.plotting import plot_hierarchy_2d
from amr.grid.grid2d import UniformGrid2D
from amr.grid.hierarchy2d import AMRHierarchy2D
from amr.refinement.criteria import (
    bounding_box_from_flags_2d,
    flag_gradient_2d,
)


def main() -> None:
    """Flag a Gaussian, build one rectangular child, and save a mesh figure."""

    grid = UniformGrid2D(0.0, 1.0, 64, 0.0, 1.0, 64)
    x, y = grid.cell_centres
    values = periodic_gaussian_2d(x, y, width=(0.09, 0.06))
    hierarchy = AMRHierarchy2D(grid, values, refinement_ratio=2)
    flags = flag_gradient_2d(
        values,
        grid.dx,
        grid.dy,
        threshold=2.0,
        n_buffer=2,
        periodic=True,
    )
    box = bounding_box_from_flags_2d(flags)
    if box is None:
        raise RuntimeError("Configured Gaussian unexpectedly produced no refinement flags")
    hierarchy.add_patch(hierarchy.root, *box)

    levels = np.zeros(grid.shape)
    x_start, x_stop, y_start, y_stop = box
    levels[y_start:y_stop, x_start:x_stop] = 1.0

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    field_image = axes[0].pcolormesh(
        grid.x_edges, grid.y_edges, values, shading="flat", cmap="viridis"
    )
    axes[0].scatter(x[flags], y[flags], s=4, color="white", alpha=0.5)
    plot_hierarchy_2d(hierarchy, axes[0])
    axes[0].set_title("Gradient flags and rectangular patch")
    figure.colorbar(field_image, ax=axes[0], label="u")

    level_image = axes[1].pcolormesh(
        grid.x_edges,
        grid.y_edges,
        levels,
        shading="flat",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    plot_hierarchy_2d(hierarchy, axes[1])
    axes[1].set_title("Composite refinement level")
    figure.colorbar(level_image, ax=axes[1], ticks=[0, 1], label="Level")
    figure.tight_layout()

    output = ROOT / "figures" / "gradient_selected_amr_hierarchy_2d.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Parent-cell box: {box}")
    print(f"Patches: {len(hierarchy.patches)}")
    print(f"Stored cells: {hierarchy.n_stored_cells}")
    print(f"Active cells: {hierarchy.n_active_cells}")
    print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
