"""Build a static level-one hierarchy from a configurable gradient criterion."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from amr.benchmarks.advection import gaussian
from amr.diagnostics.plotting import plot_hierarchy_1d
from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.refinement.criteria import flag_gradient, regions_from_flags


def main() -> None:
    """Flag a Gaussian, create fine patches, and save the hierarchy figure."""

    grid = UniformGrid1D(0.0, 1.0, 64)
    values = gaussian(grid.cell_centres, centre=0.5, width=0.08)
    hierarchy = AMRHierarchy1D(grid, values, refinement_ratio=2)

    flags = flag_gradient(
        values,
        grid.dx,
        threshold=3.0,
        n_buffer=2,
        periodic=True,
    )
    regions = regions_from_flags(flags, merge_gap=4)
    for start, stop in regions:
        hierarchy.add_patch(hierarchy.root, start, stop)

    figure, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(grid.cell_centres, values, color="black", label="Base-grid field")
    axes[0].plot(grid.cell_centres[flags], values[flags], "o", color="tab:red", label="Flagged + buffer")
    for child in hierarchy.root.children:
        axes[0].axvspan(*child.physical_bounds, color="tab:blue", alpha=0.15)
    axes[0].set(ylabel="u", title="Gradient-selected refinement region")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    plot_hierarchy_1d(hierarchy, axes[1])
    figure.tight_layout()

    output = ROOT / "figures" / "phase2_static_hierarchy.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    print(f"Flagged regions: {regions}")
    print(f"Patches: {len(hierarchy.patches)}")
    print(f"Stored cells: {hierarchy.n_stored_cells}")
    print(f"Active cells: {hierarchy.n_active_cells}")
    print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

