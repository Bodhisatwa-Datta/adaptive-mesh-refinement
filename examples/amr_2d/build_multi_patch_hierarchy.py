"""Compare multi-box clustering with one enclosing 2D refinement box."""

from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from amr.benchmarks.advection2d import periodic_gaussian_2d
from amr.diagnostics.plotting import plot_hierarchy_2d
from amr.grid.grid2d import UniformGrid2D
from amr.grid.hierarchy2d import AMRHierarchy2D
from amr.refinement.criteria import (
    bounding_box_from_flags_2d,
    boxes_from_flags_2d,
    flag_gradient_2d,
)


def main() -> None:
    """Build both layouts, write cell counts, and save their comparison."""

    grid = UniformGrid2D(0.0, 1.0, 64, 0.0, 1.0, 64)
    x, y = grid.cell_centres
    values = periodic_gaussian_2d(
        x, y, centre=(0.28, 0.35), width=(0.06, 0.06)
    ) + periodic_gaussian_2d(
        x, y, centre=(0.72, 0.65), width=(0.06, 0.06)
    )
    flags = flag_gradient_2d(
        values,
        grid.dx,
        grid.dy,
        threshold=2.0,
        n_buffer=2,
        periodic=False,
    )
    boxes = boxes_from_flags_2d(flags, merge_gap=1)
    enclosing = bounding_box_from_flags_2d(flags)
    if enclosing is None or len(boxes) < 2:
        raise RuntimeError("Configured features unexpectedly failed to form two boxes")

    multi = AMRHierarchy2D(grid, values, refinement_ratio=2)
    for box in boxes:
        multi.add_patch(multi.root, *box)
    single = AMRHierarchy2D(grid, values, refinement_ratio=2)
    single.add_patch(single.root, *enclosing)

    output_directory = ROOT / "benchmarks" / "uniform_vs_amr"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "multi_box_clustering_2d.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layout", "patches", "active_cells", "stored_cells"])
        writer.writerow(
            ["multi_box", len(boxes), multi.n_active_cells, multi.n_stored_cells]
        )
        writer.writerow(["single_box", 1, single.n_active_cells, single.n_stored_cells])

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for ax, hierarchy, title in (
        (axes[0], multi, "Connected-component boxes"),
        (axes[1], single, "One enclosing box"),
    ):
        image = ax.pcolormesh(
            grid.x_edges,
            grid.y_edges,
            values,
            shading="flat",
            cmap="viridis",
        )
        ax.scatter(x[flags], y[flags], s=3, color="white", alpha=0.45)
        plot_hierarchy_2d(hierarchy, ax)
        ax.set_title(title)
    figure.colorbar(image, ax=axes[:2], shrink=0.82, label="u")

    labels = ["Multi-box", "Single box"]
    active = [multi.n_active_cells, single.n_active_cells]
    stored = [multi.n_stored_cells, single.n_stored_cells]
    positions = [0, 1]
    axes[2].bar(
        [position - 0.18 for position in positions],
        active,
        width=0.36,
        label="Active",
    )
    axes[2].bar(
        [position + 0.18 for position in positions],
        stored,
        width=0.36,
        label="Stored",
    )
    axes[2].set(
        xticks=positions,
        xticklabels=labels,
        ylabel="Cells",
        title="Hierarchy size",
    )
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.88, wspace=0.35)
    figure_path = figure_directory / "multi_patch_hierarchy_2d.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Boxes: {boxes}")
    print(
        f"Multi-box: {multi.n_active_cells} active, {multi.n_stored_cells} stored"
    )
    print(
        f"Single box: {single.n_active_cells} active, {single.n_stored_cells} stored"
    )
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
