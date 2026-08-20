"""Compare moving-patch 2D AMR advection with static and uniform grids."""

from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from amr.benchmarks.advection2d import (
    periodic_gaussian_2d,
    translated_gaussian_2d,
)
from amr.diagnostics.conservation import total_mass_2d
from amr.diagnostics.errors import composite_error_norms_2d, error_norms
from amr.diagnostics.plotting import plot_hierarchy_2d
from amr.grid.grid2d import UniformGrid2D
from amr.grid.hierarchy2d import AMRHierarchy2D
from amr.refinement.regrid2d import (
    GradientRegridConfig2D,
    level_one_boxes_2d,
    regrid_from_gradient_2d,
)
from amr.solvers.advection2d import LinearAdvection2D
from amr.solvers.amr_advection2d import AMRLinearAdvection2D

VELOCITY = (0.6, 0.3)
FINAL_TIME = 0.5
PROFILE = {"centre": (0.3, 0.3), "width": (0.06, 0.06)}
REGRID_CONFIG = GradientRegridConfig2D(2.0, 1.0, n_buffer=3)


def exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return translated_gaussian_2d(x, y, FINAL_TIME, VELOCITY, **PROFILE)


def initialized_hierarchy(grid: UniformGrid2D) -> AMRHierarchy2D:
    """Create the initial gradient-selected hierarchy."""

    x, y = grid.cell_centres
    hierarchy = AMRHierarchy2D(
        grid, periodic_gaussian_2d(x, y, **PROFILE), refinement_ratio=2
    )
    regrid_from_gradient_2d(hierarchy, REGRID_CONFIG)
    return hierarchy


def main() -> None:
    """Run coarse, static, dynamic, and fine calculations and save results."""

    base = UniformGrid2D(0.0, 1.0, 32, 0.0, 1.0, 32)
    base_x, base_y = base.cell_centres
    initial = periodic_gaussian_2d(base_x, base_y, **PROFILE)
    coarse_result = LinearAdvection2D(base, *VELOCITY).solve(initial, FINAL_TIME)
    coarse_errors = error_norms(coarse_result.values, exact(base_x, base_y))
    coarse_mass = total_mass_2d(coarse_result.values, base) - total_mass_2d(
        initial, base
    )

    static = initialized_hierarchy(base)
    static_result = AMRLinearAdvection2D(
        static, *VELOCITY, reflux=True, subcycling=True
    ).solve(FINAL_TIME)
    static_errors = composite_error_norms_2d(static, exact)

    dynamic = initialized_hierarchy(base)
    initial_box = level_one_boxes_2d(dynamic)[0]
    dynamic_result = AMRLinearAdvection2D(
        dynamic,
        *VELOCITY,
        reflux=True,
        subcycling=True,
        regrid_config=REGRID_CONFIG,
        regrid_interval=2,
    ).solve(FINAL_TIME)
    dynamic_errors = composite_error_norms_2d(dynamic, exact)

    fine = UniformGrid2D(0.0, 1.0, 64, 0.0, 1.0, 64)
    fine_x, fine_y = fine.cell_centres
    fine_initial = periodic_gaussian_2d(fine_x, fine_y, **PROFILE)
    fine_result = LinearAdvection2D(fine, *VELOCITY).solve(
        fine_initial, FINAL_TIME
    )
    fine_errors = error_norms(fine_result.values, exact(fine_x, fine_y))
    fine_mass = total_mass_2d(fine_result.values, fine) - total_mass_2d(
        fine_initial, fine
    )

    rows = [
        (
            "uniform_32",
            base.nx * base.ny,
            base.nx * base.ny,
            coarse_result.n_steps * base.nx * base.ny,
            coarse_errors,
            coarse_mass,
            0,
        ),
        (
            "static_subcycled_amr",
            static.n_active_cells,
            static_result.peak_active_cells,
            static_result.cell_updates,
            static_errors,
            static_result.mass_error,
            0,
        ),
        (
            "dynamic_subcycled_amr",
            dynamic.n_active_cells,
            dynamic_result.peak_active_cells,
            dynamic_result.cell_updates,
            dynamic_errors,
            dynamic_result.mass_error,
            len(dynamic_result.regrid_events),
        ),
        (
            "uniform_64",
            fine.nx * fine.ny,
            fine.nx * fine.ny,
            fine_result.n_steps * fine.nx * fine.ny,
            fine_errors,
            fine_mass,
            0,
        ),
    ]

    output_directory = ROOT / "benchmarks" / "uniform_vs_amr"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "advection_2d_dynamic_amr.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case",
                "final_active_cells",
                "peak_active_cells",
                "cell_updates",
                "l1",
                "l2",
                "linf",
                "mass_error",
                "regrid_events",
            ]
        )
        for name, active, peak, updates, errors, mass_error, events in rows:
            writer.writerow(
                [
                    name,
                    active,
                    peak,
                    updates,
                    errors.l1,
                    errors.l2,
                    errors.linf,
                    mass_error,
                    events,
                ]
            )

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    root = dynamic.root
    field_image = axes[0].pcolormesh(
        root.grid.x_edges,
        root.grid.y_edges,
        root.values,
        shading="flat",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    for child in root.children:
        axes[0].pcolormesh(
            child.grid.x_edges,
            child.grid.y_edges,
            child.values,
            shading="flat",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
    plot_hierarchy_2d(dynamic, axes[0])
    axes[0].set_title("Dynamic subcycled AMR")
    figure.colorbar(field_image, ax=axes[0], label="u")

    axes[1].pcolormesh(
        fine.x_edges,
        fine.y_edges,
        exact(fine_x, fine_y),
        shading="flat",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    boxes = [initial_box]
    boxes.extend(
        event.new_boxes[0]
        for event in dynamic_result.regrid_events
        if event.new_boxes
    )
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(boxes)))
    for box, color in zip(boxes, colors):
        x_start, x_stop, y_start, y_stop = box
        axes[1].add_patch(
            Rectangle(
                (base.x_edges[x_start], base.y_edges[y_start]),
                base.x_edges[x_stop] - base.x_edges[x_start],
                base.y_edges[y_stop] - base.y_edges[y_start],
                fill=False,
                edgecolor=color,
                linewidth=1.2,
                alpha=0.75,
            )
        )
    axes[1].set(
        title="Patch trajectory",
        xlabel="x",
        ylabel="y",
        aspect="equal",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )

    names = ["Uniform 32", "Static AMR", "Dynamic AMR", "Uniform 64"]
    errors = [row[4].l1 for row in rows]
    axes[2].bar(
        names,
        errors,
        color=["tab:gray", "tab:green", "tab:orange", "tab:blue"],
    )
    axes[2].set(ylabel=r"$L_1$ error", title="Accuracy comparison")
    axes[2].tick_params(axis="x", rotation=24)
    axes[2].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure_path = figure_directory / "dynamic_amr_advection_2d.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Initial box: {initial_box}")
    print(f"Final box: {level_one_boxes_2d(dynamic)[0]}")
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")
    for name, active, peak, updates, errors, mass_error, events in rows:
        print(
            f"{name:22s} active={active:4d} peak={peak:4d} "
            f"updates={updates:6d} L1={errors.l1:.6e} "
            f"mass={mass_error:+.3e} regrids={events}"
        )


if __name__ == "__main__":
    main()
