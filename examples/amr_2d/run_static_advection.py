"""Compare static refluxed 2D AMR advection with uniform grids."""

from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from amr.refinement.criteria import bounding_box_from_flags_2d, flag_gradient_2d
from amr.solvers.advection2d import LinearAdvection2D
from amr.solvers.amr_advection2d import AMRLinearAdvection2D

VELOCITY = (0.5, 0.3)
FINAL_TIME = 0.1
PROFILE = {"centre": (0.3, 0.4), "width": (0.08, 0.07)}


def exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the exact field at the benchmark final time."""

    return translated_gaussian_2d(x, y, FINAL_TIME, VELOCITY, **PROFILE)


def main() -> None:
    """Run all comparisons and write measured data plus a validation figure."""

    base = UniformGrid2D(0.0, 1.0, 32, 0.0, 1.0, 32)
    base_x, base_y = base.cell_centres
    initial = periodic_gaussian_2d(base_x, base_y, **PROFILE)
    flags = flag_gradient_2d(
        initial, base.dx, base.dy, threshold=2.0, n_buffer=4, periodic=True
    )
    box = bounding_box_from_flags_2d(flags)
    if box is None:
        raise RuntimeError("Configured Gaussian unexpectedly produced no refinement flags")

    coarse_result = LinearAdvection2D(base, *VELOCITY).solve(initial, FINAL_TIME)
    coarse_errors = error_norms(coarse_result.values, exact(base_x, base_y))
    coarse_mass_error = total_mass_2d(coarse_result.values, base) - total_mass_2d(
        initial, base
    )

    hierarchy = AMRHierarchy2D(base, initial, refinement_ratio=2)
    hierarchy.add_patch(hierarchy.root, *box)
    amr_result = AMRLinearAdvection2D(
        hierarchy, *VELOCITY, reflux=True
    ).solve(FINAL_TIME)
    amr_errors = composite_error_norms_2d(hierarchy, exact)

    fine = UniformGrid2D(0.0, 1.0, 64, 0.0, 1.0, 64)
    fine_x, fine_y = fine.cell_centres
    fine_initial = periodic_gaussian_2d(fine_x, fine_y, **PROFILE)
    fine_result = LinearAdvection2D(fine, *VELOCITY).solve(
        fine_initial, FINAL_TIME
    )
    fine_errors = error_norms(fine_result.values, exact(fine_x, fine_y))
    fine_mass_error = total_mass_2d(fine_result.values, fine) - total_mass_2d(
        fine_initial, fine
    )

    rows = [
        (
            "uniform_32",
            base.nx * base.ny,
            base.nx * base.ny,
            coarse_result.n_steps * base.nx * base.ny,
            coarse_errors,
            coarse_mass_error,
        ),
        (
            "static_amr",
            hierarchy.n_active_cells,
            hierarchy.n_stored_cells,
            amr_result.cell_updates,
            amr_errors,
            amr_result.mass_error,
        ),
        (
            "uniform_64",
            fine.nx * fine.ny,
            fine.nx * fine.ny,
            fine_result.n_steps * fine.nx * fine.ny,
            fine_errors,
            fine_mass_error,
        ),
    ]

    output_directory = ROOT / "benchmarks" / "uniform_vs_amr"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "advection_2d_static_amr.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case",
                "active_cells",
                "stored_cells",
                "cell_updates",
                "l1",
                "l2",
                "linf",
                "mass_error",
            ]
        )
        for name, active, stored, updates, errors, mass_error in rows:
            writer.writerow(
                [
                    name,
                    active,
                    stored,
                    updates,
                    errors.l1,
                    errors.l2,
                    errors.linf,
                    mass_error,
                ]
            )

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    root = hierarchy.root
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
    plot_hierarchy_2d(hierarchy, axes[0])
    axes[0].set_title("Static refluxed AMR")
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
    axes[1].set(title="Exact translated field", xlabel="x", ylabel="y", aspect="equal")

    names = ["Uniform 32", "Static AMR", "Uniform 64"]
    errors = [row[4].l1 for row in rows]
    axes[2].bar(names, errors, color=["tab:gray", "tab:orange", "tab:blue"])
    axes[2].set(ylabel=r"$L_1$ error", title="Accuracy comparison")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure_path = figure_directory / "static_amr_advection_2d.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Parent-cell box: {box}")
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")
    for name, active, stored, updates, errors, mass_error in rows:
        print(
            f"{name:12s} active={active:4d} stored={stored:4d} "
            f"updates={updates:6d} L1={errors.l1:.6e} mass={mass_error:+.3e}"
        )
if __name__ == "__main__":
    main()
