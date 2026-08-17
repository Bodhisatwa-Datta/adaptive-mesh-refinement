"""Compare shock-tracking Burgers AMR with uniform-grid calculations."""

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

from amr.benchmarks.burgers import smooth_periodic_profile
from amr.diagnostics.errors import composite_error_norms, error_norms
from amr.diagnostics.plotting import plot_hierarchy_1d
from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.refinement.regrid import GradientRegridConfig, regrid_from_gradient
from amr.solvers.amr_burgers1d import AMRInviscidBurgers1D
from amr.solvers.burgers1d import InviscidBurgers1D


FINAL_TIME = 1.0


def run_uniform(n_cells: int) -> tuple[UniformGrid1D, np.ndarray, dict[str, float | int]]:
    grid = UniformGrid1D(0.0, 1.0, n_cells)
    initial = smooth_periodic_profile(grid.cell_centres)
    result = InviscidBurgers1D(grid).solve(initial, FINAL_TIME)
    return grid, result.values, {
        "active_cells": n_cells,
        "cell_updates": n_cells * result.n_steps,
        "mass_error": float(np.sum(result.values - initial) * grid.dx),
    }


def leaf_data(hierarchy: AMRHierarchy1D) -> tuple[np.ndarray, np.ndarray]:
    root = hierarchy.root
    covered = np.zeros(root.n_valid_cells, dtype=bool)
    for child in root.children:
        if child.parent_range is None:
            raise RuntimeError("Hierarchy invariant violated: child has no parent range")
        start, stop = child.parent_range
        covered[start:stop] = True
    coordinates = [root.grid.cell_centres[~covered]]
    values = [root.values[~covered]]
    for child in root.children:
        coordinates.append(child.grid.cell_centres)
        values.append(child.values)
    x = np.concatenate(coordinates)
    order = np.argsort(x)
    return x[order], np.concatenate(values)[order]


def main() -> None:
    reference_grid, reference_values, _ = run_uniform(2048)
    reference = lambda x: np.interp(
        x,
        reference_grid.cell_centres,
        reference_values,
        period=reference_grid.length,
    )

    coarse_grid, coarse_values, coarse = run_uniform(64)
    fine_grid, fine_values, fine = run_uniform(128)
    coarse_errors = error_norms(coarse_values, reference(coarse_grid.cell_centres))
    fine_errors = error_norms(fine_values, reference(fine_grid.cell_centres))
    coarse.update(l1=coarse_errors.l1, l2=coarse_errors.l2, linf=coarse_errors.linf)
    fine.update(l1=fine_errors.l1, l2=fine_errors.l2, linf=fine_errors.linf)

    hierarchy = AMRHierarchy1D(
        coarse_grid,
        smooth_periodic_profile(coarse_grid.cell_centres),
        refinement_ratio=2,
    )
    config = GradientRegridConfig(1.0, 0.7, n_buffer=2, merge_gap=0)
    regrid_from_gradient(hierarchy, config)
    amr_result = AMRInviscidBurgers1D(
        hierarchy,
        regrid_config=config,
        regrid_interval=2,
        subcycling=True,
        reflux=True,
    ).solve(FINAL_TIME)
    amr_errors = composite_error_norms(hierarchy, reference)
    amr = {
        "active_cells": hierarchy.n_active_cells,
        "cell_updates": amr_result.cell_updates,
        "mass_error": amr_result.mass_error,
        "l1": amr_errors.l1,
        "l2": amr_errors.l2,
        "linf": amr_errors.linf,
    }

    named = [("uniform_64", coarse), ("dynamic_amr", amr), ("uniform_128", fine)]
    output_directory = ROOT / "benchmarks" / "uniform_vs_amr"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    columns = ["method", "active_cells", "cell_updates", "l1", "l2", "linf", "mass_error"]
    csv_path = output_directory / "burgers_shock.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, values in named:
            writer.writerow({column: name if column == "method" else values[column] for column in columns})

    amr_x, amr_u = leaf_data(hierarchy)
    figure = plt.figure(figsize=(11, 7.2))
    grid_spec = figure.add_gridspec(2, 2, height_ratios=[1.5, 1.0])
    solution_ax = figure.add_subplot(grid_spec[0, 0])
    error_ax = figure.add_subplot(grid_spec[0, 1])
    hierarchy_ax = figure.add_subplot(grid_spec[1, :])
    solution_ax.plot(reference_grid.cell_centres, reference_values, "k--", label="Uniform N=2048 reference")
    solution_ax.plot(coarse_grid.cell_centres, coarse_values, color="tab:gray", label="Uniform N=64")
    solution_ax.plot(amr_x, amr_u, ".", color="tab:purple", label="Dynamic refluxed AMR")
    solution_ax.set(xlabel="x", ylabel="u", title="Burgers shock at t=1.0")
    solution_ax.legend(fontsize=8)
    solution_ax.grid(alpha=0.2)

    error_ax.bar(
        ["Uniform 64", "Dynamic AMR", "Uniform 128"],
        [coarse_errors.l1, amr_errors.l1, fine_errors.l1],
        color=["tab:gray", "tab:purple", "tab:green"],
    )
    error_ax.set(ylabel=r"$L_1$ error vs N=2048 reference", title="Shock accuracy")
    error_ax.tick_params(axis="x", rotation=15)
    error_ax.grid(axis="y", alpha=0.2)
    plot_hierarchy_1d(hierarchy, hierarchy_ax)
    hierarchy_ax.set_title("Final refinement concentrated at the periodic shock")
    figure.tight_layout()
    figure_path = figure_directory / "burgers_shock_amr.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")
    for name, values in named:
        print(
            f"{name:11s} active={int(values['active_cells']):3d} "
            f"updates={int(values['cell_updates']):5d} L1={float(values['l1']):.6e} "
            f"mass change={float(values['mass_error']):+.3e}"
        )


if __name__ == "__main__":
    main()
