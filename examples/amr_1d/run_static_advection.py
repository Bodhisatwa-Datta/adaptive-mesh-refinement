"""Compare synchronized static AMR advection with two uniform grids."""

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

from amr.benchmarks.advection import gaussian, translated_profile
from amr.diagnostics.conservation import composite_mass, total_mass
from amr.diagnostics.errors import composite_error_norms, error_norms
from amr.diagnostics.plotting import plot_hierarchy_1d
from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.refinement.criteria import flag_gradient, regions_from_flags
from amr.solvers.advection1d import LinearAdvection1D
from amr.solvers.amr_advection1d import AMRLinearAdvection1D


FINAL_TIME = 0.1
VELOCITY = 1.0
CFL = 0.8


def profile(x: np.ndarray) -> np.ndarray:
    """Localized initial condition used by all three calculations."""

    return gaussian(x, centre=0.3, width=0.07)


def run_uniform(n_cells: int) -> dict[str, float | int | np.ndarray | UniformGrid1D]:
    """Run one uniform-grid reference calculation."""

    grid = UniformGrid1D(0.0, 1.0, n_cells)
    initial = profile(grid.cell_centres)
    result = LinearAdvection1D(grid, VELOCITY, CFL).solve(initial, FINAL_TIME)
    exact = translated_profile(grid.cell_centres, FINAL_TIME, VELOCITY, profile)
    errors = error_norms(result.values, exact)
    return {
        "grid": grid,
        "values": result.values,
        "steps": result.n_steps,
        "active_cells": n_cells,
        "stored_cells": n_cells,
        "cell_updates": n_cells * result.n_steps,
        "l1": errors.l1,
        "l2": errors.l2,
        "linf": errors.linf,
        "mass_error": total_mass(result.values, grid) - total_mass(initial, grid),
    }


def run_amr() -> tuple[AMRHierarchy1D, dict[str, float | int]]:
    """Build a solution-selected static patch and advance it synchronously."""

    grid = UniformGrid1D(0.0, 1.0, 64)
    initial = profile(grid.cell_centres)
    flags = flag_gradient(initial, grid.dx, threshold=3.0, n_buffer=8)
    regions = regions_from_flags(flags, merge_gap=4)
    hierarchy = AMRHierarchy1D(grid, initial, refinement_ratio=2)
    for start, stop in regions:
        hierarchy.add_patch(hierarchy.root, start, stop)

    initial_mass = composite_mass(hierarchy)
    result = AMRLinearAdvection1D(hierarchy, VELOCITY, CFL).solve(FINAL_TIME)
    exact = lambda x: translated_profile(x, FINAL_TIME, VELOCITY, profile)
    errors = composite_error_norms(hierarchy, exact)
    return hierarchy, {
        "steps": result.n_steps,
        "active_cells": hierarchy.n_active_cells,
        "stored_cells": hierarchy.n_stored_cells,
        # The synchronized baseline advances covered coarse cells before
        # restriction, so actual work scales with stored rather than leaf cells.
        "cell_updates": hierarchy.n_stored_cells * result.n_steps,
        "l1": errors.l1,
        "l2": errors.l2,
        "linf": errors.linf,
        "mass_error": composite_mass(hierarchy) - initial_mass,
    }


def leaf_data(hierarchy: AMRHierarchy1D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted leaf-cell centres, values, and levels for plotting."""

    root = hierarchy.root
    covered = np.zeros(root.n_valid_cells, dtype=bool)
    for child in root.children:
        if child.parent_range is None:
            raise RuntimeError("Hierarchy invariant violated: child has no parent range")
        start, stop = child.parent_range
        covered[start:stop] = True
    coordinates = [root.grid.cell_centres[~covered]]
    values = [root.values[~covered]]
    levels = [np.zeros(np.count_nonzero(~covered), dtype=int)]
    for child in root.children:
        coordinates.append(child.grid.cell_centres)
        values.append(child.values)
        levels.append(np.full(child.n_valid_cells, child.level, dtype=int))
    x = np.concatenate(coordinates)
    order = np.argsort(x)
    return np.concatenate(coordinates)[order], np.concatenate(values)[order], np.concatenate(levels)[order]


def main() -> None:
    """Run the comparison and write measured results and a figure."""

    coarse = run_uniform(64)
    fine = run_uniform(128)
    hierarchy, amr = run_amr()
    named = [("uniform_64", coarse), ("uniform_128", fine), ("static_amr", amr)]

    benchmark_directory = ROOT / "benchmarks" / "uniform_vs_amr"
    figure_directory = ROOT / "figures"
    benchmark_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    csv_path = benchmark_directory / "static_advection_1d.csv"
    columns = [
        "method",
        "active_cells",
        "stored_cells",
        "steps",
        "cell_updates",
        "l1",
        "l2",
        "linf",
        "mass_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, values in named:
            writer.writerow({column: name if column == "method" else values[column] for column in columns})

    exact_x = np.linspace(0.0, 1.0, 1000, endpoint=False)
    exact_u = translated_profile(exact_x, FINAL_TIME, VELOCITY, profile)
    amr_x, amr_u, amr_level = leaf_data(hierarchy)
    figure = plt.figure(figsize=(11, 7.5))
    grid_spec = figure.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
    solution_ax = figure.add_subplot(grid_spec[0, 0])
    error_ax = figure.add_subplot(grid_spec[0, 1])
    hierarchy_ax = figure.add_subplot(grid_spec[1, :])

    solution_ax.plot(exact_x, exact_u, "k--", label="Exact")
    solution_ax.plot(coarse["grid"].cell_centres, coarse["values"], color="tab:gray", label="Uniform N=64")
    for level, colour in ((0, "tab:blue"), (1, "tab:orange")):
        selected = amr_level == level
        solution_ax.plot(amr_x[selected], amr_u[selected], ".", color=colour, label=f"AMR level {level}")
    solution_ax.set(xlabel="x", ylabel="u", title=f"Static AMR advection at t={FINAL_TIME}")
    solution_ax.legend(fontsize=8)
    solution_ax.grid(alpha=0.2)

    labels = ["Uniform 64", "Static AMR", "Uniform 128"]
    l1_values = [float(coarse["l1"]), float(amr["l1"]), float(fine["l1"])]
    error_ax.bar(labels, l1_values, color=["tab:gray", "tab:orange", "tab:green"])
    error_ax.set(ylabel=r"$L_1$ error", title="Measured composite accuracy")
    error_ax.tick_params(axis="x", rotation=15)
    error_ax.grid(axis="y", alpha=0.2)

    plot_hierarchy_1d(hierarchy, hierarchy_ax)
    hierarchy_ax.set_title("Static hierarchy used for the update")
    figure.tight_layout()
    figure_path = figure_directory / "static_amr_advection_comparison.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")
    for name, values in named:
        print(
            f"{name:12s} active={int(values['active_cells']):3d} "
            f"updates={int(values['cell_updates']):4d} L1={float(values['l1']):.6e} "
            f"mass change={float(values['mass_error']):+.3e}"
        )


if __name__ == "__main__":
    main()
