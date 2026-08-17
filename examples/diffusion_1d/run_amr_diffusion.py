"""Compare dynamic AMR diffusion with coarse and fine uniform grids."""

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

from amr.benchmarks.diffusion import periodic_gaussian_diffusion
from amr.diagnostics.errors import composite_error_norms, error_norms
from amr.diagnostics.plotting import plot_hierarchy_1d
from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.refinement.regrid import GradientRegridConfig, regrid_from_gradient
from amr.solvers.amr_diffusion1d import AMRExplicitDiffusion1D
from amr.solvers.diffusion1d import ExplicitDiffusion1D


DIFFUSIVITY = 0.01
FINAL_TIME = 0.05


def exact(x: np.ndarray) -> np.ndarray:
    return periodic_gaussian_diffusion(x, FINAL_TIME, DIFFUSIVITY)


def uniform_case(n_cells: int) -> tuple[UniformGrid1D, np.ndarray, dict[str, float | int]]:
    grid = UniformGrid1D(0.0, 1.0, n_cells)
    initial = periodic_gaussian_diffusion(grid.cell_centres, 0.0, DIFFUSIVITY)
    result = ExplicitDiffusion1D(grid, DIFFUSIVITY).solve(initial, FINAL_TIME)
    errors = error_norms(result.values, exact(grid.cell_centres))
    return grid, result.values, {
        "active_cells": n_cells,
        "cell_updates": n_cells * result.n_steps,
        "l1": errors.l1,
        "l2": errors.l2,
        "linf": errors.linf,
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
    x_parts = [root.grid.cell_centres[~covered]]
    u_parts = [root.values[~covered]]
    for child in root.children:
        x_parts.append(child.grid.cell_centres)
        u_parts.append(child.values)
    x = np.concatenate(x_parts)
    order = np.argsort(x)
    return x[order], np.concatenate(u_parts)[order]


def main() -> None:
    coarse_grid, coarse_values, coarse = uniform_case(64)
    _, _, fine = uniform_case(128)
    initial = periodic_gaussian_diffusion(coarse_grid.cell_centres, 0.0, DIFFUSIVITY)
    hierarchy = AMRHierarchy1D(coarse_grid, initial, refinement_ratio=2)
    config = GradientRegridConfig(
        1.0,
        0.5,
        n_buffer=4,
        merge_gap=4,
        prolongation="conservative_linear",
    )
    regrid_from_gradient(hierarchy, config)
    result = AMRExplicitDiffusion1D(
        hierarchy,
        DIFFUSIVITY,
        regrid_config=config,
        regrid_interval=2,
        subcycling=True,
        reflux=True,
    ).solve(FINAL_TIME)
    errors = composite_error_norms(hierarchy, exact)
    amr = {
        "active_cells": hierarchy.n_active_cells,
        "cell_updates": result.cell_updates,
        "l1": errors.l1,
        "l2": errors.l2,
        "linf": errors.linf,
        "mass_error": result.mass_error,
    }
    named = [("uniform_64", coarse), ("dynamic_amr", amr), ("uniform_128", fine)]

    output_directory = ROOT / "benchmarks" / "uniform_vs_amr"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    columns = ["method", "active_cells", "cell_updates", "l1", "l2", "linf", "mass_error"]
    csv_path = output_directory / "diffusion_1d.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, values in named:
            writer.writerow({column: name if column == "method" else values[column] for column in columns})

    amr_x, amr_u = leaf_data(hierarchy)
    plot_x = np.linspace(0.0, 1.0, 1000, endpoint=False)
    figure = plt.figure(figsize=(11, 7.2))
    grid_spec = figure.add_gridspec(2, 2, height_ratios=[1.5, 1.0])
    solution_ax = figure.add_subplot(grid_spec[0, 0])
    error_ax = figure.add_subplot(grid_spec[0, 1])
    hierarchy_ax = figure.add_subplot(grid_spec[1, :])
    solution_ax.plot(plot_x, exact(plot_x), "k--", label="Analytical")
    solution_ax.plot(coarse_grid.cell_centres, coarse_values, color="tab:gray", label="Uniform N=64")
    solution_ax.plot(amr_x, amr_u, ".", color="tab:cyan", label="Dynamic AMR")
    solution_ax.set(xlabel="x", ylabel="u", title="Gaussian diffusion at t=0.05")
    solution_ax.legend()
    solution_ax.grid(alpha=0.2)
    error_ax.bar(
        ["Uniform 64", "Dynamic AMR", "Uniform 128"],
        [float(coarse["l1"]), errors.l1, float(fine["l1"])],
        color=["tab:gray", "tab:cyan", "tab:green"],
    )
    error_ax.set(ylabel=r"$L_1$ error", title="Measured diffusion accuracy")
    error_ax.tick_params(axis="x", rotation=15)
    error_ax.grid(axis="y", alpha=0.2)
    plot_hierarchy_1d(hierarchy, hierarchy_ax)
    hierarchy_ax.set_title("Refinement expands with the spreading Gaussian")
    figure.tight_layout()
    figure_path = figure_directory / "diffusion_amr_comparison.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")
    for name, values in named:
        print(
            f"{name:11s} active={int(values['active_cells']):3d} "
            f"updates={int(values['cell_updates']):4d} L1={float(values['l1']):.6e} "
            f"mass change={float(values['mass_error']):+.3e}"
        )


if __name__ == "__main__":
    main()
