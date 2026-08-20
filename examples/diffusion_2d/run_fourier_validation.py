"""Validate explicit 2D diffusion using an analytical periodic Fourier mode."""

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

from amr.benchmarks.diffusion2d import (
    periodic_fourier_diffusion_2d_cell_averages,
)
from amr.diagnostics.conservation import total_mass_2d
from amr.diagnostics.errors import error_norms
from amr.grid.grid2d import UniformGrid2D
from amr.solvers.diffusion2d import ExplicitDiffusion2D

DIFFUSIVITY = 0.01
FINAL_TIME = 0.05
MODES = (1, 2)


def run_case(n_cells: int) -> dict[str, object]:
    """Run one square-grid resolution and return measured diagnostics."""

    grid = UniformGrid2D(0.0, 1.0, n_cells, 0.0, 1.0, n_cells)
    initial = periodic_fourier_diffusion_2d_cell_averages(
        grid.x_edges, grid.y_edges, 0.0, DIFFUSIVITY, modes=MODES
    )
    result = ExplicitDiffusion2D(grid, DIFFUSIVITY).solve(initial, FINAL_TIME)
    exact = periodic_fourier_diffusion_2d_cell_averages(
        grid.x_edges, grid.y_edges, FINAL_TIME, DIFFUSIVITY, modes=MODES
    )
    errors = error_norms(result.values, exact)
    return {
        "grid": grid,
        "values": result.values,
        "exact": exact,
        "steps": result.n_steps,
        "l1": errors.l1,
        "l2": errors.l2,
        "linf": errors.linf,
        "mass_error": total_mass_2d(result.values, grid)
        - total_mass_2d(initial, grid),
    }


def main() -> None:
    """Write measured convergence data and the validation figure."""

    resolutions = [40, 80, 160, 320]
    cases = [run_case(n_cells) for n_cells in resolutions]
    orders = [np.nan]
    for coarse, fine in zip(cases[:-1], cases[1:]):
        orders.append(
            np.log(float(coarse["l1"]) / float(fine["l1"])) / np.log(2.0)
        )

    output_directory = ROOT / "benchmarks" / "convergence"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "diffusion_2d_fourier.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["n", "steps", "l1", "l2", "linf", "l1_order", "mass_error"]
        )
        for n_cells, case, order in zip(resolutions, cases, orders):
            writer.writerow(
                [
                    n_cells,
                    case["steps"],
                    case["l1"],
                    case["l2"],
                    case["linf"],
                    order,
                    case["mass_error"],
                ]
            )

    case = cases[2]
    numerical = np.asarray(case["values"])
    exact = np.asarray(case["exact"])
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    limits = (float(exact.min()), float(exact.max()))
    image_options = {
        "origin": "lower",
        "extent": (0.0, 1.0, 0.0, 1.0),
        "aspect": "equal",
        "cmap": "coolwarm",
        "vmin": limits[0],
        "vmax": limits[1],
    }
    image = axes[0].imshow(numerical, **image_options)
    axes[0].set(title="Numerical field (N=160)", xlabel="x", ylabel="y")
    axes[1].imshow(exact, **image_options)
    axes[1].set(title="Analytical decay", xlabel="x", ylabel="y")
    figure.colorbar(image, ax=axes[:2], shrink=0.82, label="u")

    errors = np.asarray([float(case["l1"]) for case in cases])
    axes[2].loglog(resolutions, errors, "o-", label=r"Measured $L_1$")
    axes[2].loglog(
        resolutions,
        errors[0] * (resolutions[0] / np.asarray(resolutions)) ** 2,
        "k--",
        label=r"$O(\Delta x^2)$",
    )
    axes[2].set(xlabel="Cells per direction", ylabel=r"$L_1$ error", title="Convergence")
    axes[2].grid(which="both", alpha=0.25)
    axes[2].legend()
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.88, wspace=0.38)
    figure_path = figure_directory / "diffusion_2d_convergence.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")
    for n_cells, case, order in zip(resolutions, cases, orders):
        print(
            f"N={n_cells:4d} L1={float(case['l1']):.6e} "
            f"order={order:.3f} mass change={float(case['mass_error']):+.3e}"
        )


if __name__ == "__main__":
    main()
