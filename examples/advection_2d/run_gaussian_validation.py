"""Validate periodic diagonal advection of a two-dimensional Gaussian."""

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
from amr.diagnostics.errors import error_norms
from amr.grid.grid2d import UniformGrid2D
from amr.solvers.advection2d import LinearAdvection2D

VELOCITY = (0.7, -0.4)
FINAL_TIME = 0.25
PROFILE_PARAMETERS = {"centre": (0.25, 0.35), "width": (0.09, 0.07)}


def run_case(n_cells: int) -> dict[str, object]:
    """Run one square-grid resolution and return measured diagnostics."""

    grid = UniformGrid2D(0.0, 1.0, n_cells, 0.0, 1.0, n_cells)
    x, y = grid.cell_centres
    initial = periodic_gaussian_2d(x, y, **PROFILE_PARAMETERS)
    result = LinearAdvection2D(grid, *VELOCITY, cfl=0.8).solve(
        initial, FINAL_TIME
    )
    exact = translated_gaussian_2d(
        x, y, FINAL_TIME, VELOCITY, **PROFILE_PARAMETERS
    )
    errors = error_norms(result.values, exact)
    return {
        "grid": grid,
        "numerical": result.values,
        "exact": exact,
        "steps": result.n_steps,
        "l1": errors.l1,
        "l2": errors.l2,
        "linf": errors.linf,
        "mass_error": abs(
            total_mass_2d(result.values, grid) - total_mass_2d(initial, grid)
        ),
    }


def main() -> None:
    """Write a convergence table and a field/convergence figure."""

    resolutions = [24, 48, 96, 192]
    cases = [run_case(n) for n in resolutions]
    rates = [np.nan]
    for coarse, fine in zip(cases[:-1], cases[1:]):
        rates.append(
            np.log(float(coarse["l1"]) / float(fine["l1"])) / np.log(2.0)
        )

    output_directory = ROOT / "benchmarks" / "convergence"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    csv_path = output_directory / "advection_2d_gaussian.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["n", "steps", "l1", "l2", "linf", "l1_order", "mass_error"]
        )
        for n_cells, case, rate in zip(resolutions, cases, rates):
            writer.writerow(
                [
                    n_cells,
                    case["steps"],
                    case["l1"],
                    case["l2"],
                    case["linf"],
                    rate,
                    case["mass_error"],
                ]
            )

    case = cases[2]
    numerical = np.asarray(case["numerical"])
    exact = np.asarray(case["exact"])
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    image_options = {
        "origin": "lower",
        "extent": (0.0, 1.0, 0.0, 1.0),
        "aspect": "equal",
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 1.0,
    }
    numerical_image = axes[0].imshow(numerical, **image_options)
    axes[0].set(title="Numerical field (N=96)", xlabel="x", ylabel="y")
    axes[1].imshow(exact, **image_options)
    axes[1].set(title="Exact translated field", xlabel="x", ylabel="y")
    figure.colorbar(numerical_image, ax=axes[:2], shrink=0.82, label="u")

    l1_errors = np.asarray([float(case["l1"]) for case in cases])
    axes[2].loglog(resolutions, l1_errors, "o-", label=r"Measured $L_1$")
    reference = l1_errors[0] * resolutions[0] / np.asarray(resolutions)
    axes[2].loglog(resolutions, reference, "k--", label=r"$O(\Delta x)$")
    axes[2].set(xlabel="Cells per direction", ylabel=r"$L_1$ error", title="Convergence")
    axes[2].grid(which="both", alpha=0.25)
    axes[2].legend()
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.88, wspace=0.38)
    figure_path = figure_directory / "advection_2d_convergence.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")
    for n_cells, measured, rate in zip(resolutions, cases, rates):
        print(
            f"N={n_cells:4d}  L1={float(measured['l1']):.6e}  "
            f"order={rate:.3f}  mass error={float(measured['mass_error']):.3e}"
        )


if __name__ == "__main__":
    main()
