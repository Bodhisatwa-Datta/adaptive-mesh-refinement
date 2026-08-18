"""Validate explicit diffusion with an exactly decaying periodic Fourier mode."""

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

from amr.benchmarks.diffusion import periodic_sine_diffusion_cell_averages
from amr.diagnostics.conservation import total_mass
from amr.diagnostics.errors import error_norms
from amr.grid.grid1d import UniformGrid1D
from amr.solvers.diffusion1d import ExplicitDiffusion1D


DIFFUSIVITY = 0.01
FINAL_TIME = 0.05
MODE = 2


def run_case(n_cells: int) -> dict[str, object]:
    grid = UniformGrid1D(0.0, 1.0, n_cells)
    initial = periodic_sine_diffusion_cell_averages(
        grid.cell_edges, 0.0, DIFFUSIVITY, mode=MODE
    )
    result = ExplicitDiffusion1D(grid, DIFFUSIVITY).solve(initial, FINAL_TIME)
    exact = periodic_sine_diffusion_cell_averages(
        grid.cell_edges, FINAL_TIME, DIFFUSIVITY, mode=MODE
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
        "mass_error": total_mass(result.values, grid) - total_mass(initial, grid),
    }


def main() -> None:
    resolutions = [40, 80, 160, 320]
    cases = [run_case(n_cells) for n_cells in resolutions]
    orders = [np.nan]
    for coarse, fine in zip(cases[:-1], cases[1:]):
        orders.append(np.log(float(coarse["l1"]) / float(fine["l1"])) / np.log(2.0))

    output_directory = ROOT / "benchmarks" / "convergence"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "diffusion_1d_fourier.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["n_cells", "steps", "l1", "l2", "linf", "l1_order", "mass_error"])
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
    grid = case["grid"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(grid.cell_centres, case["exact"], "k--", label="Analytical averages")
    axes[0].plot(grid.cell_centres, case["values"], color="tab:blue", label="Explicit FV")
    axes[0].set(xlabel="x", ylabel="u", title="Fourier-mode decay (mode 2)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    errors = np.asarray([float(case["l1"]) for case in cases])
    axes[1].loglog(resolutions, errors, "o-", label=r"Measured $L_1$")
    axes[1].loglog(
        resolutions,
        errors[0] * (resolutions[0] / np.asarray(resolutions)) ** 2,
        "k--",
        label=r"Second order $O(\Delta x^2)$",
    )
    axes[1].set(xlabel="Number of cells", ylabel=r"$L_1$ error", title="Grid convergence")
    axes[1].legend()
    axes[1].grid(which="both", alpha=0.25)
    figure.tight_layout()
    figure_path = figure_directory / "diffusion_fourier_convergence.png"
    figure.savefig(figure_path, dpi=180)
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
