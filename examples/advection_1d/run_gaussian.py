"""Run and plot the Phase 1 Gaussian-advection validation benchmark."""

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
from amr.diagnostics.conservation import total_mass
from amr.diagnostics.errors import error_norms
from amr.grid.grid1d import UniformGrid1D
from amr.solvers.advection1d import LinearAdvection1D


def run_case(n_cells: int, final_time: float = 0.5) -> dict[str, object]:
    """Run one resolution and return fields plus measured diagnostics."""

    grid = UniformGrid1D(0.0, 1.0, n_cells)
    profile = lambda x: gaussian(x, centre=0.25, width=0.07)
    initial = profile(grid.cell_centres)
    solver = LinearAdvection1D(grid, velocity=1.0, cfl=0.8)
    result = solver.solve(initial, final_time)
    exact = translated_profile(grid.cell_centres, final_time, solver.velocity, profile)
    errors = error_norms(result.values, exact)
    return {
        "grid": grid,
        "initial": initial,
        "numerical": result.values,
        "exact": exact,
        "steps": result.n_steps,
        "l1": errors.l1,
        "l2": errors.l2,
        "linf": errors.linf,
        "mass_error": abs(total_mass(result.values, grid) - total_mass(initial, grid)),
    }


def main() -> None:
    """Execute convergence cases, write measured data, and save a figure."""

    resolutions = [50, 100, 200, 400]
    cases = [run_case(n) for n in resolutions]
    rates = [np.nan]
    for coarse, fine in zip(cases[:-1], cases[1:]):
        rates.append(np.log(float(coarse["l1"]) / float(fine["l1"])) / np.log(2.0))

    output_directory = ROOT / "benchmarks" / "convergence"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    csv_path = output_directory / "advection_1d_gaussian.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["n_cells", "steps", "l1", "l2", "linf", "l1_order", "mass_error"])
        for n_cells, case, rate in zip(resolutions, cases, rates):
            writer.writerow(
                [n_cells, case["steps"], case["l1"], case["l2"], case["linf"], rate, case["mass_error"]]
            )

    case = cases[2]
    grid = case["grid"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(grid.cell_centres, case["exact"], "k--", label="Exact")
    axes[0].plot(grid.cell_centres, case["numerical"], color="tab:blue", label="Upwind FV")
    axes[0].set(xlabel="x", ylabel="u", title="Gaussian advection (N=200, t=0.5)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    l1_errors = np.array([float(item["l1"]) for item in cases])
    axes[1].loglog(resolutions, l1_errors, "o-", label=r"Measured $L_1$")
    reference = l1_errors[0] * resolutions[0] / np.asarray(resolutions)
    axes[1].loglog(resolutions, reference, "k--", label=r"First order $O(\Delta x)$")
    axes[1].set(xlabel="Number of cells", ylabel=r"$L_1$ error", title="Grid convergence")
    axes[1].legend()
    axes[1].grid(which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_directory / "phase1_gaussian_advection.png", dpi=180)
    plt.close(figure)

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {(figure_directory / 'phase1_gaussian_advection.png').relative_to(ROOT)}")
    for n_cells, measured, rate in zip(resolutions, cases, rates):
        print(
            f"N={n_cells:4d}  L1={float(measured['l1']):.6e}  "
            f"order={rate:.3f}  mass error={float(measured['mass_error']):.3e}"
        )


if __name__ == "__main__":
    main()
