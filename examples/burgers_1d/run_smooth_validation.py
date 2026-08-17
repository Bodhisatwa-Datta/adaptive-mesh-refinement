"""Validate the uniform Burgers solver against the pre-shock exact solution."""

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

from amr.benchmarks.burgers import exact_smooth_solution, smooth_periodic_profile
from amr.diagnostics.conservation import total_mass
from amr.diagnostics.errors import error_norms
from amr.grid.grid1d import UniformGrid1D
from amr.solvers.burgers1d import InviscidBurgers1D


FINAL_TIME = 0.2


def run_case(n_cells: int) -> dict[str, object]:
    grid = UniformGrid1D(0.0, 1.0, n_cells)
    initial = smooth_periodic_profile(grid.cell_centres)
    result = InviscidBurgers1D(grid, cfl=0.8).solve(initial, FINAL_TIME)
    exact = exact_smooth_solution(grid.cell_centres, FINAL_TIME)
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
    resolutions = [50, 100, 200, 400]
    cases = [run_case(n_cells) for n_cells in resolutions]
    orders = [np.nan]
    for coarse, fine in zip(cases[:-1], cases[1:]):
        orders.append(np.log(float(coarse["l1"]) / float(fine["l1"])) / np.log(2.0))

    output_directory = ROOT / "benchmarks" / "convergence"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "burgers_1d_smooth.csv"
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
    axes[0].plot(grid.cell_centres, case["exact"], "k--", label="Exact characteristic solution")
    axes[0].plot(grid.cell_centres, case["values"], color="tab:blue", label="Rusanov FV")
    axes[0].set(xlabel="x", ylabel="u", title="Smooth Burgers solution (N=200, t=0.2)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    l1_errors = np.array([float(case["l1"]) for case in cases])
    axes[1].loglog(resolutions, l1_errors, "o-", label=r"Measured $L_1$")
    axes[1].loglog(
        resolutions,
        l1_errors[0] * resolutions[0] / np.asarray(resolutions),
        "k--",
        label=r"First order $O(\Delta x)$",
    )
    axes[1].set(xlabel="Number of cells", ylabel=r"$L_1$ error", title="Pre-shock convergence")
    axes[1].legend()
    axes[1].grid(which="both", alpha=0.25)
    figure.tight_layout()
    figure_path = figure_directory / "burgers_smooth_convergence.png"
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
