"""Compare first- and second-order finite-volume advection."""

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

from amr.benchmarks.advection import sinusoid, translated_profile
from amr.diagnostics.conservation import total_mass
from amr.diagnostics.errors import error_norms
from amr.grid.grid1d import UniformGrid1D
from amr.solvers.advection1d import LinearAdvection1D
from amr.solvers.second_order_advection1d import SecondOrderLinearAdvection1D


FINAL_TIME = 0.5
VELOCITY = 1.0
CFL = 0.6


def run_case(n_cells: int) -> dict[str, object]:
    grid = UniformGrid1D(0.0, 1.0, n_cells)
    initial = sinusoid(grid.cell_centres)
    exact = translated_profile(grid.cell_centres, FINAL_TIME, VELOCITY, sinusoid)
    first = LinearAdvection1D(grid, VELOCITY, CFL).solve(initial, FINAL_TIME)
    second = SecondOrderLinearAdvection1D(grid, VELOCITY, CFL).solve(
        initial, FINAL_TIME
    )
    first_error = error_norms(first.values, exact)
    second_error = error_norms(second.values, exact)
    return {
        "grid": grid,
        "initial": initial,
        "exact": exact,
        "first": first.values,
        "second": second.values,
        "steps": second.n_steps,
        "first_l1": first_error.l1,
        "second_l1": second_error.l1,
        "second_l2": second_error.l2,
        "second_linf": second_error.linf,
        "mass_error": total_mass(second.values, grid) - total_mass(initial, grid),
    }


def main() -> None:
    resolutions = [40, 80, 160, 320]
    cases = [run_case(n_cells) for n_cells in resolutions]
    first_orders = [np.nan]
    second_orders = [np.nan]
    for coarse, fine in zip(cases[:-1], cases[1:]):
        first_orders.append(
            np.log(float(coarse["first_l1"]) / float(fine["first_l1"])) / np.log(2.0)
        )
        second_orders.append(
            np.log(float(coarse["second_l1"]) / float(fine["second_l1"])) / np.log(2.0)
        )

    output_directory = ROOT / "benchmarks" / "convergence"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "advection_1d_second_order.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "n_cells",
                "steps",
                "first_l1",
                "first_l1_order",
                "second_l1",
                "second_l2",
                "second_linf",
                "second_l1_order",
                "mass_error",
            ]
        )
        for n_cells, case, first_order, second_order in zip(
            resolutions, cases, first_orders, second_orders
        ):
            writer.writerow(
                [
                    n_cells,
                    case["steps"],
                    case["first_l1"],
                    first_order,
                    case["second_l1"],
                    case["second_l2"],
                    case["second_linf"],
                    second_order,
                    case["mass_error"],
                ]
            )

    case = cases[1]
    grid = case["grid"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(grid.cell_centres, case["exact"], "k--", label="Exact")
    axes[0].plot(grid.cell_centres, case["first"], label="First-order upwind")
    axes[0].plot(grid.cell_centres, case["second"], label="MUSCL + SSP-RK2")
    axes[0].set(xlabel="x", ylabel="u", title="Sinusoidal advection (N=80)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].loglog(
        resolutions,
        [case["first_l1"] for case in cases],
        "o-",
        label="First-order upwind",
    )
    axes[1].loglog(
        resolutions,
        [case["second_l1"] for case in cases],
        "o-",
        label="MUSCL + SSP-RK2",
    )
    axes[1].set(xlabel="Number of cells", ylabel=r"$L_1$ error", title="Measured convergence")
    axes[1].legend()
    axes[1].grid(which="both", alpha=0.25)
    figure.tight_layout()
    figure_path = figure_directory / "advection_second_order_convergence.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")
    for n_cells, case, order in zip(resolutions, cases, second_orders):
        print(
            f"N={n_cells:4d} second L1={float(case['second_l1']):.6e} "
            f"order={order:.3f} mass change={float(case['mass_error']):+.3e}"
        )


if __name__ == "__main__":
    main()
