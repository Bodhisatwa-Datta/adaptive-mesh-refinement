"""Repeated runtime and peak-memory study for the two-dimensional solvers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amr.benchmarks.advection2d import periodic_gaussian_2d, translated_gaussian_2d
from amr.benchmarks.diffusion2d import periodic_gaussian_diffusion_2d_cell_averages
from amr.diagnostics.errors import (
    composite_cell_average_error_norms_2d,
    composite_error_norms_2d,
    error_norms,
)
from amr.grid.grid2d import UniformGrid2D
from amr.grid.hierarchy2d import AMRHierarchy2D
from amr.refinement.regrid2d import GradientRegridConfig2D, regrid_from_gradient_2d
from amr.solvers.advection2d import LinearAdvection2D
from amr.solvers.amr_advection2d import AMRLinearAdvection2D
from amr.solvers.amr_diffusion2d import AMRExplicitDiffusion2D
from amr.solvers.diffusion2d import ExplicitDiffusion2D

Measurement = dict[str, float | int]
Calculation = Callable[[], Measurement]
ADVECTION_TIME = 0.5
DIFFUSION_TIME = 0.1
VELOCITY = (0.6, 0.3)
DIFFUSIVITY = 0.01
ADVECTION_PROFILE = {"centre": (0.3, 0.3), "width": (0.06, 0.06)}
DIFFUSION_PROFILE = {"centre": (0.5, 0.5), "initial_width": (0.06, 0.06)}


def run_uniform_advection(n_cells: int) -> Measurement:
    """Run one complete uniform advection calculation."""

    grid = UniformGrid2D(0.0, 1.0, n_cells, 0.0, 1.0, n_cells)
    x, y = grid.cell_centres
    initial = periodic_gaussian_2d(x, y, **ADVECTION_PROFILE)
    result = LinearAdvection2D(grid, *VELOCITY).solve(initial, ADVECTION_TIME)
    exact = translated_gaussian_2d(
        x, y, ADVECTION_TIME, VELOCITY, **ADVECTION_PROFILE
    )
    errors = error_norms(result.values, exact)
    cells = n_cells * n_cells
    return {
        "final_active_cells": cells,
        "peak_stored_cells": cells,
        "cell_updates": cells * result.n_steps,
        "l1": errors.l1,
        "mass_error": float(np.sum(result.values - initial) * grid.cell_area),
    }


def run_amr_advection(base_cells: int) -> Measurement:
    """Run one complete dynamic, subcycled, refluxed AMR advection calculation."""

    grid = UniformGrid2D(0.0, 1.0, base_cells, 0.0, 1.0, base_cells)
    x, y = grid.cell_centres
    hierarchy = AMRHierarchy2D(
        grid,
        periodic_gaussian_2d(x, y, **ADVECTION_PROFILE),
        refinement_ratio=2,
    )
    config = GradientRegridConfig2D(2.0, 1.0, n_buffer=3, merge_gap=1)
    regrid_from_gradient_2d(hierarchy, config)
    result = AMRLinearAdvection2D(
        hierarchy,
        *VELOCITY,
        reflux=True,
        subcycling=True,
        regrid_config=config,
        regrid_interval=2,
    ).solve(ADVECTION_TIME)
    exact = lambda x_values, y_values: translated_gaussian_2d(
        x_values,
        y_values,
        ADVECTION_TIME,
        VELOCITY,
        **ADVECTION_PROFILE,
    )
    errors = composite_error_norms_2d(hierarchy, exact)
    return {
        "final_active_cells": hierarchy.n_active_cells,
        "peak_stored_cells": result.peak_stored_cells,
        "cell_updates": result.cell_updates,
        "l1": errors.l1,
        "mass_error": result.mass_error,
    }


def diffusion_averages(grid: UniformGrid2D, time: float) -> np.ndarray:
    return periodic_gaussian_diffusion_2d_cell_averages(
        grid.x_edges,
        grid.y_edges,
        time,
        DIFFUSIVITY,
        **DIFFUSION_PROFILE,
    )


def run_uniform_diffusion(n_cells: int) -> Measurement:
    """Run one complete uniform diffusion calculation."""

    grid = UniformGrid2D(0.0, 1.0, n_cells, 0.0, 1.0, n_cells)
    initial = diffusion_averages(grid, 0.0)
    result = ExplicitDiffusion2D(grid, DIFFUSIVITY).solve(initial, DIFFUSION_TIME)
    errors = error_norms(result.values, diffusion_averages(grid, DIFFUSION_TIME))
    cells = n_cells * n_cells
    return {
        "final_active_cells": cells,
        "peak_stored_cells": cells,
        "cell_updates": cells * result.n_steps,
        "l1": errors.l1,
        "mass_error": float(np.sum(result.values - initial) * grid.cell_area),
    }


def run_amr_diffusion(base_cells: int) -> Measurement:
    """Run one complete dynamic, subcycled, refluxed AMR diffusion calculation."""

    grid = UniformGrid2D(0.0, 1.0, base_cells, 0.0, 1.0, base_cells)
    hierarchy = AMRHierarchy2D(grid, diffusion_averages(grid, 0.0), refinement_ratio=2)
    config = GradientRegridConfig2D(
        1.0,
        0.5,
        n_buffer=4,
        merge_gap=1,
        prolongation="conservative_quadratic",
    )
    regrid_from_gradient_2d(hierarchy, config)
    result = AMRExplicitDiffusion2D(
        hierarchy,
        DIFFUSIVITY,
        reflux=True,
        subcycling=True,
        regrid_config=config,
    ).solve(DIFFUSION_TIME)
    exact = lambda x_edges, y_edges: periodic_gaussian_diffusion_2d_cell_averages(
        x_edges,
        y_edges,
        DIFFUSION_TIME,
        DIFFUSIVITY,
        **DIFFUSION_PROFILE,
    )
    errors = composite_cell_average_error_norms_2d(hierarchy, exact)
    return {
        "final_active_cells": hierarchy.n_active_cells,
        "peak_stored_cells": result.peak_stored_cells,
        "cell_updates": result.cell_updates,
        "l1": errors.l1,
        "mass_error": result.mass_error,
    }


def measure(calculation: Calculation, repeats: int) -> Measurement:
    """Warm once, time repeated runs, then separately sample traced peak memory."""

    calculation()
    samples = []
    result: Measurement = {}
    for _ in range(repeats):
        start = perf_counter()
        result = calculation()
        samples.append(perf_counter() - start)
    tracemalloc.start()
    calculation()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        **result,
        "runtime_median_s": float(np.median(samples)),
        "runtime_min_s": float(np.min(samples)),
        "runtime_max_s": float(np.max(samples)),
        "traced_peak_bytes": peak_bytes,
    }


def write_outputs(records: list[dict[str, float | int | str]], repeats: int) -> None:
    output_directory = ROOT / "benchmarks" / "performance"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "two_dimensional_accuracy_runtime_memory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "repeats": repeats,
        "timer": "time.perf_counter",
        "memory": "tracemalloc peak from one separate complete calculation",
        "scope": "initialization, initial regrid, integration, and diagnostics",
        "warm_up": "one untimed run per case",
    }
    metadata_path = output_directory / "two_dimensional_benchmark_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    colours = {
        "uniform_coarse": "tab:gray",
        "dynamic_amr": "tab:orange",
        "uniform_fine": "tab:blue",
    }
    for row, equation in enumerate(("advection", "diffusion")):
        equation_records = [record for record in records if record["equation"] == equation]
        for method, colour in colours.items():
            selected = [record for record in equation_records if record["method"] == method]
            label = method.replace("_", " ").title()
            axes[row, 0].loglog(
                [record["runtime_median_s"] for record in selected],
                [record["l1"] for record in selected],
                "o-",
                color=colour,
                label=label,
            )
            axes[row, 1].plot(
                [record["base_cells"] for record in selected],
                [record["traced_peak_bytes"] / 1024**2 for record in selected],
                "o-",
                color=colour,
                label=label,
            )
        axes[row, 0].set(
            xlabel="Median runtime [s]",
            ylabel=r"$L_1$ error",
            title=f"{equation.title()}: accuracy versus runtime",
        )
        axes[row, 1].set(
            xlabel="Base cells per direction",
            ylabel="Traced peak [MiB]",
            title=f"{equation.title()}: traced memory",
        )
        for axis in axes[row]:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure_path = figure_directory / "two_dimensional_performance.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {metadata_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--base-cells", type=int, nargs="+", default=[24, 32, 48])
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    if any(n_cells < 8 for n_cells in args.base_cells):
        parser.error("every base resolution must be at least 8")

    records: list[dict[str, float | int | str]] = []
    runners = {
        "advection": (run_uniform_advection, run_amr_advection),
        "diffusion": (run_uniform_diffusion, run_amr_diffusion),
    }
    for equation, (uniform, amr) in runners.items():
        for base_cells in args.base_cells:
            calculations = (
                ("uniform_coarse", lambda n=base_cells, run=uniform: run(n)),
                ("dynamic_amr", lambda n=base_cells, run=amr: run(n)),
                ("uniform_fine", lambda n=2 * base_cells, run=uniform: run(n)),
            )
            for method, calculation in calculations:
                result = measure(calculation, args.repeats)
                record = {
                    "equation": equation,
                    "base_cells": base_cells,
                    "method": method,
                    **result,
                    "repeats": args.repeats,
                }
                records.append(record)
                print(
                    f"{equation:9s} base={base_cells:2d} {method:14s} "
                    f"L1={record['l1']:.4e} median={record['runtime_median_s']:.5f}s "
                    f"peak={record['traced_peak_bytes'] / 1024**2:.2f} MiB"
                )
    write_outputs(records, args.repeats)


if __name__ == "__main__":
    main()
