"""Repeated accuracy and runtime study for uniform and refluxed AMR advection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amr.benchmarks.advection import gaussian, translated_profile
from amr.diagnostics.errors import composite_error_norms, error_norms
from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.refinement.regrid import GradientRegridConfig, regrid_from_gradient
from amr.solvers.advection1d import LinearAdvection1D
from amr.solvers.amr_advection1d import AMRLinearAdvection1D


FINAL_TIME = 0.3
VELOCITY = 1.0
CFL = 0.8


def profile(x: np.ndarray) -> np.ndarray:
    return gaussian(x, centre=0.25, width=0.07)


def run_uniform(n_cells: int) -> dict[str, float | int]:
    grid = UniformGrid1D(0.0, 1.0, n_cells)
    initial = profile(grid.cell_centres)
    result = LinearAdvection1D(grid, VELOCITY, CFL).solve(initial, FINAL_TIME)
    exact = translated_profile(grid.cell_centres, FINAL_TIME, VELOCITY, profile)
    errors = error_norms(result.values, exact)
    return {
        "active_cells": n_cells,
        "peak_stored_cells": n_cells,
        "cell_updates": n_cells * result.n_steps,
        "l1": errors.l1,
        "l2": errors.l2,
        "linf": errors.linf,
        "mass_error": float(np.sum(result.values - initial) * grid.dx),
    }


def run_amr(base_cells: int) -> dict[str, float | int]:
    grid = UniformGrid1D(0.0, 1.0, base_cells)
    hierarchy = AMRHierarchy1D(grid, profile(grid.cell_centres), refinement_ratio=2)
    config = GradientRegridConfig(
        refine_threshold=3.0,
        derefine_threshold=1.5,
        n_buffer=6,
        merge_gap=4,
    )
    regrid_from_gradient(hierarchy, config)
    result = AMRLinearAdvection1D(
        hierarchy,
        VELOCITY,
        CFL,
        regrid_config=config,
        regrid_interval=2,
        subcycling=True,
        reflux=True,
    ).solve(FINAL_TIME)
    exact = lambda x: translated_profile(x, FINAL_TIME, VELOCITY, profile)
    errors = composite_error_norms(hierarchy, exact)
    return {
        "active_cells": hierarchy.n_active_cells,
        "peak_stored_cells": result.peak_stored_cells,
        "cell_updates": result.cell_updates,
        "l1": errors.l1,
        "l2": errors.l2,
        "linf": errors.linf,
        "mass_error": result.mass_error,
    }


def timed_run(function, argument: int, repeats: int) -> tuple[dict[str, float | int], list[float]]:
    """Warm once, then return a fresh result and repeated wall-clock samples."""

    function(argument)
    samples = []
    result = {}
    for _ in range(repeats):
        start = perf_counter()
        result = function(argument)
        samples.append(perf_counter() - start)
    return result, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")

    records = []
    for base_cells in (32, 64, 128):
        calculations = (
            ("uniform_coarse", run_uniform, base_cells),
            ("refluxed_amr", run_amr, base_cells),
            ("uniform_fine", run_uniform, 2 * base_cells),
        )
        for method, function, argument in calculations:
            result, samples = timed_run(function, argument, args.repeats)
            records.append(
                {
                    "base_cells": base_cells,
                    "method": method,
                    **result,
                    "runtime_median_s": float(np.median(samples)),
                    "runtime_min_s": float(np.min(samples)),
                    "runtime_max_s": float(np.max(samples)),
                    "repeats": args.repeats,
                }
            )

    output_directory = ROOT / "benchmarks" / "performance"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "advection_accuracy_runtime.csv"
    columns = list(records[0])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "repeats": args.repeats,
        "timer": "time.perf_counter",
        "scope": "initialization plus integration; one untimed warm-up per case",
    }
    metadata_path = output_directory / "advection_accuracy_runtime_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    colours = {
        "uniform_coarse": "tab:gray",
        "refluxed_amr": "tab:purple",
        "uniform_fine": "tab:green",
    }
    labels = {
        "uniform_coarse": "Uniform coarse",
        "refluxed_amr": "Dynamic refluxed AMR",
        "uniform_fine": "Uniform fine",
    }
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for method in colours:
        selected = [record for record in records if record["method"] == method]
        axes[0].loglog(
            [record["runtime_median_s"] for record in selected],
            [record["l1"] for record in selected],
            "o-",
            color=colours[method],
            label=labels[method],
        )
        axes[1].loglog(
            [record["cell_updates"] for record in selected],
            [record["l1"] for record in selected],
            "o-",
            color=colours[method],
            label=labels[method],
        )
        axes[2].loglog(
            [record["base_cells"] for record in selected],
            [record["runtime_median_s"] for record in selected],
            "o-",
            color=colours[method],
            label=labels[method],
        )

    axes[0].set(xlabel="Median runtime [s]", ylabel=r"$L_1$ error", title="Accuracy versus runtime")
    axes[1].set(xlabel="Cell updates", ylabel=r"$L_1$ error", title="Accuracy versus updates")
    axes[2].set(xlabel="Base-grid cells", ylabel="Median runtime [s]", title="Runtime scaling")
    for axis in axes:
        axis.grid(which="both", alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure_path = figure_directory / "advection_accuracy_runtime.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {metadata_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")
    for record in records:
        print(
            f"base={record['base_cells']:3d} {record['method']:14s} "
            f"L1={record['l1']:.4e} updates={record['cell_updates']:6d} "
            f"median={record['runtime_median_s']:.6f}s"
        )


if __name__ == "__main__":
    main()
