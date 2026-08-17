"""Repeated AMR diffusion performance and refinement-sensitivity study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amr.benchmarks.diffusion import periodic_gaussian_diffusion_cell_averages
from amr.diagnostics.errors import composite_cell_average_error_norms, error_norms
from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.refinement.regrid import GradientRegridConfig, regrid_from_gradient
from amr.solvers.amr_diffusion1d import AMRExplicitDiffusion1D
from amr.solvers.diffusion1d import ExplicitDiffusion1D


DIFFUSIVITY = 0.01
FINAL_TIME = 0.05
STABILITY_FACTOR = 0.8
REFINEMENT_RATIO = 2


def analytical_averages(
    edges: np.ndarray, time: float = FINAL_TIME
) -> np.ndarray:
    return periodic_gaussian_diffusion_cell_averages(edges, time, DIFFUSIVITY)


def run_uniform(n_cells: int) -> dict[str, float | int]:
    grid = UniformGrid1D(0.0, 1.0, n_cells)
    initial = analytical_averages(grid.cell_edges, 0.0)
    result = ExplicitDiffusion1D(grid, DIFFUSIVITY, STABILITY_FACTOR).solve(
        initial, FINAL_TIME
    )
    errors = error_norms(result.values, analytical_averages(grid.cell_edges))
    return {
        "final_active_cells": n_cells,
        "peak_active_cells": n_cells,
        "peak_stored_cells": n_cells,
        "final_refined_fraction": 0.0,
        "cell_updates": n_cells * result.n_steps,
        "coarse_steps": result.n_steps,
        "fine_steps": 0,
        "regrid_events": 0,
        "max_regrid_mass_change": 0.0,
        "l1": errors.l1,
        "l2": errors.l2,
        "linf": errors.linf,
        "mass_error": float(np.sum(result.values - initial) * grid.dx),
    }


def run_amr(
    base_cells: int,
    refine_threshold: float = 1.0,
    n_buffer: int = 4,
    prolongation: str = "conservative_quadratic",
) -> dict[str, float | int]:
    grid = UniformGrid1D(0.0, 1.0, base_cells)
    hierarchy = AMRHierarchy1D(
        grid,
        analytical_averages(grid.cell_edges, 0.0),
        refinement_ratio=REFINEMENT_RATIO,
    )
    config = GradientRegridConfig(
        refine_threshold=refine_threshold,
        derefine_threshold=0.5 * refine_threshold,
        n_buffer=n_buffer,
        merge_gap=4,
        prolongation=prolongation,
    )
    regrid_from_gradient(hierarchy, config)
    result = AMRExplicitDiffusion1D(
        hierarchy,
        DIFFUSIVITY,
        STABILITY_FACTOR,
        regrid_config=config,
        regrid_interval=2,
        subcycling=True,
        reflux=True,
    ).solve(FINAL_TIME)
    errors = composite_cell_average_error_norms(hierarchy, analytical_averages)
    refined_parent_cells = 0
    for child in hierarchy.root.children:
        if child.parent_range is None:
            raise RuntimeError("Hierarchy invariant violated: child has no parent range")
        start, stop = child.parent_range
        refined_parent_cells += stop - start
    return {
        "final_active_cells": hierarchy.n_active_cells,
        "peak_active_cells": result.peak_active_cells,
        "peak_stored_cells": result.peak_stored_cells,
        "final_refined_fraction": refined_parent_cells / base_cells,
        "cell_updates": result.cell_updates,
        "coarse_steps": result.n_steps,
        "fine_steps": result.fine_steps,
        "regrid_events": len(result.regrid_events),
        "max_regrid_mass_change": max(
            (abs(event.mass_change) for event in result.regrid_events), default=0.0
        ),
        "l1": errors.l1,
        "l2": errors.l2,
        "linf": errors.linf,
        "mass_error": result.mass_error,
    }


def timed_run(
    function: Callable[[], dict[str, float | int]], repeats: int
) -> tuple[dict[str, float | int], list[float]]:
    """Warm once, then time complete fresh calculations."""

    function()
    samples = []
    result = {}
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        samples.append(perf_counter() - start)
    return result, samples


def runtime_summary(samples: list[float]) -> dict[str, float]:
    return {
        "runtime_median_s": float(np.median(samples)),
        "runtime_min_s": float(np.min(samples)),
        "runtime_max_s": float(np.max(samples)),
    }


def write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--sensitivity-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3 or args.sensitivity_repeats < 3:
        parser.error("both repeat counts must be at least 3")

    scaling_records: list[dict[str, float | int | str]] = []
    for base_cells in (32, 64, 128):
        calculations = (
            ("uniform_coarse", lambda n=base_cells: run_uniform(n)),
            ("dynamic_amr", lambda n=base_cells: run_amr(n)),
            ("uniform_fine", lambda n=2 * base_cells: run_uniform(n)),
        )
        for method, calculation in calculations:
            result, samples = timed_run(calculation, args.repeats)
            scaling_records.append(
                {
                    "base_cells": base_cells,
                    "method": method,
                    **result,
                    **runtime_summary(samples),
                    "repeats": args.repeats,
                }
            )

    sensitivity_records: list[dict[str, float | int | str]] = []
    for threshold in (0.5, 1.0, 2.0):
        for n_buffer in (2, 4, 8):
            calculation = lambda t=threshold, b=n_buffer: run_amr(64, t, b)
            result, samples = timed_run(calculation, args.sensitivity_repeats)
            sensitivity_records.append(
                {
                    "base_cells": 64,
                    "refine_threshold": threshold,
                    "derefine_threshold": 0.5 * threshold,
                    "buffer_cells": n_buffer,
                    **result,
                    **runtime_summary(samples),
                    "repeats": args.sensitivity_repeats,
                }
            )

    prolongation_records: list[dict[str, float | int | str]] = []
    for prolongation in (
        "piecewise_constant",
        "conservative_linear",
        "conservative_quadratic",
    ):
        calculation = lambda method=prolongation: run_amr(
            64, prolongation=method
        )
        result, samples = timed_run(calculation, args.sensitivity_repeats)
        prolongation_records.append(
            {
                "base_cells": 64,
                "prolongation": prolongation,
                **result,
                **runtime_summary(samples),
                "repeats": args.sensitivity_repeats,
            }
        )

    output_directory = ROOT / "benchmarks" / "performance"
    figure_directory = ROOT / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    scaling_path = output_directory / "diffusion_accuracy_runtime.csv"
    sensitivity_path = output_directory / "diffusion_refinement_sensitivity.csv"
    prolongation_path = output_directory / "diffusion_prolongation_comparison.csv"
    metadata_path = output_directory / "diffusion_accuracy_runtime_metadata.json"
    write_csv(scaling_path, scaling_records)
    write_csv(sensitivity_path, sensitivity_records)
    write_csv(prolongation_path, prolongation_records)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "repeats": args.repeats,
        "sensitivity_repeats": args.sensitivity_repeats,
        "timer": "time.perf_counter",
        "scope": "initialization, initial regrid, integration, and diagnostics; one untimed warm-up per case",
        "diffusivity": DIFFUSIVITY,
        "final_time": FINAL_TIME,
        "stability_factor": STABILITY_FACTOR,
        "refinement_ratio": REFINEMENT_RATIO,
        "state_interpretation": "analytical finite-volume cell averages",
        "amr_prolongation": "conservative_quadratic",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    colours = {
        "uniform_coarse": "tab:gray",
        "dynamic_amr": "tab:cyan",
        "uniform_fine": "tab:green",
    }
    labels = {
        "uniform_coarse": "Uniform coarse",
        "dynamic_amr": "Dynamic AMR",
        "uniform_fine": "Uniform fine",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    for method in colours:
        selected = [record for record in scaling_records if record["method"] == method]
        axes[0, 0].loglog(
            [record["runtime_median_s"] for record in selected],
            [record["l1"] for record in selected],
            "o-",
            color=colours[method],
            label=labels[method],
        )
        axes[0, 1].loglog(
            [record["cell_updates"] for record in selected],
            [record["l1"] for record in selected],
            "o-",
            color=colours[method],
            label=labels[method],
        )
    for threshold, marker in zip((0.5, 1.0, 2.0), ("o", "s", "^")):
        selected = [
            record
            for record in sensitivity_records
            if record["refine_threshold"] == threshold
        ]
        axes[1, 0].plot(
            [record["final_refined_fraction"] for record in selected],
            [record["l1"] for record in selected],
            marker + "-",
            label=f"threshold {threshold:g}",
        )
        axes[1, 1].plot(
            [record["runtime_median_s"] for record in selected],
            [record["l1"] for record in selected],
            marker + "-",
            label=f"threshold {threshold:g}",
        )

    axes[0, 0].set(
        xlabel="Median runtime [s]", ylabel=r"$L_1$ error", title="Accuracy versus runtime"
    )
    axes[0, 1].set(
        xlabel="Cell updates", ylabel=r"$L_1$ error", title="Accuracy versus updates"
    )
    axes[1, 0].set(
        xlabel="Final refined fraction",
        ylabel=r"$L_1$ error",
        title="Sensitivity to refined coverage",
    )
    axes[1, 1].set(
        ylabel=r"$L_1$ error",
        title="New-patch initialization",
    )
    short_labels = ["Constant", "Limited linear", "Quadratic"]
    axes[1, 1].clear()
    axes[1, 1].bar(
        short_labels,
        [record["l1"] for record in prolongation_records],
        color=["tab:gray", "tab:blue", "tab:purple"],
    )
    uniform_64_error = next(
        record["l1"]
        for record in scaling_records
        if record["base_cells"] == 64 and record["method"] == "uniform_coarse"
    )
    uniform_128_error = next(
        record["l1"]
        for record in scaling_records
        if record["base_cells"] == 64 and record["method"] == "uniform_fine"
    )
    axes[1, 1].axhline(
        uniform_64_error, color="tab:gray", linestyle="--", label="Uniform 64"
    )
    axes[1, 1].axhline(
        uniform_128_error, color="tab:green", linestyle="--", label="Uniform 128"
    )
    axes[1, 1].set(ylabel=r"$L_1$ error", title="New-patch initialization")
    axes[1, 1].tick_params(axis="x", rotation=12)
    for axis in axes.flat:
        axis.grid(which="both", alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure_path = figure_directory / "diffusion_accuracy_runtime.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(f"Wrote {scaling_path.relative_to(ROOT)}")
    print(f"Wrote {sensitivity_path.relative_to(ROOT)}")
    print(f"Wrote {prolongation_path.relative_to(ROOT)}")
    print(f"Wrote {metadata_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")
    for record in scaling_records:
        print(
            f"base={record['base_cells']:3d} {record['method']:14s} "
            f"L1={record['l1']:.4e} updates={record['cell_updates']:7d} "
            f"median={record['runtime_median_s']:.6f}s"
        )


if __name__ == "__main__":
    main()
