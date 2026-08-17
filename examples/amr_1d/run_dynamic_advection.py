"""Measure dynamic one-level AMR for a transported Gaussian."""

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
from amr.diagnostics.errors import composite_error_norms, error_norms
from amr.grid.grid1d import UniformGrid1D
from amr.grid.hierarchy import AMRHierarchy1D
from amr.refinement.regrid import (
    GradientRegridConfig,
    level_one_regions,
    regrid_from_gradient,
)
from amr.solvers.advection1d import LinearAdvection1D
from amr.solvers.amr_advection1d import AMRAdvectionResult, AMRLinearAdvection1D


FINAL_TIME = 0.3
VELOCITY = 1.0
CFL = 0.8
CONFIG = GradientRegridConfig(
    refine_threshold=3.0,
    derefine_threshold=1.5,
    n_buffer=6,
    merge_gap=4,
)


def profile(x: np.ndarray) -> np.ndarray:
    return gaussian(x, centre=0.25, width=0.07)


def exact(x: np.ndarray) -> np.ndarray:
    return translated_profile(x, FINAL_TIME, VELOCITY, profile)


def uniform_case(n_cells: int) -> dict[str, float | int]:
    grid = UniformGrid1D(0.0, 1.0, n_cells)
    initial = profile(grid.cell_centres)
    result = LinearAdvection1D(grid, VELOCITY, CFL).solve(initial, FINAL_TIME)
    errors = error_norms(result.values, exact(grid.cell_centres))
    return {
        "active_cells": n_cells,
        "peak_stored_cells": n_cells,
        "steps": result.n_steps,
        "cell_updates": n_cells * result.n_steps,
        "regrids": 0,
        "l1": errors.l1,
        "l2": errors.l2,
        "linf": errors.linf,
        "mass_error": total_mass(result.values, grid) - total_mass(initial, grid),
        "max_regrid_mass_change": 0.0,
    }


def amr_case(
    dynamic: bool,
    subcycling: bool = False,
    reflux: bool = False,
) -> tuple[AMRHierarchy1D, AMRAdvectionResult, dict[str, float | int]]:
    grid = UniformGrid1D(0.0, 1.0, 64)
    hierarchy = AMRHierarchy1D(grid, profile(grid.cell_centres), refinement_ratio=2)
    initial_regrid = regrid_from_gradient(hierarchy, CONFIG)
    solver = AMRLinearAdvection1D(
        hierarchy,
        VELOCITY,
        CFL,
        regrid_config=CONFIG if dynamic else None,
        # Both choices regrid every 0.025 time units: four synchronized fine
        # steps or two subcycled coarse steps.
        regrid_interval=2 if subcycling else 4,
        subcycling=subcycling,
        reflux=reflux,
    )
    result = solver.solve(FINAL_TIME)
    errors = composite_error_norms(hierarchy, exact)
    event_changes = [abs(event.mass_change) for event in result.regrid_events]
    return hierarchy, result, {
        "active_cells": hierarchy.n_active_cells,
        "peak_stored_cells": result.peak_stored_cells,
        "steps": result.n_steps,
        "cell_updates": result.cell_updates,
        "regrids": len(result.regrid_events),
        "l1": errors.l1,
        "l2": errors.l2,
        "linf": errors.linf,
        "mass_error": result.mass_error,
        "max_regrid_mass_change": max(
            [abs(initial_regrid.mass_change), *event_changes], default=0.0
        ),
    }


def leaf_data(hierarchy: AMRHierarchy1D) -> tuple[np.ndarray, np.ndarray]:
    root = hierarchy.root
    covered = np.zeros(root.n_valid_cells, dtype=bool)
    for child in root.children:
        if child.parent_range is None:
            raise RuntimeError("Hierarchy invariant violated: child has no parent range")
        start, stop = child.parent_range
        covered[start:stop] = True
    coordinates = [root.grid.cell_centres[~covered]]
    values = [root.values[~covered]]
    for child in root.children:
        coordinates.append(child.grid.cell_centres)
        values.append(child.values)
    x = np.concatenate(coordinates)
    order = np.argsort(x)
    return x[order], np.concatenate(values)[order]


def main() -> None:
    uniform_64 = uniform_case(64)
    uniform_128 = uniform_case(128)
    _, _, static_amr = amr_case(dynamic=False)
    _, _, dynamic_amr = amr_case(dynamic=True)
    _, _, subcycled_amr = amr_case(dynamic=True, subcycling=True)
    hierarchy, dynamic_result, refluxed_amr = amr_case(
        dynamic=True,
        subcycling=True,
        reflux=True,
    )
    named = [
        ("uniform_64", uniform_64),
        ("static_amr", static_amr),
        ("dynamic_amr", dynamic_amr),
        ("subcycled_amr", subcycled_amr),
        ("refluxed_amr", refluxed_amr),
        ("uniform_128", uniform_128),
    ]

    benchmark_directory = ROOT / "benchmarks" / "uniform_vs_amr"
    figure_directory = ROOT / "figures"
    benchmark_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    columns = [
        "method",
        "active_cells",
        "peak_stored_cells",
        "steps",
        "cell_updates",
        "regrids",
        "l1",
        "l2",
        "linf",
        "mass_error",
        "max_regrid_mass_change",
    ]
    csv_path = benchmark_directory / "dynamic_advection_1d.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, values in named:
            writer.writerow({column: name if column == "method" else values[column] for column in columns})

    exact_x = np.linspace(0.0, 1.0, 1000, endpoint=False)
    amr_x, amr_u = leaf_data(hierarchy)
    figure = plt.figure(figsize=(11, 7.5))
    grid_spec = figure.add_gridspec(2, 2, height_ratios=[1.5, 1.0])
    solution_ax = figure.add_subplot(grid_spec[0, 0])
    error_ax = figure.add_subplot(grid_spec[0, 1])
    trajectory_ax = figure.add_subplot(grid_spec[1, :])

    solution_ax.plot(exact_x, exact(exact_x), "k--", label="Exact")
    solution_ax.plot(amr_x, amr_u, ".", color="tab:purple", label="Refluxed dynamic AMR")
    solution_ax.set(xlabel="x", ylabel="u", title=f"Refluxed AMR at t={FINAL_TIME}")
    solution_ax.legend()
    solution_ax.grid(alpha=0.2)

    labels = [
        "Uniform 64",
        "Static AMR",
        "Dynamic AMR",
        "Subcycled AMR",
        "Refluxed AMR",
        "Uniform 128",
    ]
    errors = [float(values["l1"]) for _, values in named]
    error_ax.bar(
        labels,
        errors,
        color=[
            "tab:gray",
            "tab:red",
            "tab:orange",
            "tab:blue",
            "tab:purple",
            "tab:green",
        ],
    )
    error_ax.set(ylabel=r"$L_1$ error", title="Measured accuracy")
    error_ax.tick_params(axis="x", rotation=18)
    error_ax.grid(axis="y", alpha=0.2)

    initial_grid = hierarchy.root.grid
    times = [0.0]
    # Reconstruct the initial region from the first event's old range.
    events = dynamic_result.regrid_events
    regions = [events[0].old_regions if events else level_one_regions(hierarchy)]
    for event in events:
        times.append(event.time)
        regions.append(event.new_regions)
    starts = [initial_grid.cell_edges[item[0][0]] for item in regions]
    stops = [initial_grid.cell_edges[item[0][1]] for item in regions]
    trajectory_ax.step(times, starts, where="post", label="Patch left boundary")
    trajectory_ax.step(times, stops, where="post", label="Patch right boundary")
    trajectory_ax.fill_between(times, starts, stops, step="post", alpha=0.2)
    trajectory_ax.plot(
        times,
        [0.25 + VELOCITY * time for time in times],
        "k--",
        label="Exact Gaussian centre",
    )
    trajectory_ax.set(
        xlabel="Time",
        ylabel="x",
        title="Refined patch follows the transported feature",
        xlim=(0.0, FINAL_TIME),
        ylim=(0.0, 1.0),
    )
    trajectory_ax.legend(ncol=3)
    trajectory_ax.grid(alpha=0.2)
    figure.tight_layout()
    figure_path = figure_directory / "dynamic_amr_advection.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")
    for name, values in named:
        print(
            f"{name:12s} active={int(values['active_cells']):3d} "
            f"updates={int(values['cell_updates']):4d} L1={float(values['l1']):.6e} "
            f"mass change={float(values['mass_error']):+.3e}"
        )
    print(
        "Largest mass change from any single regrid: "
        f"{float(refluxed_amr['max_regrid_mass_change']):.3e}"
    )


if __name__ == "__main__":
    main()
