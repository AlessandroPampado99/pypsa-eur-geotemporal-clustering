#!/usr/bin/env python3
"""Plot solve time and peak memory for all runs below a results prefix."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Benchmark:
    name: str
    seconds: float
    max_rss_mb: float


NODE_COUNT_PATTERN = re.compile(r"(?:^|-)n(\d+)(?:-|\Z)")


def run_order(benchmark: Benchmark) -> tuple[int, int, str]:
    """Order runs by the numeric node count in names such as gtf-n10-d40."""
    match = NODE_COUNT_PATTERN.search(benchmark.name)
    if match:
        return (0, int(match.group(1)), benchmark.name)
    return (1, 0, benchmark.name)


def annotate_values(axis, x_values, y_values, unit: str) -> None:
    """Add compact, readable values above the available data points."""
    for x, value in zip(x_values, y_values):
        if not math.isfinite(value):
            continue
        axis.annotate(
            f"{value:.0f} {unit}",
            (x, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Snakemake solve_network benchmarks below results/PREFIX and "
            "plot wall time and peak resident memory."
        )
    )
    parser.add_argument(
        "prefix",
        help="Results prefix (for example DE_400) or path to its directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results root used when PREFIX is a name (default: results).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output figure (default: PREFIX/solve_network_benchmarks.png).",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        help="Output data (default: PREFIX/solve_network_benchmarks.csv).",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Only plot solve time, omitting the memory subplot.",
    )
    return parser.parse_args()


def resolve_prefix(prefix: str, results_dir: Path) -> Path:
    path = Path(prefix)
    return path if path.is_dir() else results_dir / prefix


def read_benchmark(path: Path, run_name: str) -> Benchmark:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Benchmark is empty: {path}")
    row = rows[-1]
    try:
        seconds = float(row["s"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Expected a numeric 's' column in {path}") from exc
    try:
        max_rss_mb = float(row["max_rss"])
    except (KeyError, ValueError):
        max_rss_mb = math.nan
        print(f"Warning: peak memory is unavailable for {run_name}")
    return Benchmark(run_name, seconds, max_rss_mb)


def collect_benchmarks(prefix_dir: Path) -> list[Benchmark]:
    if not prefix_dir.is_dir():
        raise FileNotFoundError(f"Results prefix does not exist: {prefix_dir}")

    benchmarks: list[Benchmark] = []
    for run_dir in prefix_dir.iterdir():
        benchmark_dir = run_dir / "benchmarks" / "solve_network"
        if not run_dir.is_dir() or not benchmark_dir.is_dir():
            continue
        files = [path for path in benchmark_dir.iterdir() if path.is_file()]
        if not files:
            print(f"Warning: no solve benchmark found for {run_dir.name}")
            continue
        if len(files) > 1:
            names = ", ".join(path.name for path in files)
            raise ValueError(
                f"Expected one solve benchmark for {run_dir.name}, found: {names}"
            )
        benchmarks.append(read_benchmark(files[0], run_dir.name))

    if not benchmarks:
        raise ValueError(f"No solve_network benchmarks found below {prefix_dir}")
    return benchmarks


def write_benchmarks_csv(benchmarks: list[Benchmark], output: Path) -> None:
    """Write every plotted point, including the complete-network reference."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run", "nodes", "seconds", "max_rss_mb", "is_reference"],
        )
        writer.writeheader()
        for item in sorted(benchmarks, key=run_order):
            match = NODE_COUNT_PATTERN.search(item.name)
            writer.writerow(
                {
                    "run": item.name,
                    "nodes": int(match.group(1)) if match else "",
                    "seconds": item.seconds,
                    "max_rss_mb": item.max_rss_mb,
                    "is_reference": item.name == "complete",
                }
            )


def plot_benchmarks(
    benchmarks: list[Benchmark], output: Path, include_memory: bool
) -> None:
    complete = next((item for item in benchmarks if item.name == "complete"), None)
    runs = sorted(
        (item for item in benchmarks if item.name != "complete"), key=run_order
    )
    if not runs:
        raise ValueError("No non-complete runs are available to plot")

    figure, axes = plt.subplots(
        2 if include_memory else 1,
        1,
        figsize=(max(10, 0.55 * len(runs)), 8 if include_memory else 5),
        sharex=include_memory,
        constrained_layout=True,
        squeeze=False,
    )
    x = range(len(runs))

    time_axis = axes[0, 0]
    solve_times = [item.seconds for item in runs]
    time_axis.plot(x, solve_times, "o-", label="Clustered runs")
    annotate_values(time_axis, x, solve_times, "s")
    if complete is not None:
        time_axis.axhline(
            complete.seconds,
            color="tab:red",
            linestyle="--",
            label=f"Complete ({complete.seconds:.0f} s)",
        )
        time_axis.text(
            0.99,
            complete.seconds,
            f" {complete.seconds:.0f} s",
            transform=time_axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color="tab:red",
        )
    time_axis.set_yscale("log")
    time_axis.set_ylabel("Solve time [s]")
    time_axis.grid(axis="y", which="both", alpha=0.3)
    time_axis.legend()

    bottom_axis = time_axis
    if include_memory:
        memory_axis = axes[1, 0]
        peak_memory = [item.max_rss_mb / 1024 for item in runs]
        memory_axis.plot(
            x,
            peak_memory,
            "o-",
            color="tab:green",
            label="Clustered runs",
        )
        annotate_values(memory_axis, x, peak_memory, "GiB")
        if complete is not None:
            memory_axis.axhline(
                complete.max_rss_mb / 1024,
                color="tab:red",
                linestyle="--",
                label=f"Complete ({complete.max_rss_mb / 1024:.1f} GiB)",
            )
            memory_axis.text(
                0.99,
                complete.max_rss_mb / 1024,
                f" {complete.max_rss_mb / 1024:.0f} GiB",
                transform=memory_axis.get_yaxis_transform(),
                ha="right",
                va="bottom",
                color="tab:red",
            )
        memory_axis.set_yscale("log")
        memory_axis.set_ylabel("Peak RSS [GiB]")
        memory_axis.grid(axis="y", which="both", alpha=0.3)
        memory_axis.legend()
        bottom_axis = memory_axis

    bottom_axis.set_xticks(
        list(x), [item.name for item in runs], rotation=45, ha="right"
    )
    bottom_axis.set_xlabel("Run")
    figure.suptitle(f"solve_network benchmarks: {output.parent.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    prefix_dir = resolve_prefix(args.prefix, args.results_dir)
    output = args.output or prefix_dir / "solve_network_benchmarks.png"
    csv_output = args.csv_output or prefix_dir / "solve_network_benchmarks.csv"
    benchmarks = collect_benchmarks(prefix_dir)
    write_benchmarks_csv(benchmarks, csv_output)
    plot_benchmarks(benchmarks, output, include_memory=not args.no_memory)
    print(f"Collected {len(benchmarks)} benchmarks and wrote {output} and {csv_output}")


if __name__ == "__main__":
    main()

# python scripts/analysis/plot_solve_benchmarks.py DE_400