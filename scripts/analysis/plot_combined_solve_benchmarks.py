#!/usr/bin/env python3
"""Compare CSV files produced by plot_solve_benchmarks.py."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

# Select cases here. Keys become legend labels; values are directories relative
# to RESULTS_DIR (or absolute directories).
CASES = {
    "IT 200": "IT_200",
    "IT 400": "IT_400",
    "IT 900": "IT_900",
    "IT 1600": "IT_1600",
}
RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "combined_solve_benchmarks"
REFERENCE_REL_TOL = 1e-6
REFERENCE_ABS_TOL = 1e-6
FIGSIZE_COMBINED = (18, 11)
FIGSIZE_SINGLE = (18, 7)
FONT_INCREASE = 6


@dataclass(frozen=True)
class Point:
    run: str
    nodes: int | None
    seconds: float
    max_rss_mb: float
    is_reference: bool


def read_case(path: Path) -> list[Point]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run plot_solve_benchmarks.py for this case first"
        )
    with path.open(newline="") as handle:
        try:
            points = [
                Point(
                    row["run"],
                    int(row["nodes"]) if row["nodes"] else None,
                    float(row["seconds"]),
                    float(row["max_rss_mb"]),
                    row["is_reference"].lower() == "true",
                )
                for row in csv.DictReader(handle)
            ]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid benchmark data in {path}") from exc
    if not points:
        raise ValueError(f"Benchmark CSV is empty: {path}")
    return points


def get_reference(label: str, points: list[Point]) -> Point:
    references = [point for point in points if point.is_reference]
    if len(references) != 1:
        raise ValueError(
            f"Case {label!r} must have exactly one complete reference; "
            f"found {len(references)}"
        )
    return references[0]


def validate_references(cases: dict[str, list[Point]]) -> Point:
    first_label = next(iter(cases))
    reference = get_reference(first_label, cases[first_label])
    for label, points in list(cases.items())[1:]:
        candidate = get_reference(label, points)
        for metric in ("seconds", "max_rss_mb"):
            expected = getattr(reference, metric)
            actual = getattr(candidate, metric)
            if not math.isclose(
                expected,
                actual,
                rel_tol=REFERENCE_REL_TOL,
                abs_tol=REFERENCE_ABS_TOL,
            ):
                raise ValueError(
                    f"Reference {metric} differs: {first_label!r}={expected}, "
                    f"{label!r}={actual}. Only cases with the same reference "
                    "can be combined."
                )
    return reference


def clustered_points(label: str, points: list[Point]) -> list[Point]:
    runs = sorted(
        (point for point in points if not point.is_reference),
        key=lambda point: (point.nodes is None, point.nodes or 0, point.run),
    )
    missing = [point.run for point in runs if point.nodes is None]
    if not runs:
        raise ValueError(f"Case {label!r} has no clustered runs")
    if missing:
        raise ValueError(f"No node count in {label!r} for: {', '.join(missing)}")
    return runs


def draw_time(axis, cases: dict[str, list[Point]], reference: Point) -> None:
    for label, points in cases.items():
        runs = clustered_points(label, points)
        axis.plot(
            [point.nodes for point in runs],
            [point.seconds for point in runs],
            "o-",
            label=label,
        )
    axis.axhline(
        reference.seconds,
        color="black",
        linestyle="--",
        label=f"Complete reference ({reference.seconds:.0f} s)",
    )
    axis.set_yscale("log")
    axis.set_ylabel("Solve time [s]", fontweight="bold", fontsize=24)
    axis.grid(which="both", alpha=0.3)
    axis.legend(fontsize=21)


def draw_memory(axis, cases: dict[str, list[Point]], reference: Point) -> None:
    for label, points in cases.items():
        runs = [
            point
            for point in clustered_points(label, points)
            if math.isfinite(point.max_rss_mb)
        ]
        axis.plot(
            [point.nodes for point in runs],
            [point.max_rss_mb / 1024 for point in runs],
            linestyle="none",
            marker="o",
            label=label,
        )
    if math.isfinite(reference.max_rss_mb):
        memory_gib = reference.max_rss_mb / 1024
        axis.axhline(
            memory_gib,
            color="black",
            linestyle="--",
            label=f"Complete reference ({memory_gib:.1f} GiB)",
        )
    axis.set_yscale("log")
    axis.set_ylabel("Peak RSS [GiB]", fontweight="bold", fontsize=24)
    axis.grid(which="both", alpha=0.3)
    axis.legend(fontsize=21)


def save_plot(
    cases: dict[str, list[Point]], reference: Point, metrics: tuple[str, ...]
) -> Path:
    font_increase = FONT_INCREASE
    figure, axes = plt.subplots(
        len(metrics),
        1,
        figsize=FIGSIZE_COMBINED if len(metrics) == 2 else FIGSIZE_SINGLE,
        sharex=len(metrics) == 2,
        constrained_layout=True,
        squeeze=False,
    )
    for index, metric in enumerate(metrics):
        axes[index, 0].tick_params(axis="both", which="both", labelsize=22)
        if metric == "time":
            draw_time(axes[index, 0], cases, reference)
        else:
            draw_memory(axes[index, 0], cases, reference)
        if font_increase:
            axis = axes[index, 0]
            axis.tick_params(axis="both", which="both", labelsize=22 + font_increase)
            axis.yaxis.label.set_fontsize(24 + font_increase)
            for label in axis.get_legend().get_texts():
                label.set_fontsize(21 + font_increase)
    axes[-1, 0].set_xlabel("Number of spatial nodes", fontweight="bold", fontsize=24 + font_increase)
    figure.suptitle("Network solution benchmarks", fontweight="bold", fontsize=28 + font_increase)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"combined_solve_benchmarks_{'_and_'.join(metrics)}.png"
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def main() -> None:
    if not CASES:
        raise ValueError("Select at least one case in CASES")
    cases = {}
    for label, case in CASES.items():
        case_dir = Path(case) if Path(case).is_absolute() else RESULTS_DIR / case
        cases[label] = read_case(case_dir / "solve_network_benchmarks.csv")
    reference = validate_references(cases)
    outputs = [
        save_plot(cases, reference, ("time", "memory")),
        save_plot(cases, reference, ("time",)),
        save_plot(cases, reference, ("memory",)),
    ]
    print("Wrote " + ", ".join(map(str, outputs)))


if __name__ == "__main__":
    main()
