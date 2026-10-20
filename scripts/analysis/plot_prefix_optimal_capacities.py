#!/usr/bin/env python3
"""Plot installed optimal capacities for solved networks below a results prefix."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import yaml
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._helpers import rename_techs  # noqa: E402

NODE_COUNT_PATTERN = re.compile(r"(?:^|-)n(\d+)(?:-|\Z)")
SEGMENT_LABEL_MIN_GW = 5.0
SEGMENT_LABEL_MIN_SHARE = 5.0
COUNTRY_NAMES = {"DE": "Germany", "ES": "Spain", "IT": "Italy"}
PREFERRED_ORDER = pd.Index(
    [
        "hydroelectricity",
        "hydro reservoir",
        "run of river",
        "pumped hydro storage",
        "solid biomass",
        "biogas",
        "onshore wind",
        "offshore wind",
        "offshore wind (AC)",
        "offshore wind (DC)",
        "solar PV",
        "solar",
        "battery storage",
        "hydrogen storage",
        "OCGT",
        "CCGT",
        "coal",
        "lignite",
        "nuclear",
        "oil",
    ]
)
TRANSMISSION_ORDER = pd.Index(
    [
        "DC transmission links",
        "AC transmission lines",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot optimal capacities for every solved run in a prefix."
    )
    parser.add_argument("prefix", help="Prefix name (for example DE_400) or path.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        help="Base output image (default: PREFIX/optimal_capacities.png).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Hide technologies below this capacity in every run [GW].",
    )
    parser.add_argument(
        "--plotting-config",
        type=Path,
        default=Path("config/plotting.default.yaml"),
    )
    return parser.parse_args()


def run_order(name: str) -> tuple[int, int, str]:
    if name == "complete":
        return (-1, 0, name)
    match = NODE_COUNT_PATTERN.search(name)
    return (0, int(match.group(1)), name) if match else (1, 0, name)


def plot_subtitle(prefix_name: str) -> str:
    match = re.search(r"(?:^|_)([A-Za-z]{2})_(\d+)(?:_|\Z)", prefix_name)
    if not match:
        return prefix_name
    country_code, nodes = match.groups()
    country = COUNTRY_NAMES.get(country_code.upper(), country_code.upper())
    return f"{country}, {nodes} geo-temporal nodes"


def solved_network(run_dir: Path) -> Path:
    candidates = list((run_dir / "networks").glob("*.nc"))
    if not candidates:
        raise FileNotFoundError(f"No network found for {run_dir.name}")
    clustered = [path for path in candidates if path.stem.endswith("_gt")]
    if run_dir.name != "complete" and len(clustered) == 1:
        return clustered[0]
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(path.name for path in candidates)
    raise ValueError(f"Cannot identify solved network for {run_dir.name}: {names}")


def clean_carrier(carrier: object) -> str:
    if pd.isna(carrier) or not str(carrier).strip():
        return "unknown"
    return rename_techs(str(carrier))


def optimal_capacities(path: Path, include_lines: bool) -> pd.Series:
    """Return generator, link, and optionally line capacities in GW."""
    network = pypsa.Network(path)
    pieces = []

    if not network.generators.empty:
        generators = network.generators.groupby("carrier").p_nom_opt.sum() / 1e3
        generators.index = generators.index.map(clean_carrier)
        pieces.append(generators)

    if not network.links.empty:
        links = network.links.groupby("carrier").p_nom_opt.sum() / 1e3
        links.index = links.index.map(clean_carrier)
        links = links.rename(index={"DC": "DC transmission links"})
        pieces.append(links)

    if include_lines and not network.lines.empty:
        pieces.append(
            pd.Series({"AC transmission lines": network.lines.s_nom_opt.sum() / 1e3})
        )

    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).groupby(level=0).sum()


def collect_capacities(prefix_dir: Path, include_lines: bool) -> pd.DataFrame:
    if not prefix_dir.is_dir():
        raise FileNotFoundError(f"Results prefix does not exist: {prefix_dir}")
    run_dirs = sorted(
        (
            path
            for path in prefix_dir.iterdir()
            if path.is_dir() and (path / "networks").is_dir()
        ),
        key=lambda path: run_order(path.name),
    )
    capacities = {}
    for run_dir in run_dirs:
        network = solved_network(run_dir)
        print(f"Reading {run_dir.name}: {network.name}")
        capacities[run_dir.name] = optimal_capacities(network, include_lines)
    if not capacities:
        raise ValueError(f"No solved networks found below {prefix_dir}")
    return pd.DataFrame(capacities).fillna(0.0).T


def technology_colors(
    technologies: list[str], plotting_config: Path
) -> dict[str, object]:
    with plotting_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    configured = config.get("plotting", {}).get("tech_colors", {})
    fallback = plt.get_cmap("tab20")
    return {
        technology: configured.get(technology, fallback(i % fallback.N))
        for i, technology in enumerate(technologies)
    }


def plot_capacities(
    table: pd.DataFrame,
    output: Path,
    plotting_config: Path,
    threshold: float,
    prefix_name: str,
    include_lines: bool,
    segment_fontsize: float = 13,
) -> None:
    table = table.loc[:, table.abs().max(axis=0) >= threshold]
    if table.empty:
        raise ValueError(f"No technology exceeds the {threshold:g} GW threshold")
    # Add dynamically discovered carriers before transmission so both DC links
    # and AC lines are always drawn at the top of the stack. AC is topmost.
    non_transmission = table.columns.difference(TRANSMISSION_ORDER, sort=False)
    technologies = (
        PREFERRED_ORDER.intersection(non_transmission)
        .append(non_transmission.difference(PREFERRED_ORDER, sort=False))
        .append(TRANSMISSION_ORDER.intersection(table.columns))
        .tolist()
    )
    table = table.loc[:, technologies]
    colors = technology_colors(technologies, plotting_config)
    x = np.arange(len(table))
    bottom = np.zeros(len(table))
    totals = table.sum(axis=1).to_numpy()

    figure, axis = plt.subplots(figsize=(max(19, 1.0 * len(table)), 8))
    for technology in technologies:
        values = table[technology].to_numpy(dtype=float)
        axis.bar(x, values, 0.78, bottom=bottom, color=colors[technology], linewidth=0)
        for i, value in enumerate(values):
            share = 100 * value / totals[i] if totals[i] else 0
            if value >= SEGMENT_LABEL_MIN_GW and share >= SEGMENT_LABEL_MIN_SHARE:
                axis.text(
                    x[i],
                    bottom[i] + value / 2,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=segment_fontsize,
                    fontweight="bold",
                )
        bottom += values

    if table.index[0] == "complete" and len(table) > 1:
        axis.axvline(0.5, color="tab:red", linewidth=1.5)
        axis.text(
            0,
            1.01,
            "Reference",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            color="tab:red",
            fontsize=21,
            fontweight="bold",
        )
    axis.set_xticks(x, [str(name).removeprefix("gtf-") for name in table.index], rotation=45, ha="right", fontweight="bold", fontsize=21)
    axis.tick_params(axis="y", labelsize=15)
    axis.set_ylabel("Optimal capacity [GW]", fontweight="bold", fontsize=22)
    axis.set_xlabel("Network", fontweight="bold", fontsize=22)
    title = (
        "Optimal capacities (Generators, links and lines)"
        if include_lines
        else "Optimal capacities"
    )
    axis.set_title(
        f"{title} — {plot_subtitle(prefix_name)}",
        fontweight="bold", fontsize=26, pad=28,
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(
        [Patch(facecolor=colors[tech], edgecolor="none") for tech in technologies],
        technologies,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
        fontsize=21,
    )
    axis.set_ylim(0, max(bottom.max() * 1.08, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    path = Path(args.prefix)
    prefix_dir = path if path.is_dir() else args.results_dir / args.prefix
    output = args.output or prefix_dir / "optimal_capacities.png"
    without_lines_output = output.with_name(
        f"{output.stem}_without_lines{output.suffix}"
    )

    with_lines = collect_capacities(prefix_dir, include_lines=True)
    without_lines = with_lines.drop(columns="AC transmission lines", errors="ignore")
    with_lines.to_csv(output.with_suffix(".csv"))
    without_lines.to_csv(without_lines_output.with_suffix(".csv"))
    plot_capacities(
        with_lines,
        output,
        args.plotting_config,
        args.threshold,
        prefix_dir.name,
        include_lines=True,
    )
    plot_capacities(
        without_lines,
        without_lines_output,
        args.plotting_config,
        args.threshold,
        prefix_dir.name,
        include_lines=False,
    )
    print(
        f"Wrote {output}, {output.with_suffix('.csv')}, "
        f"{without_lines_output}, and {without_lines_output.with_suffix('.csv')}"
    )


if __name__ == "__main__":
    main()
