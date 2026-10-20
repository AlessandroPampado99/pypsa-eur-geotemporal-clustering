#!/usr/bin/env python3
"""Plot electricity energy balances for solved networks below a results prefix."""

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
SEGMENT_LABEL_MIN_TWH = 10.0
SEGMENT_LABEL_MIN_SHARE = 5.0
COUNTRY_NAMES = {"DE": "Germany", "ES": "Spain", "IT": "Italy"}
PREFERRED_ORDER = pd.Index(
    [
        "transmission lines",
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
        "AC load",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked electricity balances for every run in a prefix."
    )
    parser.add_argument("prefix", help="Prefix name (for example DE_400) or path.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image (default: PREFIX/electricity_energy_balance.png).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Hide technologies below this absolute energy in every run [TWh].",
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
    if match:
        return (0, int(match.group(1)), name)
    return (1, 0, name)


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


def electricity_balance(path: Path) -> pd.Series:
    network = pypsa.Network(path)
    balance = network.statistics.energy_balance(
        bus_carrier="AC", groupby="carrier", nice_names=False
    )
    balance = balance.groupby(level="carrier").sum() / 1e6
    balance.index = [
        (
            "AC load"
            if pd.isna(carrier) or str(carrier).strip() in {"", "-"}
            else rename_techs(str(carrier))
        )
        for carrier in balance.index
    ]
    return balance.groupby(level=0).sum()


def collect_balances(prefix_dir: Path) -> pd.DataFrame:
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
    balances = {}
    for run_dir in run_dirs:
        network = solved_network(run_dir)
        print(f"Reading {run_dir.name}: {network.name}")
        balances[run_dir.name] = electricity_balance(network)
    if not balances:
        raise ValueError(f"No solved networks found below {prefix_dir}")
    return pd.DataFrame(balances).fillna(0.0).T


def technology_colors(
    technologies: list[str], plotting_config: Path
) -> dict[str, object]:
    with plotting_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    configured = config.get("plotting", {}).get("tech_colors", {})
    fallback = plt.get_cmap("tab20")
    return {
        technology: (
            "#e57373"
            if technology == "AC load"
            else configured.get(technology, fallback(i % fallback.N))
        )
        for i, technology in enumerate(technologies)
    }


def plot_balance(
    table: pd.DataFrame,
    output: Path,
    plotting_config: Path,
    threshold: float,
    prefix_name: str,
    segment_fontsize: float = 13,
) -> None:
    keep = table.abs().max(axis=0) >= threshold
    table = table.loc[:, keep]
    if table.empty:
        raise ValueError(f"No technology exceeds the {threshold:g} TWh threshold")
    technologies = (
        PREFERRED_ORDER.intersection(table.columns)
        .append(table.columns.difference(PREFERRED_ORDER, sort=False))
        .tolist()
    )
    table = table.loc[:, technologies]
    colors = technology_colors(technologies, plotting_config)
    x = np.arange(len(table))
    positive_bottom = np.zeros(len(table))
    negative_bottom = np.zeros(len(table))
    positive_total = table.clip(lower=0).sum(axis=1).to_numpy()
    negative_total = -table.clip(upper=0).sum(axis=1).to_numpy()

    figure, axis = plt.subplots(figsize=(max(19, 1.0 * len(table)), 8))
    for technology in technologies:
        values = table[technology].to_numpy(dtype=float)
        positive = np.clip(values, 0, None)
        negative = np.clip(values, None, 0)
        axis.bar(
            x,
            positive,
            0.78,
            bottom=positive_bottom,
            color=colors[technology],
            linewidth=0,
        )
        axis.bar(
            x,
            negative,
            0.78,
            bottom=negative_bottom,
            color=colors[technology],
            edgecolor="black",
            linewidth=0.25,
            hatch="//",
        )
        for i, value in enumerate(values):
            side_total = positive_total[i] if value > 0 else negative_total[i]
            share = 100 * abs(value) / side_total if side_total else 0
            if abs(value) < SEGMENT_LABEL_MIN_TWH or share < SEGMENT_LABEL_MIN_SHARE:
                continue
            bottom = positive_bottom[i] if value > 0 else negative_bottom[i]
            axis.text(
                x[i],
                bottom + value / 2,
                f"{abs(value):.0f}",
                ha="center",
                va="center",
                fontsize=segment_fontsize,
                rotation=0,
                fontweight="bold",
            )
        positive_bottom += positive
        negative_bottom += negative

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
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, [str(name).removeprefix("gtf-") for name in table.index], rotation=45, ha="right", fontweight="bold", fontsize=21)
    axis.tick_params(axis="y", labelsize=15)
    axis.set_ylabel("Electricity energy balance [TWh/a]", fontweight="bold", fontsize=22)
    axis.set_xlabel("Network", fontweight="bold", fontsize=22)
    axis.set_title(
        f"Electricity energy balance — {plot_subtitle(prefix_name)}",
        fontweight="bold", fontsize=26, pad=28,
    )
    for label in axis.get_yticklabels():
        label.set_fontweight("bold")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)

    supply = [tech for tech in technologies if (table[tech] > 0).any()]
    consumption = [tech for tech in technologies if (table[tech] < 0).any()]
    heading = Patch(facecolor="none", edgecolor="none")
    handles = [heading] + [
        Patch(facecolor=colors[tech], edgecolor="none") for tech in supply
    ]
    labels = ["SUPPLY", *supply]
    handles += [heading] + [
        Patch(facecolor=colors[tech], edgecolor="black", hatch="//")
        for tech in consumption
    ]
    labels += ["CONSUMPTION", *consumption]
    legend = axis.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=21
    )
    for text in legend.get_texts():
        if text.get_text() in {"SUPPLY", "CONSUMPTION"}:
            text.set_fontweight("bold")

    span = max(positive_bottom.max() - negative_bottom.min(), 1.0)
    axis.set_ylim(
        negative_bottom.min() - 0.04 * span,
        positive_bottom.max() + 0.1 * span,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    path = Path(args.prefix)
    prefix_dir = path if path.is_dir() else args.results_dir / args.prefix
    output = args.output or prefix_dir / "electricity_energy_balance.png"
    table = collect_balances(prefix_dir)
    table.to_csv(output.with_suffix(".csv"))
    plot_balance(table, output, args.plotting_config, args.threshold, prefix_dir.name)
    print(f"Wrote {output} and {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
