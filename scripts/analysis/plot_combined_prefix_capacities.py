#!/usr/bin/env python3
"""Combine existing capacity CSVs with and without AC lines in two panels."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from plot_prefix_optimal_capacities import (
    PREFERRED_ORDER,
    TRANSMISSION_ORDER,
    technology_colors,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", nargs="?", default="IT_400")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    directory = root / "results" / args.prefix
    tables = [pd.read_csv(directory / filename, index_col=0) for filename in
              ("optimal_capacities_without_lines.csv", "optimal_capacities.csv")]
    if not tables[0].index.equals(tables[1].index):
        raise ValueError("The two capacity tables must contain the same networks")
    tables = [table.loc[:, table.abs().max() >= 0.1] for table in tables]
    columns = tables[0].columns.union(tables[1].columns, sort=False)
    non_transmission = columns.difference(TRANSMISSION_ORDER, sort=False)
    technologies = PREFERRED_ORDER.intersection(non_transmission).append(
        non_transmission.difference(PREFERRED_ORDER, sort=False)
    ).append(TRANSMISSION_ORDER.intersection(columns)).tolist()
    colors = technology_colors(technologies, root / "config/plotting.default.yaml")
    fig, axes = plt.subplots(2, 1, figsize=(19, 14), sharex=True)
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.25, hspace=0.17)
    x = np.arange(len(tables[0]))
    for ax, table, title in zip(axes, tables, [
        "(a) Optimal capacities without AC transmission lines",
        "(b) Optimal capacities including AC transmission lines",
    ]):
        bottom = np.zeros(len(table))
        totals = table.sum(axis=1).to_numpy()
        for tech in technologies:
            if tech not in table:
                continue
            values = table[tech].to_numpy()
            ax.bar(x, values, 0.78, bottom=bottom, color=colors[tech], linewidth=0)
            for i, value in enumerate(values):
                if value >= 5 and totals[i] and value / totals[i] >= 0.05:
                    ax.text(x[i], bottom[i] + value / 2, f"{value:.0f}", ha="center", va="center", fontsize=19, fontweight="bold")
            bottom += values
        ax.axvline(0.5, color="tab:red", linewidth=1.5)
        ax.set_ylim(0, max(bottom.max() * 1.08, 1))
        ax.set_title(title, loc="left", fontsize=24, fontweight="bold", pad=15)
        ax.set_ylabel("Optimal capacity [GW]", fontsize=22, fontweight="bold")
        ax.tick_params(axis="y", labelsize=18)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    labels = ["reference" if name == "complete" else str(name).removeprefix("gtf-") for name in tables[0].index]
    axes[-1].set_xticks(x, labels, rotation=45, ha="right", fontsize=19, fontweight="bold")
    handles = [Patch(facecolor=colors[tech], label=tech) for tech in technologies]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.115), ncol=4, frameon=False, fontsize=17, columnspacing=1.5, labelspacing=0.7)
    for extension in ("png", "pdf"):
        path = directory / f"optimal_capacities_combined.{extension}"
        fig.savefig(path, dpi=250, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
