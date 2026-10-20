#!/usr/bin/env python3
"""Combine the selected Italian balances and capacities with a shared legend."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from plot_prefix_energy_balance import PREFERRED_ORDER, technology_colors
from plot_prefix_optimal_capacities import TRANSMISSION_ORDER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", help="Results prefix, e.g. IT_400; otherwise use selected solutions")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / args.prefix if args.prefix else root / "results/comparison_plots/IT_selected_solutions"
    balance = pd.read_csv(output / "electricity_energy_balance.csv", index_col=0)
    capacity = pd.read_csv(output / ("optimal_capacities_without_lines.csv" if args.prefix else "optimal_capacities.csv"), index_col=0)
    if not args.prefix:
        capacity = capacity.drop(
            columns=["transmission lines", "AC transmission lines", "DC transmission links"],
            errors="ignore",
        )
    assert balance.index.equals(capacity.index), "The networks must match in both panels"
    tables = [table.loc[:, table.abs().max() >= 0.1] for table in (balance, capacity)]
    columns = tables[0].columns.union(tables[1].columns, sort=False)
    preferred = PREFERRED_ORDER.drop(["AC load", "transmission lines"])
    transmission = pd.Index(["transmission lines"]).append(TRANSMISSION_ORDER)
    technologies = preferred.intersection(columns).append(columns.difference(preferred.append(transmission).append(pd.Index(["AC load"])), sort=False)).append(transmission.intersection(columns)).append(pd.Index(["AC load"]).intersection(columns)).tolist()
    colors = technology_colors(technologies, root / "config/plotting.default.yaml")
    colors["AC transmission lines"] = "#8c564b"
    fig, axes = plt.subplots(2, 1, figsize=(19 if args.prefix else 12, 16), sharex=True)
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.09 if args.prefix else 0.14, right=0.98, top=0.96, bottom=0.21 if args.prefix else 0.14, hspace=0.16)
    x = np.arange(len(balance))
    for ax, table, title, ylabel, minimum in zip(
        axes, tables,
        ["(a) Electricity energy balance", "(b) Optimal capacities without lines"],
        ["Energy balance [TWh/a]", "Optimal capacity [GW]"],
        [10, 5],
    ):
        positive_bottom = np.zeros(len(table))
        negative_bottom = np.zeros(len(table))
        positive_total = table.clip(lower=0).sum(axis=1).to_numpy()
        negative_total = -table.clip(upper=0).sum(axis=1).to_numpy()
        for tech in technologies:
            if tech not in table:
                continue
            values = table[tech].to_numpy()
            positive, negative = np.maximum(values, 0), np.minimum(values, 0)
            ax.bar(x, positive, 0.72, bottom=positive_bottom, color=colors[tech], linewidth=0)
            ax.bar(x, negative, 0.72, bottom=negative_bottom, color=colors[tech], hatch="//", edgecolor="black", linewidth=0.25)
            for i, value in enumerate(values):
                total = positive_total[i] if value > 0 else negative_total[i]
                if abs(value) >= minimum and total and abs(value) / total >= 0.05:
                    bottom = positive_bottom[i] if value > 0 else negative_bottom[i]
                    ax.text(x[i], bottom + value / 2, f"{abs(value):.0f}", ha="center", va="center", fontsize=19, fontweight="bold")
            positive_bottom += positive
            negative_bottom += negative
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0.5, color="tab:red", linewidth=1.3)
        ax.set_title(title, loc="left", fontsize=22, fontweight="bold", pad=14)
        ax.set_ylabel(ylabel, fontsize=20, fontweight="bold")
        ax.tick_params(axis="both", labelsize=17)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        span = max(positive_bottom.max() - negative_bottom.min(), 1)
        ax.set_ylim(negative_bottom.min() - (0.04 * span if negative_bottom.min() < 0 else 0), positive_bottom.max() + 0.08 * span)
    axes[-1].set_xticks(x, ["reference" if label == "complete" else str(label).removeprefix("gtf-") for label in balance.index], fontsize=16, fontweight="bold", rotation=45 if args.prefix else 0, ha="right" if args.prefix else "center")
    handles = [Patch(facecolor=colors[tech], label=tech) for tech in technologies]
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="//", label="Consumption\n(panel a)"))
    # Matplotlib fills legends by column; reorder so technologies read by row.
    if args.prefix:
        # Fill three rows from left to right, padding the final row if needed.
        legend_columns = (len(handles) + 2) // 3
        padded = handles + [Patch(facecolor="none", edgecolor="none", label=" ")
                            for _ in range(3 * legend_columns - len(handles))]
        handles = [padded[row * legend_columns + column]
                   for column in range(legend_columns) for row in range(3)]
    elif len(handles) == 12:
        handles = [handle for column in zip(handles[:4], handles[4:8], handles[8:]) for handle in column]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.132 if args.prefix else 0.078), ncol=legend_columns if args.prefix else 4, frameon=False, fontsize=23 if args.prefix else 17, columnspacing=0.9, handlelength=1.3, handletextpad=0.5, labelspacing=0.65)
    for extension in ("png", "pdf"):
        path = output / f"energy_balance_and_optimal_capacities.{extension}"
        fig.savefig(path, dpi=250, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
