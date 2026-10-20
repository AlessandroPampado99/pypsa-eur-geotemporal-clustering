#!/usr/bin/env python3
"""Compare the Italian reference with four selected geo-temporal solutions."""

from pathlib import Path

import pandas as pd

from plot_prefix_energy_balance import electricity_balance, plot_balance, solved_network
from plot_prefix_optimal_capacities import optimal_capacities, plot_capacities


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/comparison_plots/IT_selected_solutions"
    output.mkdir(parents=True, exist_ok=True)
    plotting = root / "config/plotting.default.yaml"
    selections = [
        ("complete", "IT_400/complete"),
        ("k=200\nn18-d11", "IT_200/gtf-n18-d11"),
        ("k=400\nn20-d20", "IT_400/gtf-n20-d20"),
        ("k=900\nn20-d40", "IT_900/gtf-n20-d40"),
        ("k=1600\nn25-d64", "IT_1600/gtf-n25-d64"),
    ]
    balances, capacities, sources = {}, {}, []
    for label, run in selections:
        path = solved_network(root / "results" / run)
        print(f"Reading {label!r}: {path}", flush=True)
        balances[label] = electricity_balance(path)
        capacities[label] = optimal_capacities(path, include_lines=True)
        sources.append({"label": label, "network": str(path.relative_to(root))})

    balance = pd.DataFrame(balances).fillna(0.0).T
    capacity = pd.DataFrame(capacities).fillna(0.0).T
    balance.to_csv(output / "electricity_energy_balance.csv")
    plot_balance(balance, output / "electricity_energy_balance.png", plotting, 0.1, "Italy — selected solutions", segment_fontsize=19)
    for include_lines, suffix in [(True, ""), (False, "_without_lines")]:
        table = capacity if include_lines else capacity.drop(columns="AC transmission lines", errors="ignore")
        table.to_csv(output / f"optimal_capacities{suffix}.csv")
        plot_capacities(table, output / f"optimal_capacities{suffix}.png", plotting, 0.1, "Italy — selected solutions", include_lines, segment_fontsize=19)
    pd.DataFrame(sources).to_csv(output / "source_networks.csv", index=False)
    print(f"Wrote plots and data to {output}")


if __name__ == "__main__":
    main()
