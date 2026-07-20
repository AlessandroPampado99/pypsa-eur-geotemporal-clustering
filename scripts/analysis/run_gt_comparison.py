#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Run geo-temporal clustering comparison analysis.

This script:
1. reads the analysis config;
2. discovers or loads the selected runs;
3. loads each PyPSA network;
4. computes aggregate metrics;
5. computes deltas against the complete/reference case;
6. writes CSV/XLSX outputs;
7. optionally creates 2D, 3D, and capacity heatmap plots.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gt_comparison.config import load_config
from gt_comparison.io import build_run_table, load_network, add_clustering_objectives
from gt_comparison.metrics import collect_metrics_for_network
from gt_comparison.postprocess import add_reference_deltas
from gt_comparison.plotting import (
    plot_metric_2d_scans,
    plot_metric_3d,
    plot_capacity_heatmaps,
    plot_budget_combined_relative_profiles,
    plot_optimization_vs_clustering_objective,
    plot_optimization_without_line_cost_vs_clustering_objective,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare geo-temporal clustering runs against a complete reference network."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(HERE / "config_gt_comparison.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Compute and save metrics, but skip plot generation.",
    )
    return parser.parse_args()


def write_outputs(
    output_dir: Path,
    metrics_summary: pd.DataFrame,
    generation_by_carrier: pd.DataFrame,
    capacity_by_component_carrier: pd.DataFrame,
    missing_carriers_report: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    line_cost_columns = [
        "run",
        "n_nodes",
        "n_days",
        "total_steps",
        "scan_type",
        "objective",
        "line_capex",
        "line_opex",
        "line_total_cost",
        "line_cost_percentage_of_total",
        "objective_without_line_cost",
    ]
    available_line_cost_columns = [
        column for column in line_cost_columns if column in metrics_summary.columns
    ]
    line_cost_summary = metrics_summary.loc[:, available_line_cost_columns].copy()

    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    line_cost_summary.to_csv(output_dir / "line_cost_summary.csv", index=False)
    generation_by_carrier.to_csv(output_dir / "generation_by_carrier.csv", index=False)
    capacity_by_component_carrier.to_csv(
        output_dir / "capacity_by_component_carrier.csv", index=False
    )
    missing_carriers_report.to_csv(output_dir / "missing_carriers_report.csv", index=False)

    xlsx_path = output_dir / "gt_comparison_summary.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        metrics_summary.to_excel(writer, sheet_name="metrics_summary", index=False)
        line_cost_summary.to_excel(writer, sheet_name="line_cost_summary", index=False)
        generation_by_carrier.to_excel(writer, sheet_name="generation_by_carrier", index=False)
        capacity_by_component_carrier.to_excel(
            writer, sheet_name="capacity_by_component_carrier", index=False
        )
        missing_carriers_report.to_excel(
            writer, sheet_name="missing_carriers_report", index=False
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    root_dir = Path(config["paths"]["root_dir"]).expanduser().resolve()
    output_dir = root_dir / config["paths"]["output_dir"]

    run_table = build_run_table(config)

    all_summary_rows = []
    all_generation_rows = []
    all_capacity_rows = []
    all_missing_rows = []

    for _, run_row in run_table.iterrows():
        run_name = run_row["run"]
        network_path = Path(run_row["network_path"])

        if not network_path.exists():
            print(f"[WARNING] Missing network for run '{run_name}': {network_path}")
            continue

        print(f"[INFO] Loading run '{run_name}': {network_path}")
        n = load_network(network_path)

        metrics = collect_metrics_for_network(
            n=n,
            run_info=run_row.to_dict(),
            config=config,
        )

        all_summary_rows.append(metrics["summary"])
        all_generation_rows.extend(metrics["generation_by_carrier"])
        all_capacity_rows.extend(metrics["capacity_by_component_carrier"])
        all_missing_rows.extend(metrics["missing_carriers"])

    if not all_summary_rows:
        raise RuntimeError("No valid runs were processed. Check paths and config.")

    metrics_summary = pd.DataFrame(all_summary_rows)
    generation_by_carrier = pd.DataFrame(all_generation_rows)
    capacity_by_component_carrier = pd.DataFrame(all_capacity_rows)
    missing_carriers_report = pd.DataFrame(all_missing_rows)

    reference_run = config["reference"]["run"]

    metrics_summary = add_reference_deltas(
        df=metrics_summary,
        reference_run=reference_run,
        id_columns=["run", "n_nodes", "n_days", "total_steps", "scan_type", "network_path"],
    )

    generation_by_carrier = add_reference_deltas(
        df=generation_by_carrier,
        reference_run=reference_run,
        id_columns=["run", "n_nodes", "n_days", "total_steps", "scan_type", "component", "carrier"],
        value_column="value",
        group_columns=["component", "carrier"],
    )

    capacity_by_component_carrier = add_reference_deltas(
        df=capacity_by_component_carrier,
        reference_run=reference_run,
        id_columns=[
            "run",
            "n_nodes",
            "n_days",
            "total_steps",
            "scan_type",
            "component",
            "carrier",
            "attribute",
        ],
        value_column="value",
        group_columns=["component", "carrier", "attribute"],
    )

    metrics_summary = add_clustering_objectives(
        df=metrics_summary,
        config=config,
    )

    write_outputs(
        output_dir=output_dir,
        metrics_summary=metrics_summary,
        generation_by_carrier=generation_by_carrier,
        capacity_by_component_carrier=capacity_by_component_carrier,
        missing_carriers_report=missing_carriers_report,
    )

    plots_enabled = bool(config.get("plots", {}).get("enabled", True))
    if args.skip_plots:
        plots_enabled = False

    if plots_enabled:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_config = config.get("plots", {})
        plot_metrics = plot_config.get("metrics", [])
        plot_values = plot_config.get("plot_values", ["absolute", "relative_delta"])

        if plot_config.get("make_2d", True):
            for metric in plot_metrics:
                if metric not in metrics_summary.columns:
                    print(f"[WARNING] Metric '{metric}' not found in metrics_summary. Skipping 2D.")
                    continue

                for value_kind in plot_values:
                    plot_metric_2d_scans(
                        df=metrics_summary,
                        metric=metric,
                        value_kind=value_kind,
                        plots_dir=plots_dir,
                        config=config,
                    )

        if plot_config.get("make_3d", True):
            for metric in plot_metrics:
                if metric not in metrics_summary.columns:
                    print(f"[WARNING] Metric '{metric}' not found in metrics_summary. Skipping 3D.")
                    continue

                for value_kind in plot_values:
                    plot_metric_3d(
                        df=metrics_summary,
                        metric=metric,
                        value_kind=value_kind,
                        plots_dir=plots_dir,
                        config=config,
                    )
                    
        if plot_config.get("make_budget_combined", True):
            plot_budget_combined_relative_profiles(
                df=metrics_summary,
                plots_dir=plots_dir,
                config=config,
            )

        if plot_config.get("make_objective_comparison", True):
            plot_optimization_vs_clustering_objective(
                df=metrics_summary,
                plots_dir=plots_dir,
                config=config,
            )

        if plot_config.get("make_objective_without_line_cost_comparison", True):
            plot_optimization_without_line_cost_vs_clustering_objective(
                df=metrics_summary,
                plots_dir=plots_dir,
                config=config,
            )

        if plot_config.get("capacity_heatmaps", {}).get("enabled", True):
            plot_capacity_heatmaps(
                capacity_df=capacity_by_component_carrier,
                plots_dir=plots_dir,
                config=config,
            )

    print(f"[DONE] Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()

# python scripts/analysis/run_gt_comparison.py \
#  --config scripts/analysis/config_gt_comparison.yaml