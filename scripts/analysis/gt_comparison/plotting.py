# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Plotting utilities for geo-temporal comparison analysis.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sanitize_filename(text: str) -> str:
    """Return a filesystem-friendly string."""
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
    )


def get_plot_column(metric: str, value_kind: str) -> str:
    """Map a metric and value kind to a dataframe column."""
    if value_kind == "absolute":
        return metric
    if value_kind == "delta":
        return f"{metric}_delta"
    if value_kind == "relative_delta":
        return f"{metric}_relative_delta"

    raise ValueError(f"Unsupported value_kind: {value_kind}")

def export_plot_data(
    plot_df: pd.DataFrame,
    plots_dir: Path,
    filename: str,
) -> None:
    """Export the exact dataframe used for a plot."""
    data_dir = plots_dir / "plot_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(data_dir / f"{filename}.csv", index=False)


def get_ylabel(metric: str, value_kind: str) -> str:
    """Build a readable y-axis label."""
    if value_kind == "absolute":
        return metric.replace("_", " ")
    if value_kind == "delta":
        return f"{metric.replace('_', ' ')} delta vs complete"
    if value_kind == "relative_delta":
        return f"{metric.replace('_', ' ')} relative delta vs complete"

    return metric.replace("_", " ")


def save_figure(fig, path_without_suffix: Path, config: dict[str, Any]) -> None:
    """Save a figure in all configured formats."""
    formats = config.get("plots", {}).get("formats", ["png"])
    dpi = int(config.get("plots", {}).get("dpi", 300))

    for fmt in formats:
        fig.savefig(path_without_suffix.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


def plot_metric_2d_scans(
    df: pd.DataFrame,
    metric: str,
    value_kind: str,
    plots_dir: Path,
    config: dict[str, Any],
) -> None:
    """
    Plot 2D scan curves.

    For nodes_scan:
        x = n_nodes
        color = n_days

    For days_scan:
        x = n_days
        color = n_nodes

    For budget_scan:
        x = n_nodes
        color = n_days

    Scans with fewer than two non-reference points are skipped.
    """
    column = get_plot_column(metric, value_kind)
    if column not in df.columns:
        print(f"[WARNING] Column '{column}' not found. Skipping 2D plot.")
        return

    scan_specs = [
        {
            "scan_type": "nodes_scan",
            "x_col": "n_nodes",
            "color_col": "n_days",
            "x_label": "Number of nodes",
            "color_label": "Number of days",
            "title_suffix": "spatial scan",
            "include_complete": True,
        },
        {
            "scan_type": "days_scan",
            "x_col": "n_days",
            "color_col": "n_nodes",
            "x_label": "Number of days",
            "color_label": "Number of nodes",
            "title_suffix": "temporal scan",
            "include_complete": True,
        },
        {
            "scan_type": "budget_scan",
            "x_col": "n_nodes",
            "color_col": "n_days",
            "x_label": "Number of nodes",
            "color_label": "Number of days",
            "title_suffix": "constant geo-temporal budget",
            "include_complete": False,
        },
    ]

    reference_df = df[df["scan_type"] == "complete"].copy()
    reference_value = None
    if not reference_df.empty and column in reference_df.columns:
        reference_value = float(reference_df.iloc[0][column])

    for spec in scan_specs:
        scan_type = spec["scan_type"]
        x_col = spec["x_col"]
        color_col = spec["color_col"]
        x_label = spec["x_label"]
        color_label = spec["color_label"]
        title_suffix = spec["title_suffix"]
        include_complete = spec["include_complete"]

        non_reference_df = df[df["scan_type"] == scan_type].copy()

        # Skip meaningless 2D scans with only one actual reduced point.
        if len(non_reference_df) < 2:
            print(
                f"[INFO] Skipping {scan_type} 2D plot for '{metric}' "
                f"because it has fewer than two non-reference points."
            )
            continue

        if include_complete:
            plot_df = pd.concat([reference_df, non_reference_df], ignore_index=True)
        else:
            plot_df = non_reference_df

        plot_df = plot_df.dropna(subset=[column, x_col, color_col])
        plot_df = plot_df.sort_values(x_col)

        if plot_df.empty:
            continue

        filename = sanitize_filename(f"{metric}_{value_kind}_2d_{scan_type}")

        export_plot_data(
            plot_df=plot_df,
            plots_dir=plots_dir,
            filename=filename,
        )

        fig, ax = plt.subplots(figsize=(7.6, 4.9))

        ax.plot(
            plot_df[x_col],
            plot_df[column],
            linewidth=1.4,
            alpha=0.55,
            zorder=1,
        )

        scatter = ax.scatter(
            plot_df[x_col],
            plot_df[column],
            c=plot_df[color_col],
            s=65,
            edgecolors="black",
            linewidths=0.4,
            zorder=2,
        )

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color_label, fontweight="bold")

        if scan_type == "budget_scan":
            for _, row in plot_df.iterrows():
                ax.annotate(
                    f"d={int(row['n_days'])}",
                    xy=(row[x_col], row[column]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

        ax.set_xlabel(x_label, fontweight="bold")
        ax.set_ylabel(get_ylabel(metric, value_kind), fontweight="bold")
        ax.set_title(
            f"{metric.replace('_', ' ')} - {title_suffix}",
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        if value_kind in {"delta", "relative_delta"}:
            ax.axhline(0.0, linewidth=1.0, linestyle="--")
        elif reference_value is not None and scan_type == "budget_scan":
            ax.axhline(
                reference_value,
                linewidth=1.0,
                linestyle="--",
                label="complete",
            )
            ax.legend(frameon=False)

        if value_kind == "relative_delta":
            ax.yaxis.set_major_formatter(lambda x, pos: f"{x * 100:.1f}%")

        fig.tight_layout()

        save_figure(fig, plots_dir / filename, config)
        plt.close(fig)


def plot_metric_3d(
    df: pd.DataFrame,
    metric: str,
    value_kind: str,
    plots_dir: Path,
    config: dict[str, Any],
) -> None:
    """Plot all available points in the N-D-metric space."""
    column = get_plot_column(metric, value_kind)
    if column not in df.columns:
        print(f"[WARNING] Column '{column}' not found. Skipping 3D plot.")
        return

    plot_df = df[["run", "n_nodes", "n_days", column, "scan_type"]].copy()
    plot_df = plot_df.dropna(subset=[column, "n_nodes", "n_days"])
    plot_df = plot_df.sort_values(["n_nodes", "n_days"])

    if plot_df.empty:
        return

    filename = sanitize_filename(f"{metric}_{value_kind}_3d")

    export_plot_data(
        plot_df=plot_df,
        plots_dir=plots_dir,
        filename=filename,
    )

    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        plot_df["n_nodes"],
        plot_df["n_days"],
        plot_df[column],
        c=plot_df[column],
        s=65,
        edgecolors="black",
        linewidths=0.4,
        depthshade=False,
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.12)
    cbar.set_label(get_ylabel(metric, value_kind), fontweight="bold")

    reference_df = plot_df[plot_df["scan_type"] == "complete"]
    if not reference_df.empty:
        ref = reference_df.iloc[0]
        ax.scatter(
            [ref["n_nodes"]],
            [ref["n_days"]],
            [ref[column]],
            s=120,
            marker="*",
            edgecolors="black",
            linewidths=0.7,
            depthshade=False,
            label="complete",
        )
        ax.legend(frameon=False)

    for _, row in plot_df.iterrows():
        ax.text(
            row["n_nodes"],
            row["n_days"],
            row[column],
            str(row["run"]),
            fontsize=7,
        )

    ax.set_xlabel("Number of nodes", fontweight="bold")
    ax.set_ylabel("Number of days", fontweight="bold")
    ax.set_zlabel(get_ylabel(metric, value_kind), fontweight="bold")
    ax.set_title(f"{metric.replace('_', ' ')} - 3D scan", fontweight="bold")

    # This angle makes z differences more visible for budget-scan points.
    ax.view_init(elev=24, azim=-55)

    if value_kind == "relative_delta":
        ax.zaxis.set_major_formatter(lambda x, pos: f"{x * 100:.1f}%")

    fig.tight_layout()

    save_figure(fig, plots_dir / filename, config)
    plt.close(fig)


def order_runs_for_heatmap(df: pd.DataFrame) -> list[str]:
    """Return a stable run order for heatmaps."""
    scan_order = {
        "complete": 0,
        "budget_scan": 1,
        "nodes_scan": 2,
        "days_scan": 3,
        "mixed_scan": 4,
    }

    columns = ["run", "scan_type", "n_nodes", "n_days"]
    if "total_steps" in df.columns:
        columns.append("total_steps")

    tmp = df[columns].drop_duplicates().copy()
    tmp["_scan_order"] = tmp["scan_type"].map(scan_order).fillna(99)

    sort_columns = ["_scan_order", "n_nodes", "n_days", "run"]
    tmp = tmp.sort_values(sort_columns)

    return tmp["run"].tolist()


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path_without_suffix: Path,
    config: dict[str, Any],
    value_kind: str,
) -> None:
    """Plot a basic heatmap with numeric annotations."""
    if matrix.empty:
        return

    n_rows, n_cols = matrix.shape
    fig_width = max(7.0, 0.45 * n_cols + 2.5)
    fig_height = max(4.5, 0.35 * n_rows + 2.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    values = matrix.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)

    im = ax.imshow(masked_values, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Carrier", fontweight="bold")
    ax.set_ylabel("Run", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    if value_kind == "relative_delta":
        cbar.ax.set_ylabel("Relative delta vs complete", fontweight="bold")
    elif value_kind == "delta":
        cbar.ax.set_ylabel("Delta vs complete", fontweight="bold")
    else:
        cbar.ax.set_ylabel("Value", fontweight="bold")

    for i in range(n_rows):
        for j in range(n_cols):
            value = values[i, j]
            if np.isnan(value):
                text = ""
            elif value_kind == "relative_delta":
                text = f"{value * 100:.1f}%"
            else:
                text = f"{value:.2g}"

            if text:
                ax.text(j, i, text, ha="center", va="center", fontsize=7)

    fig.tight_layout()
    save_figure(fig, output_path_without_suffix, config)
    plt.close(fig)


def filter_capacity_heatmap_df(
    capacity_df: pd.DataFrame,
    component: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Filter capacity dataframe for heatmap plotting."""
    df = capacity_df[capacity_df["component"] == component].copy()

    if component == "Generator":
        selected = config.get("plots", {}).get("capacity_heatmaps", {}).get(
            "generator_carriers", []
        )
        if selected:
            df = df[df["carrier"].isin(selected)]

    return df


def plot_capacity_heatmaps(
    capacity_df: pd.DataFrame,
    plots_dir: Path,
    config: dict[str, Any],
) -> None:
    """Create capacity heatmaps by component and carrier."""
    if capacity_df.empty:
        print("[WARNING] Capacity dataframe is empty. Skipping heatmaps.")
        return

    heatmap_cfg = config.get("plots", {}).get("capacity_heatmaps", {})
    value_kind = heatmap_cfg.get("value", "relative_delta")

    if value_kind == "absolute":
        value_col = "value"
    elif value_kind == "delta":
        value_col = "delta"
    elif value_kind == "relative_delta":
        value_col = "relative_delta"
    else:
        raise ValueError(f"Unsupported capacity heatmap value: {value_kind}")

    if value_col not in capacity_df.columns:
        print(f"[WARNING] Column '{value_col}' not found. Skipping capacity heatmaps.")
        return

    components = heatmap_cfg.get("components", sorted(capacity_df["component"].unique()))

    for component in components:
        df = filter_capacity_heatmap_df(capacity_df, component, config)

        if df.empty:
            continue

        run_order = order_runs_for_heatmap(df)

        matrix = df.pivot_table(
            index="run",
            columns="carrier",
            values=value_col,
            aggfunc="sum",
        )

        matrix = matrix.reindex(run_order)
        matrix = matrix.dropna(axis=1, how="all")

        # Drop columns that are all zero or NaN, except for absolute plots.
        if value_kind != "absolute":
            nonzero_cols = matrix.columns[
                matrix.fillna(0.0).abs().sum(axis=0) > 0.0
            ]
            matrix = matrix[nonzero_cols]

        if matrix.empty:
            continue

        title = f"{component} capacity - {value_kind.replace('_', ' ')}"
        filename = sanitize_filename(f"{component}_capacity_heatmap_{value_kind}")

        plot_heatmap(
            matrix=matrix,
            title=title,
            output_path_without_suffix=plots_dir / filename,
            config=config,
            value_kind=value_kind,
        )