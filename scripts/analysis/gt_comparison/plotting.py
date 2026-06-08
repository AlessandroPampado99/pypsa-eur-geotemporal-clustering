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
        secondary bottom x-axis = n_days
        no colorbar and no point annotations
    """
    column = get_plot_column(metric, value_kind)
    if column not in df.columns:
        print(f"[WARNING] Column '{column}' not found. Skipping 2D plot.")
        return

    reference_df = df[df["scan_type"] == "complete"].copy()
    reference_value = None
    if not reference_df.empty and column in reference_df.columns:
        reference_value = float(reference_df.iloc[0][column])

    scan_specs = [
        {
            "scan_type": "nodes_scan",
            "x_col": "n_nodes",
            "color_col": "n_days",
            "x_label": "Number of nodes",
            "color_label": "Number of days",
            "title_suffix": "spatial scan",
            "include_complete": True,
            "use_secondary_days_axis": False,
        },
        {
            "scan_type": "days_scan",
            "x_col": "n_days",
            "color_col": "n_nodes",
            "x_label": "Number of days",
            "color_label": "Number of nodes",
            "title_suffix": "temporal scan",
            "include_complete": True,
            "use_secondary_days_axis": False,
        },
        {
            "scan_type": "budget_scan",
            "x_col": "n_nodes",
            "color_col": None,
            "x_label": "Number of nodes",
            "color_label": None,
            "title_suffix": "constant geo-temporal budget",
            "include_complete": False,
            "use_secondary_days_axis": True,
        },
    ]

    created_budget_plot = False

    for spec in scan_specs:
        scan_type = spec["scan_type"]
        x_col = spec["x_col"]
        color_col = spec["color_col"]
        x_label = spec["x_label"]
        color_label = spec["color_label"]
        title_suffix = spec["title_suffix"]
        include_complete = spec["include_complete"]
        use_secondary_days_axis = spec["use_secondary_days_axis"]

        non_reference_df = df[df["scan_type"] == scan_type].copy()

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

        filename_suffix = scan_type

        _plot_single_2d_scan(
            plot_df=plot_df,
            metric=metric,
            column=column,
            value_kind=value_kind,
            x_col=x_col,
            color_col=color_col,
            x_label=x_label,
            color_label=color_label,
            title_suffix=title_suffix,
            filename_suffix=filename_suffix,
            plots_dir=plots_dir,
            config=config,
            reference_value=reference_value,
            use_secondary_days_axis=use_secondary_days_axis,
        )

        if scan_type == "budget_scan":
            created_budget_plot = True

    if not created_budget_plot:
        reduced_df = df[df["scan_type"] != "complete"].copy()

        if len(reduced_df) >= 2:
            print(
                f"[INFO] No budget_scan plot was created for '{metric}'. "
                "Creating fallback reduced_scan plot."
            )

            _plot_single_2d_scan(
                plot_df=reduced_df,
                metric=metric,
                column=column,
                value_kind=value_kind,
                x_col="n_nodes",
                color_col=None,
                x_label="Number of nodes",
                color_label=None,
                title_suffix="all reduced runs",
                filename_suffix="reduced_scan",
                plots_dir=plots_dir,
                config=config,
                reference_value=reference_value,
                use_secondary_days_axis=True,
            )

def _plot_single_2d_scan(
    plot_df: pd.DataFrame,
    metric: str,
    column: str,
    value_kind: str,
    x_col: str,
    color_col: str | None,
    x_label: str,
    color_label: str | None,
    title_suffix: str,
    filename_suffix: str,
    plots_dir: Path,
    config: dict[str, Any],
    reference_value: float | None = None,
    use_secondary_days_axis: bool = False,
) -> None:
    """Plot one 2D scan."""
    required_columns = [column, x_col]
    if color_col is not None:
        required_columns.append(color_col)

    plot_df = plot_df.dropna(subset=required_columns).copy()
    plot_df = plot_df.sort_values(x_col)

    if plot_df.empty:
        return

    filename = sanitize_filename(f"{metric}_{value_kind}_2d_{filename_suffix}")

    export_plot_data(
        plot_df=plot_df,
        plots_dir=plots_dir,
        filename=filename,
    )

    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    ax.plot(
        plot_df[x_col],
        plot_df[column],
        linewidth=1.6,
        alpha=0.75,
        zorder=1,
    )

    if color_col is None:
        ax.scatter(
            plot_df[x_col],
            plot_df[column],
            s=65,
            edgecolors="black",
            linewidths=0.4,
            zorder=2,
        )
    else:
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
        cbar.set_label(color_label or color_col, fontweight="bold")

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(get_ylabel(metric, value_kind), fontweight="bold")
    ax.set_title(
        f"{metric.replace('_', ' ')} - {title_suffix}",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    if value_kind in {"delta", "relative_delta"}:
        ax.axhline(0.0, linewidth=1.0, linestyle="--")
    elif reference_value is not None and use_secondary_days_axis:
        ax.axhline(
            reference_value,
            linewidth=1.0,
            linestyle="--",
            label="complete",
        )
        ax.legend(frameon=False)

    if value_kind == "relative_delta":
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x * 100:.1f}%")

    if use_secondary_days_axis:
        add_budget_secondary_xaxis(
            ax=ax,
            plot_df=plot_df,
            x_col=x_col,
            days_col="n_days",
        )

    fig.tight_layout()
    save_figure(fig, plots_dir / filename, config)
    plt.close(fig)

def plot_budget_combined_relative_profiles(
    df: pd.DataFrame,
    plots_dir: Path,
    config: dict[str, Any],
) -> None:
    """
    Plot selected budget-scan profiles as relative deltas against the complete case.

    x = number of nodes
    secondary x-axis = representative days
    y = relative delta [%]
    """
    plot_df = df[df["scan_type"] == "budget_scan"].copy()

    if plot_df.empty:
        plot_df = df[df["scan_type"] != "complete"].copy()
        filename_suffix = "reduced_scan"
        title_suffix = "all reduced runs"
    else:
        filename_suffix = "budget_scan"
        title_suffix = "constant geo-temporal budget"

    if plot_df.empty:
        print("[WARNING] No reduced runs available for combined budget profile plot.")
        return

    plot_df = plot_df.sort_values("n_nodes")

    profiles = config.get("plots", {}).get("budget_combined_profiles", [])
    if not profiles:
        profiles = [
            {"metric": "conventional_generation", "label": "Conventional generation"},
            {"metric": "renewable_capacity", "label": "Renewable capacity"},
            {"metric": "store_energy_capacity", "label": "Store capacity"},
            {"metric": "objective", "label": "Objective function"},
            {"metric": "renewable_generation", "label": "Renewable generation"},
            {"metric": "link_power_capacity", "label": "Link capacity"},
        ]

    fig, ax = plt.subplots(figsize=(8.6, 5.5))

    plotted_any = False

    for profile in profiles:
        metric = profile["metric"]
        label = profile.get("label", metric.replace("_", " "))
        column = f"{metric}_relative_delta"

        if column not in plot_df.columns:
            print(f"[WARNING] Column '{column}' not found. Skipping combined profile.")
            continue

        profile_df = plot_df.dropna(subset=["n_nodes", "n_days", column]).copy()
        if profile_df.empty:
            continue

        y_percent = profile_df[column] * 100.0

        ax.plot(
            profile_df["n_nodes"],
            y_percent,
            marker="o",
            linewidth=1.8,
            markersize=4.8,
            label=label,
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("[WARNING] No valid profile was plotted in combined budget profile plot.")
        return

    export_plot_data(
        plot_df=plot_df,
        plots_dir=plots_dir,
        filename=f"combined_relative_profiles_2d_{filename_suffix}",
    )

    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Number of nodes", fontweight="bold")
    ax.set_ylabel("Relative delta vs complete [%]", fontweight="bold")
    ax.set_title(
        f"Budget scan comparison - {title_suffix}",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncols=2)

    add_budget_secondary_xaxis(
        ax=ax,
        plot_df=plot_df,
        x_col="n_nodes",
        days_col="n_days",
    )

    fig.tight_layout()

    filename = sanitize_filename(f"combined_relative_profiles_2d_{filename_suffix}")
    save_figure(fig, plots_dir / filename, config)
    plt.close(fig)
    

def plot_optimization_vs_clustering_objective(
    df: pd.DataFrame,
    plots_dir: Path,
    config: dict[str, Any],
) -> None:
    """
    Plot optimization objective and clustering objective along the budget scan.

    The optimization objective is compared against the complete-network reference.
    The clustering objective is shown on a secondary y-axis because it generally
    has a different scale and unit.
    """
    required = {"objective", "clustering_objective", "n_nodes", "n_days", "scan_type", "run"}
    missing = required - set(df.columns)
    if missing:
        print(
            "[WARNING] Missing columns for objective comparison plot: "
            f"{sorted(missing)}. Skipping."
        )
        return

    reference_run = config.get("reference", {}).get("run", "complete")
    reference_df = df[df["run"] == reference_run].copy()

    optimization_reference = None
    if not reference_df.empty and reference_df["objective"].notna().any():
        optimization_reference = float(reference_df["objective"].dropna().iloc[0])

    plot_df = df[df["scan_type"] == "budget_scan"].copy()

    if plot_df.empty:
        plot_df = df[df["scan_type"] != "complete"].copy()
        filename_suffix = "reduced_scan"
        title_suffix = "all reduced runs"
    else:
        filename_suffix = "budget_scan"
        title_suffix = "constant geo-temporal budget"

    plot_df = plot_df.dropna(
        subset=["objective", "clustering_objective", "n_nodes", "n_days"]
    )
    plot_df = plot_df.sort_values("n_nodes")

    if plot_df.empty:
        print(
            "[WARNING] No runs have both optimization and clustering objectives. "
            "Skipping objective comparison plot."
        )
        return

    filename = sanitize_filename(
        f"optimization_vs_clustering_objective_2d_{filename_suffix}"
    )

    export_plot_data(
        plot_df=plot_df,
        plots_dir=plots_dir,
        filename=filename,
    )

    fig, ax1 = plt.subplots(figsize=(8.6, 5.5))

    optimization_color = "tab:blue"
    clustering_color = "tab:orange"
    reference_color = "black"

    line1 = ax1.plot(
        plot_df["n_nodes"],
        plot_df["objective"],
        marker="o",
        linewidth=1.8,
        color=optimization_color,
        label="Optimization objective",
    )

    reference_line = []
    if optimization_reference is not None:
        reference_line = [
            ax1.axhline(
                optimization_reference,
                linewidth=1.2,
                linestyle="--",
                color=reference_color,
                label="Complete optimization objective",
            )
        ]

    ax1.set_xlabel("Number of nodes", fontweight="bold")
    ax1.set_ylabel("Optimization objective", fontweight="bold", color=optimization_color)
    ax1.tick_params(axis="y", labelcolor=optimization_color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    line2 = ax2.plot(
        plot_df["n_nodes"],
        plot_df["clustering_objective"],
        marker="s",
        linewidth=1.8,
        linestyle="--",
        color=clustering_color,
        label="Clustering objective",
    )

    ax2.set_ylabel("Clustering objective", fontweight="bold", color=clustering_color)
    ax2.tick_params(axis="y", labelcolor=clustering_color)

    add_budget_secondary_xaxis(
        ax=ax1,
        plot_df=plot_df,
        x_col="n_nodes",
        days_col="n_days",
    )

    lines = line1 + line2 + reference_line
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False)

    ax1.set_title(
        f"Optimization vs clustering objective - {title_suffix}",
        fontweight="bold",
    )

    fig.tight_layout()
    save_figure(fig, plots_dir / filename, config)
    plt.close(fig)

def plot_optimization_without_line_cost_vs_clustering_objective(
    df: pd.DataFrame,
    plots_dir: Path,
    config: dict[str, Any],
) -> None:
    """
    Plot optimization objective without line costs and clustering objective.

    The corrected optimization objective removes capex and opex of selected
    components, typically Lines, from the original optimization objective.
    """
    objective_column = "objective_without_line_cost"

    required = {
        objective_column,
        "clustering_objective",
        "n_nodes",
        "n_days",
        "scan_type",
        "run",
    }
    missing = required - set(df.columns)
    if missing:
        print(
            "[WARNING] Missing columns for corrected objective comparison plot: "
            f"{sorted(missing)}. Skipping."
        )
        return

    reference_run = config.get("reference", {}).get("run", "complete")
    reference_df = df[df["run"] == reference_run].copy()

    optimization_reference = None
    if not reference_df.empty and reference_df[objective_column].notna().any():
        optimization_reference = float(reference_df[objective_column].dropna().iloc[0])

    plot_df = df[df["scan_type"] == "budget_scan"].copy()

    if plot_df.empty:
        plot_df = df[df["scan_type"] != "complete"].copy()
        filename_suffix = "reduced_scan"
        title_suffix = "all reduced runs"
    else:
        filename_suffix = "budget_scan"
        title_suffix = "constant geo-temporal budget"

    plot_df = plot_df.dropna(
        subset=[objective_column, "clustering_objective", "n_nodes", "n_days"]
    )
    plot_df = plot_df.sort_values("n_nodes")

    if plot_df.empty:
        print(
            "[WARNING] No runs have both corrected optimization and clustering objectives. "
            "Skipping corrected objective comparison plot."
        )
        return

    filename = sanitize_filename(
        f"optimization_without_line_cost_vs_clustering_objective_2d_{filename_suffix}"
    )

    export_plot_data(
        plot_df=plot_df,
        plots_dir=plots_dir,
        filename=filename,
    )

    fig, ax1 = plt.subplots(figsize=(8.6, 5.5))

    optimization_color = "tab:green"
    clustering_color = "tab:orange"
    reference_color = "black"

    line1 = ax1.plot(
        plot_df["n_nodes"],
        plot_df[objective_column],
        marker="o",
        linewidth=1.8,
        color=optimization_color,
        label="Optimization objective without line costs",
    )

    reference_line = []
    if optimization_reference is not None:
        reference_line = [
            ax1.axhline(
                optimization_reference,
                linewidth=1.2,
                linestyle="--",
                color=reference_color,
                label="Complete optimization objective without line costs",
            )
        ]

    ax1.set_xlabel("Number of nodes", fontweight="bold")
    ax1.set_ylabel(
        "Optimization objective without line costs",
        fontweight="bold",
        color=optimization_color,
    )
    ax1.tick_params(axis="y", labelcolor=optimization_color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    line2 = ax2.plot(
        plot_df["n_nodes"],
        plot_df["clustering_objective"],
        marker="s",
        linewidth=1.8,
        linestyle="--",
        color=clustering_color,
        label="Clustering objective",
    )

    ax2.set_ylabel("Clustering objective", fontweight="bold", color=clustering_color)
    ax2.tick_params(axis="y", labelcolor=clustering_color)

    add_budget_secondary_xaxis(
        ax=ax1,
        plot_df=plot_df,
        x_col="n_nodes",
        days_col="n_days",
    )

    lines = line1 + line2 + reference_line
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False)

    ax1.set_title(
        f"Optimization objective without line costs vs clustering objective - {title_suffix}",
        fontweight="bold",
    )

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

def add_budget_secondary_xaxis(
    ax,
    plot_df: pd.DataFrame,
    x_col: str = "n_nodes",
    days_col: str = "n_days",
    max_ticks: int = 12,
) -> None:
    """
    Add a second bottom x-axis showing representative days.

    This is intended for budget scans where x is the number of spatial nodes and
    the complementary temporal resolution is represented by labels on a lower axis.
    """
    if x_col not in plot_df.columns or days_col not in plot_df.columns:
        return

    tick_df = (
        plot_df[[x_col, days_col]]
        .dropna()
        .drop_duplicates(subset=[x_col])
        .sort_values(x_col)
        .reset_index(drop=True)
    )

    if tick_df.empty:
        return

    if len(tick_df) > max_ticks:
        indices = np.linspace(0, len(tick_df) - 1, max_ticks).round().astype(int)
        tick_df = tick_df.iloc[np.unique(indices)]

    secax = ax.secondary_xaxis("bottom")
    secax.spines["bottom"].set_position(("outward", 42))
    secax.set_xticks(tick_df[x_col].to_numpy())
    secax.set_xticklabels([str(int(v)) for v in tick_df[days_col].to_numpy()])
    secax.set_xlabel("Number of representative days", fontweight="bold")

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