# -*- coding: utf-8 -*-
"""
Plot summary diagnostics for geo-temporal clustering scan.

Input:
- scan_summary.csv

Outputs:
- objective_vs_initial_nodes.png
- objective_vs_initial_days.png
- objective_vs_final_nodes.png
- objective_vs_final_days.png
- final_shape_scatter.png
- initial_to_final_nodes.png
- initial_to_final_days.png
- best_objective_by_final_shape.png
- objective_by_algorithm_scenario.png, when algorithm_scenario is available
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS
# =========================

SCAN_DIR = Path("resources/geotemporal_clustering_scan/400_mean_realmean_0.15_full")

SUMMARY_CSV = SCAN_DIR / "scan_summary.csv"
FINAL_SHAPE_CSV = SCAN_DIR / "final_shape_summary.csv"

OUT_DIR = SCAN_DIR / "plots_summary"

DROP_FULL_BASELINE_FROM_INIT_PLOTS = True

TOP_N_FINAL_SHAPES = 30

PLOT_DPI = 240


# =========================
# Helpers
# =========================

def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required = [
        "run_id",
        "init_mode",
        "init_nodes",
        "init_days",
        "final_K_nodes",
        "final_K_days",
        "final_total_steps",
        "objective",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")

    return df


def _scenario_group_cols(df: pd.DataFrame) -> list[str]:
    """Return grouping columns that keep algorithm scenarios separate when present."""
    return ["algorithm_scenario"] if "algorithm_scenario" in df.columns else []


def _best_run_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "run_id",
        "algorithm_scenario",
        "spatial_algorithm",
        "temporal_algorithm",
        "objective_spatial_reconstruction",
        "objective_temporal_reconstruction",
        "init_nodes",
        "init_days",
        "final_K_nodes",
        "final_K_days",
        "final_total_steps",
        "objective",
    ]
    return [c for c in cols if c in df.columns]


def _prepare_init_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if DROP_FULL_BASELINE_FROM_INIT_PLOTS:
        out = out[out["init_mode"].astype(str) != "full"].copy()

    out = out.dropna(subset=["init_nodes", "init_days"]).copy()
    out["init_nodes"] = out["init_nodes"].astype(int)
    out["init_days"] = out["init_days"].astype(int)
    out["init_steps"] = out["init_nodes"] * out["init_days"]

    return out


def _style_axis(ax, *, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _style_colorbar(cbar, label: str) -> None:
    cbar.set_label(label, fontweight="bold")


def _save(fig, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)


def _short_scenario_label(value: object) -> str:
    """Shorten generated scenario IDs for plot labels."""
    text = str(value)
    text = text.replace("sp", "S:", 1)
    text = text.replace("_tm", "\nT:")
    text = text.replace("_obj", "\nObj:")
    text = text.replace("kmedoids", "medoids")
    text = text.replace("kmeans", "means")
    text = text.replace("_", "/")
    return text


def _shape_label(row: pd.Series, include_scenario: bool = True) -> str:
    shape = f"{int(row['final_K_nodes'])}x{int(row['final_K_days'])}"
    if include_scenario and "algorithm_scenario" in row.index:
        return f"{_short_scenario_label(row['algorithm_scenario'])}\n{shape}"
    return shape


def _scatter_objective(
    df: pd.DataFrame,
    *,
    x_col: str,
    color_col: str,
    xlabel: str,
    colorbar_label: str,
    title: str,
    out_path: Path,
) -> None:
    d = df.copy()
    if d.empty:
        print(f"Skipping {out_path.name}: no rows available.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.4))

    sc = ax.scatter(
        d[x_col],
        d["objective"],
        s=52,
        alpha=0.82,
        c=d[color_col],
        edgecolors="white",
        linewidths=0.35,
    )

    _style_axis(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel="Final objective",
    )

    cbar = fig.colorbar(sc, ax=ax)
    _style_colorbar(cbar, colorbar_label)

    _save(fig, out_path)


def _initial_to_final(
    df: pd.DataFrame,
    *,
    initial_col: str,
    final_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    d = _prepare_init_df(df)
    if d.empty:
        print(f"Skipping {out_path.name}: no non-baseline initial rows available.")
        return

    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    sc = ax.scatter(
        d[initial_col],
        d[final_col],
        s=52,
        alpha=0.82,
        c=d["objective"],
        edgecolors="white",
        linewidths=0.35,
    )

    lim_min = min(d[initial_col].min(), d[final_col].min())
    lim_max = max(d[initial_col].max(), d[final_col].max())
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        linestyle="--",
        linewidth=1.1,
        color="0.35",
        alpha=0.75,
    )

    _style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel)

    cbar = fig.colorbar(sc, ax=ax)
    _style_colorbar(cbar, "Final objective")

    _save(fig, out_path)


# =========================
# Plot functions
# =========================

def plot_objective_vs_initial_nodes(df: pd.DataFrame, out_path: Path) -> None:
    """Plot final objective as function of initial nodes."""
    _scatter_objective(
        _prepare_init_df(df),
        x_col="init_nodes",
        color_col="init_days",
        xlabel="Initial nodes",
        colorbar_label="Initial days",
        title="Final objective vs initial nodes",
        out_path=out_path,
    )


def plot_objective_vs_initial_days(df: pd.DataFrame, out_path: Path) -> None:
    """Plot final objective as function of initial days."""
    _scatter_objective(
        _prepare_init_df(df),
        x_col="init_days",
        color_col="init_nodes",
        xlabel="Initial days",
        colorbar_label="Initial nodes",
        title="Final objective vs initial days",
        out_path=out_path,
    )


def plot_objective_vs_final_nodes(df: pd.DataFrame, out_path: Path) -> None:
    """Plot final objective as function of final nodes."""
    _scatter_objective(
        df,
        x_col="final_K_nodes",
        color_col="final_K_days",
        xlabel="Final nodes",
        colorbar_label="Final days",
        title="Final objective vs final nodes",
        out_path=out_path,
    )


def plot_objective_vs_final_days(df: pd.DataFrame, out_path: Path) -> None:
    """Plot final objective as function of final days."""
    _scatter_objective(
        df,
        x_col="final_K_days",
        color_col="final_K_nodes",
        xlabel="Final days",
        colorbar_label="Final nodes",
        title="Final objective vs final days",
        out_path=out_path,
    )


def plot_final_shape_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter of final shapes: x=nodes, y=days, color=objective, size=count."""
    group_cols = _scenario_group_cols(df) + [
        "final_K_nodes",
        "final_K_days",
        "final_total_steps",
    ]
    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(
            objective_best=("objective", "min"),
            objective_mean=("objective", "mean"),
            n_runs=("objective", "size"),
        )
        .sort_values("objective_best")
    )

    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    sizes = 42 + 28 * np.sqrt(grouped["n_runs"].to_numpy(dtype=float))

    sc = ax.scatter(
        grouped["final_K_nodes"],
        grouped["final_K_days"],
        s=sizes,
        c=grouped["objective_best"],
        alpha=0.86,
        edgecolors="white",
        linewidths=0.45,
    )

    _style_axis(
        ax,
        title="Final shapes reached by the reducer",
        xlabel="Final nodes",
        ylabel="Final days",
    )

    cbar = fig.colorbar(sc, ax=ax)
    _style_colorbar(cbar, "Best objective")

    # Intentionally no point labels: dense scans are easier to read without them.
    _save(fig, out_path)


def plot_initial_to_final_nodes(df: pd.DataFrame, out_path: Path) -> None:
    """Plot how initial nodes map to final nodes."""
    _initial_to_final(
        df,
        initial_col="init_nodes",
        final_col="final_K_nodes",
        xlabel="Initial nodes",
        ylabel="Final nodes",
        title="Initial-to-final nodes",
        out_path=out_path,
    )


def plot_initial_to_final_days(df: pd.DataFrame, out_path: Path) -> None:
    """Plot how initial days map to final days."""
    _initial_to_final(
        df,
        initial_col="init_days",
        final_col="final_K_days",
        xlabel="Initial days",
        ylabel="Final days",
        title="Initial-to-final days",
        out_path=out_path,
    )


def plot_best_objective_by_final_shape(df: pd.DataFrame, out_path: Path) -> None:
    """Bar plot of the best final shapes."""
    group_cols = _scenario_group_cols(df) + ["final_K_nodes", "final_K_days"]
    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(
            objective_best=("objective", "min"),
            objective_mean=("objective", "mean"),
            n_runs=("objective", "size"),
        )
        .sort_values("objective_best")
        .head(TOP_N_FINAL_SHAPES)
        .copy()
    )

    grouped["shape"] = grouped.apply(_shape_label, axis=1)

    fig_width = max(10, 0.48 * len(grouped))
    fig, ax = plt.subplots(figsize=(fig_width, 5.6))

    ax.bar(grouped["shape"], grouped["objective_best"], color="#4C78A8")

    _style_axis(
        ax,
        title=f"Top {len(grouped)} final shapes by best objective",
        xlabel="Final shape: nodes x days",
        ylabel="Best objective",
    )

    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    _save(fig, out_path)


def plot_objective_by_algorithm_scenario(df: pd.DataFrame, out_path: Path) -> None:
    """Boxplot of objective values by algorithm scenario."""
    if "algorithm_scenario" not in df.columns:
        print(f"Skipping {out_path.name}: algorithm_scenario column not available.")
        return

    order = (
        df.groupby("algorithm_scenario")["objective"]
        .min()
        .sort_values()
        .index.tolist()
    )

    data = [df.loc[df["algorithm_scenario"] == scenario, "objective"] for scenario in order]
    labels = [_short_scenario_label(scenario) for scenario in order]

    fig_width = max(8.5, 1.35 * len(order))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))

    box = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#72B7B2")
        patch.set_alpha(0.75)

    # Add lightly jittered points to show run density.
    rng = np.random.default_rng(0)
    for i, values in enumerate(data, start=1):
        x = i + rng.normal(0.0, 0.035, size=len(values))
        ax.scatter(x, values, s=18, alpha=0.45, color="#2F4B7C", linewidths=0)

    _style_axis(
        ax,
        title="Objective distribution by algorithm scenario",
        xlabel="Algorithm scenario",
        ylabel="Final objective",
    )
    ax.tick_params(axis="x", rotation=35, labelsize=8)

    _save(fig, out_path)


# =========================
# Entrypoints
# =========================

def generate_scan_summary_plots(
    scan_dir: Path,
    *,
    out_dir: Path | None = None,
    drop_full_baseline_from_init_plots: bool = DROP_FULL_BASELINE_FROM_INIT_PLOTS,
    top_n_final_shapes: int = TOP_N_FINAL_SHAPES,
) -> None:
    global DROP_FULL_BASELINE_FROM_INIT_PLOTS, TOP_N_FINAL_SHAPES

    DROP_FULL_BASELINE_FROM_INIT_PLOTS = bool(drop_full_baseline_from_init_plots)
    TOP_N_FINAL_SHAPES = int(top_n_final_shapes)

    scan_dir = Path(scan_dir)
    summary_csv = scan_dir / "scan_summary.csv"
    out_dir = Path(out_dir) if out_dir is not None else scan_dir / "plots_summary"

    _safe_mkdir(out_dir)

    df = _load_summary(summary_csv)

    print("Loaded scan summary:")
    print(f"  rows: {len(df)}")
    print(f"  best objective: {df['objective'].min():.6e}")
    print()

    print("Best runs:")
    cols = _best_run_columns(df)
    print(df.sort_values("objective")[cols].head(20).to_string(index=False))
    print()

    group_cols = _scenario_group_cols(df) + [
        "final_K_nodes",
        "final_K_days",
        "final_total_steps",
    ]
    shape_summary = (
        df.groupby(group_cols, as_index=False)
        .agg(
            objective_best=("objective", "min"),
            objective_mean=("objective", "mean"),
            objective_std=("objective", "std"),
            n_runs=("objective", "size"),
        )
        .sort_values("objective_best")
    )

    print("Best final shapes:")
    print(shape_summary.head(20).to_string(index=False))

    shape_summary.to_csv(out_dir / "final_shape_summary_from_plot_script.csv", index=False)

    plot_objective_vs_initial_nodes(
        df,
        out_dir / "objective_vs_initial_nodes.png",
    )
    plot_objective_vs_initial_days(
        df,
        out_dir / "objective_vs_initial_days.png",
    )
    plot_objective_vs_final_nodes(
        df,
        out_dir / "objective_vs_final_nodes.png",
    )
    plot_objective_vs_final_days(
        df,
        out_dir / "objective_vs_final_days.png",
    )
    plot_final_shape_scatter(
        df,
        out_dir / "final_shape_scatter.png",
    )
    plot_initial_to_final_nodes(
        df,
        out_dir / "initial_to_final_nodes.png",
    )
    plot_initial_to_final_days(
        df,
        out_dir / "initial_to_final_days.png",
    )
    plot_best_objective_by_final_shape(
        df,
        out_dir / "best_objective_by_final_shape.png",
    )
    plot_objective_by_algorithm_scenario(
        df,
        out_dir / "objective_by_algorithm_scenario.png",
    )

    print()
    print(f"Plots written to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot summary diagnostics for a geo-temporal clustering scan."
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=SCAN_DIR,
        help="Directory containing scan_summary.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Plot output directory. Defaults to <scan-dir>/plots_summary.",
    )
    parser.add_argument(
        "--keep-full-baseline",
        action="store_true",
        help="Include init_mode=full rows in initial-condition plots.",
    )
    parser.add_argument(
        "--top-n-final-shapes",
        type=int,
        default=TOP_N_FINAL_SHAPES,
        help="Number of final shapes shown in the bar plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_scan_summary_plots(
        args.scan_dir,
        out_dir=args.out_dir,
        drop_full_baseline_from_init_plots=not args.keep_full_baseline,
        top_n_final_shapes=args.top_n_final_shapes,
    )


if __name__ == "__main__":
    main()
