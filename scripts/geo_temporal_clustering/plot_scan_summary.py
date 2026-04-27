# -*- coding: utf-8 -*-
"""
Plot summary diagnostics for geo-temporal clustering scan.

Input:
- scan_summary.csv
- final_shape_summary.csv, optional

Outputs:
- objective_vs_initial_nodes.png
- objective_vs_final_nodes.png
- final_shape_scatter.png
- initial_to_final_nodes.png
- best_objective_by_final_shape.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS
# =========================

SCAN_DIR = Path("resources/geotemporal_clustering_scan/900_band_0.15")

SUMMARY_CSV = SCAN_DIR / "scan_summary.csv"
FINAL_SHAPE_CSV = SCAN_DIR / "final_shape_summary.csv"

OUT_DIR = SCAN_DIR / "plots_summary"

DROP_FULL_BASELINE_FROM_INIT_PLOTS = True

TOP_N_FINAL_SHAPES = 30


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


def _prepare_init_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if DROP_FULL_BASELINE_FROM_INIT_PLOTS:
        out = out[out["init_mode"].astype(str) != "full"].copy()

    out = out.dropna(subset=["init_nodes", "init_days"]).copy()
    out["init_nodes"] = out["init_nodes"].astype(int)
    out["init_days"] = out["init_days"].astype(int)
    out["init_steps"] = out["init_nodes"] * out["init_days"]

    return out


def plot_objective_vs_initial_nodes(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot final objective as function of initial K_nodes.
    """
    d = _prepare_init_df(df)

    fig, ax = plt.subplots(figsize=(9, 5))

    sc = ax.scatter(
        d["init_nodes"],
        d["objective"],
        s=45,
        alpha=0.8,
        c=d["init_days"],
    )

    ax.set_xlabel("Initial K_nodes")
    ax.set_ylabel("Final objective")
    ax.set_title("Final objective vs initial spatial clusters")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Initial K_days")

    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_objective_vs_final_nodes(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot final objective as function of final K_nodes.
    """
    d = df.copy()

    fig, ax = plt.subplots(figsize=(9, 5))

    sc = ax.scatter(
        d["final_K_nodes"],
        d["objective"],
        s=45,
        alpha=0.8,
        c=d["final_K_days"],
    )

    ax.set_xlabel("Final K_nodes")
    ax.set_ylabel("Final objective")
    ax.set_title("Final objective vs final spatial clusters")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Final K_days")

    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_final_shape_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """
    Scatter of final shapes: x=K_nodes, y=K_days, color=objective, size=count.
    """
    grouped = (
        df.groupby(["final_K_nodes", "final_K_days", "final_total_steps"], as_index=False)
        .agg(
            objective_best=("objective", "min"),
            objective_mean=("objective", "mean"),
            n_runs=("objective", "size"),
        )
        .sort_values("objective_best")
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    sizes = 40 + 25 * np.sqrt(grouped["n_runs"].to_numpy(dtype=float))

    sc = ax.scatter(
        grouped["final_K_nodes"],
        grouped["final_K_days"],
        s=sizes,
        c=grouped["objective_best"],
        alpha=0.85,
    )

    ax.set_xlabel("Final K_nodes")
    ax.set_ylabel("Final K_days")
    ax.set_title("Final shapes reached by the reducer")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Best objective")

    ax.grid(True, alpha=0.25)

    # Annotate only the best few points to avoid clutter.
    for _, row in grouped.head(12).iterrows():
        ax.annotate(
            f"{int(row['final_K_nodes'])},{int(row['final_K_days'])}",
            (row["final_K_nodes"], row["final_K_days"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_initial_to_final_nodes(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot how initial K_nodes maps to final K_nodes.
    """
    d = _prepare_init_df(df)

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        d["init_nodes"],
        d["final_K_nodes"],
        s=45,
        alpha=0.8,
        c=d["objective"],
    )

    lim_min = min(d["init_nodes"].min(), d["final_K_nodes"].min())
    lim_max = max(d["init_nodes"].max(), d["final_K_nodes"].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1)

    ax.set_xlabel("Initial K_nodes")
    ax.set_ylabel("Final K_nodes")
    ax.set_title("Initial-to-final spatial resolution")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Final objective")

    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_best_objective_by_final_shape(df: pd.DataFrame, out_path: Path) -> None:
    """
    Bar plot of the best final shapes.
    """
    grouped = (
        df.groupby(["final_K_nodes", "final_K_days"], as_index=False)
        .agg(
            objective_best=("objective", "min"),
            objective_mean=("objective", "mean"),
            n_runs=("objective", "size"),
        )
        .sort_values("objective_best")
        .head(TOP_N_FINAL_SHAPES)
        .copy()
    )

    grouped["shape"] = (
        grouped["final_K_nodes"].astype(int).astype(str)
        + "x"
        + grouped["final_K_days"].astype(int).astype(str)
    )

    fig, ax = plt.subplots(figsize=(max(9, 0.35 * len(grouped)), 5))

    ax.bar(grouped["shape"], grouped["objective_best"])

    ax.set_xlabel("Final shape K_nodes x K_days")
    ax.set_ylabel("Best objective")
    ax.set_title(f"Top {len(grouped)} final shapes by best objective")

    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    _safe_mkdir(OUT_DIR)

    df = _load_summary(SUMMARY_CSV)

    print("Loaded scan summary:")
    print(f"  rows: {len(df)}")
    print(f"  best objective: {df['objective'].min():.6e}")
    print()

    print("Best runs:")
    cols = [
        "run_id",
        "init_nodes",
        "init_days",
        "final_K_nodes",
        "final_K_days",
        "final_total_steps",
        "objective",
    ]
    print(df.sort_values("objective")[cols].head(20).to_string(index=False))
    print()

    shape_summary = (
        df.groupby(["final_K_nodes", "final_K_days", "final_total_steps"], as_index=False)
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

    shape_summary.to_csv(OUT_DIR / "final_shape_summary_from_plot_script.csv", index=False)

    plot_objective_vs_initial_nodes(
        df,
        OUT_DIR / "objective_vs_initial_nodes.png",
    )

    plot_objective_vs_final_nodes(
        df,
        OUT_DIR / "objective_vs_final_nodes.png",
    )

    plot_final_shape_scatter(
        df,
        OUT_DIR / "final_shape_scatter.png",
    )

    plot_initial_to_final_nodes(
        df,
        OUT_DIR / "initial_to_final_nodes.png",
    )

    plot_best_objective_by_final_shape(
        df,
        OUT_DIR / "best_objective_by_final_shape.png",
    )

    print()
    print(f"Plots written to: {OUT_DIR}")


if __name__ == "__main__":
    main()