# scripts/geo_temporal_clustering/expand_gt_network.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def _read_days_assignment(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _infer_day_sequence(df: pd.DataFrame) -> Tuple[np.ndarray, int]:
    cols = {c.lower(): c for c in df.columns}

    day_candidates = ["day", "day_index", "original_day", "day_in_year", "d"]
    rep_candidates = ["rep_day_index", "rep_day", "representative_day", "cluster_day", "rep", "representative"]

    day_col = next((cols[c] for c in day_candidates if c in cols), None)
    rep_col = next((cols[c] for c in rep_candidates if c in cols), None)

    if day_col is None or rep_col is None:
        raise ValueError(
            f"days_assignment.csv columns not recognized. "
            f"Need day_index + rep_day_index (or equivalents). Got: {list(df.columns)}"
        )

    tmp = df[[day_col, rep_col]].copy().sort_values(day_col)
    day_vals = tmp[day_col].to_numpy()
    rep_vals = tmp[rep_col].to_numpy()

    if day_vals.min() == 1 and np.all(np.unique(day_vals) == np.arange(1, day_vals.max() + 1)):
        day_vals = day_vals - 1

    n_days = int(day_vals.max() + 1)
    day_sequence = np.empty(n_days, dtype=int)
    day_sequence[day_vals.astype(int)] = rep_vals.astype(int)
    return day_sequence, n_days


def _build_full_snapshots(day_sequence: np.ndarray) -> pd.MultiIndex:
    n_days = len(day_sequence)
    days_full = np.arange(n_days, dtype=int)
    hours = np.arange(24, dtype=int)
    full = pd.MultiIndex.from_product([days_full, hours], names=["day", "hour"])
    return full


def _build_full_to_clustered_map(n_snapshots: pd.Index, day_sequence: np.ndarray) -> pd.Index:
    n_days = len(day_sequence)
    rep_days_full = np.repeat(day_sequence, 24)
    hours_full = np.tile(np.arange(24), n_days)

    if isinstance(n_snapshots, pd.MultiIndex) and n_snapshots.nlevels >= 2:
        # Assume (rep_day, hour, ...)
        if n_snapshots.nlevels == 2:
            return pd.MultiIndex.from_arrays([rep_days_full, hours_full], names=n_snapshots.names)
        else:
            fillers = []
            for lv in range(2, n_snapshots.nlevels):
                fillers.append(np.repeat(n_snapshots.levels[lv][0], len(rep_days_full)))
            arrays = [rep_days_full, hours_full] + fillers
            return pd.MultiIndex.from_arrays(arrays, names=n_snapshots.names)

    # Fallback: not ideal; adapt if your clustered snapshots are not MultiIndex
    logger.warning(
        "Clustered snapshots are not MultiIndex; using fallback 'rep-hour' string keys. "
        "You may need to adapt mapping to your snapshot format."
    )
    return pd.Index([f"{d}-{h}" for d, h in zip(rep_days_full, hours_full)], name="snapshot")


def _expand_time_series_tables(n: pypsa.Network, full_snapshots: pd.Index, map_full_to_clustered: pd.Index) -> None:
    """
    Expand all *_t tables by mapping each full snapshot to its representative clustered snapshot.
    """
    # For each component, expand all time-dependent attributes present in component.pnl
    for c in n.iterate_components(n.components.keys()):
        pnl = c.pnl
        if pnl is None:
            continue

        for attr, df in pnl.items():
            if df is None or not isinstance(df, (pd.DataFrame, pd.Series)):
                continue

            # Only expand if indexed by snapshots
            try:
                if df.index.equals(n.snapshots):
                    # Expand by reindexing using map_full_to_clustered
                    expanded = df.reindex(map_full_to_clustered).copy()
                    expanded.index = full_snapshots
                    pnl[attr] = expanded
            except Exception:
                # Some pnl items may not have snapshot index; ignore
                continue


def _expand_snapshot_weightings(n: pypsa.Network, full_snapshots: pd.Index) -> None:
    # Keep weights uniform (1 hour). If you want to preserve objective scaling, adapt here.
    n.set_snapshots(full_snapshots)
    if hasattr(n, "snapshot_weightings") and isinstance(n.snapshot_weightings, pd.DataFrame):
        n.snapshot_weightings = pd.DataFrame(
            1.0, index=full_snapshots, columns=n.snapshot_weightings.columns
        )
    else:
        # fallback
        n.snapshot_weightings = pd.DataFrame(1.0, index=full_snapshots, columns=["objective", "stores", "generators"])


def main():
    # snakemake provided by Snakemake runtime
    network_gt_path = snakemake.input.network_gt
    days_assignment_path = snakemake.input.days_assignment
    out_path = snakemake.output.network

    logger.info(f"Loading clustered optimized network: {network_gt_path}")
    n = pypsa.Network(network_gt_path)

    logger.info(f"Reading days_assignment: {days_assignment_path}")
    df = _read_days_assignment(days_assignment_path)
    day_sequence, n_days = _infer_day_sequence(df)

    full_snapshots = _build_full_snapshots(day_sequence)
    map_full_to_clustered = _build_full_to_clustered_map(n.snapshots, day_sequence)

    logger.info(f"Expanding time series to full length: {len(full_snapshots)} snapshots")
    # Set full snapshots and weightings
    _expand_snapshot_weightings(n, full_snapshots)

    # Expand all pnl tables
    _expand_time_series_tables(n, full_snapshots, map_full_to_clustered)

    logger.info(f"Saving expanded network: {out_path}")
    n.export_to_netcdf(out_path)


if __name__ == "__main__":
    main()