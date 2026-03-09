# scripts/geo_temporal_clustering/expand_gt_network.py

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers for day mapping
# =============================================================================

def _read_days_assignment(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _infer_day_sequence(df: pd.DataFrame) -> Tuple[np.ndarray, int]:
    cols = {c.lower(): c for c in df.columns}

    day_candidates = ["day", "day_index", "original_day", "day_in_year", "d"]
    rep_candidates = [
        "rep_day_index",
        "rep_day",
        "representative_day",
        "cluster_day",
        "rep",
        "representative",
    ]

    day_col = next((cols[c] for c in day_candidates if c in cols), None)
    rep_col = next((cols[c] for c in rep_candidates if c in cols), None)

    if day_col is None or rep_col is None:
        raise ValueError(
            "days_assignment.csv columns not recognized. "
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


# =============================================================================
# Snapshot reconstruction
# =============================================================================

def _build_full_snapshots_like_clustered(n_snapshots: pd.Index, day_sequence: np.ndarray) -> pd.Index:
    """
    Build the expanded full-year snapshot index.

    If clustered snapshots are DatetimeIndex, return full DatetimeIndex.
    If clustered snapshots are MultiIndex, return full MultiIndex (day, hour).
    """
    n_days = len(day_sequence)

    if isinstance(n_snapshots, pd.DatetimeIndex):
        year = int(n_snapshots.min().year)
        base = pd.Timestamp(f"{year}-01-01 00:00:00")
        return pd.date_range(start=base, periods=n_days * 24, freq="h")

    # Fallback: use MultiIndex(day, hour)
    days_full = np.arange(n_days, dtype=int)
    hours = np.arange(24, dtype=int)
    return pd.MultiIndex.from_product([days_full, hours], names=["day", "hour"])


def _build_full_to_clustered_map(n_snapshots: pd.Index, day_sequence: np.ndarray) -> pd.Index:
    """
    Build mapping from each full-year snapshot to the clustered representative snapshot.
    """
    n_days = len(day_sequence)
    rep_days_full = np.repeat(day_sequence, 24)
    hours_full = np.tile(np.arange(24), n_days)

    if isinstance(n_snapshots, pd.MultiIndex):
        if n_snapshots.nlevels == 2:
            return pd.MultiIndex.from_arrays(
                [rep_days_full, hours_full],
                names=n_snapshots.names,
            )

        fillers = []
        for lv in range(2, n_snapshots.nlevels):
            fillers.append(np.repeat(n_snapshots.levels[lv][0], len(rep_days_full)))

        arrays = [rep_days_full, hours_full] + fillers
        return pd.MultiIndex.from_arrays(arrays, names=n_snapshots.names)

    if isinstance(n_snapshots, pd.DatetimeIndex):
        year = int(n_snapshots.min().year)
        base = pd.Timestamp(f"{year}-01-01 00:00:00")
        return pd.DatetimeIndex(
            [base + pd.Timedelta(days=int(d), hours=int(h)) for d, h in zip(rep_days_full, hours_full)]
        )

    logger.warning(
        "Clustered snapshots are neither MultiIndex nor DatetimeIndex; using fallback string keys."
    )
    return pd.Index([f"{d}-{h}" for d, h in zip(rep_days_full, hours_full)], name="snapshot")


# =============================================================================
# Expansion of time-dependent tables
# =============================================================================

def _expand_time_series_tables(
    n: pypsa.Network,
    original_snapshots: pd.Index,
    full_snapshots: pd.Index,
    map_full_to_clustered: pd.Index,
) -> None:
    """
    Expand all component time series tables indexed by original clustered snapshots.
    """
    # Set full snapshots first
    n.set_snapshots(full_snapshots)

    # Uniform weights on the reconstructed full chronology
    if hasattr(n, "snapshot_weightings") and isinstance(n.snapshot_weightings, pd.DataFrame):
        n.snapshot_weightings = pd.DataFrame(
            1.0, index=full_snapshots, columns=n.snapshot_weightings.columns
        )
    else:
        n.snapshot_weightings = pd.DataFrame(
            1.0,
            index=full_snapshots,
            columns=["objective", "stores", "generators"],
        )

    # Expand all *_t tables
    for c in n.iterate_components(n.components.keys()):
        pnl = c.pnl
        if pnl is None:
            continue

        for attr, df in pnl.items():
            if df is None or not isinstance(df, (pd.DataFrame, pd.Series)):
                continue

            try:
                if df.index.equals(original_snapshots):
                    expanded = df.reindex(map_full_to_clustered).copy()
                    expanded.index = full_snapshots
                    pnl[attr] = expanded
            except Exception:
                continue


# =============================================================================
# CSV loaders for full-length storage energy trajectories
# =============================================================================

def _read_full_timeseries_csv(path: str | Path) -> pd.DataFrame:
    """
    Read a CSV saved from solve_network post-processing.

    Expected format:
    - first column = t
    - remaining columns = component names
    """
    df = pd.read_csv(path, index_col=0)
    df.index.name = "t"
    return df


def _load_full_timeline(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index.name = "t"
    return df


def _assign_storageunit_full_soc(
    n: pypsa.Network,
    full_snapshots: pd.Index,
    soc_csv_path: str | Path,
) -> None:
    """
    Overwrite StorageUnit state_of_charge with full-length reconstructed trajectory.
    """
    if n.storage_units.empty:
        logger.info("No StorageUnits in network. Skipping full SOC assignment.")
        return

    df = _read_full_timeseries_csv(soc_csv_path)

    # keep only matching StorageUnits
    cols = n.storage_units.index.intersection(df.columns)
    if cols.empty:
        logger.warning("No matching StorageUnit columns found in %s", soc_csv_path)
        return

    df = df[cols].copy()

    expected_len = len(full_snapshots) + 1
    if len(df.index) != expected_len:
        raise ValueError(
            f"StorageUnit full SOC CSV has length {len(df.index)}, expected {expected_len} "
            "(T+1 because it includes initial state)."
        )

    # Drop t=0 initial state, keep states aligned to snapshots 1..T
    df = df.iloc[1:].copy()
    df.index = full_snapshots

    # Ensure all StorageUnits are present as columns
    df = df.reindex(columns=n.storage_units.index)

    n.storage_units_t.state_of_charge = df
    logger.info(
        "Assigned expanded StorageUnit state_of_charge with shape %s",
        n.storage_units_t.state_of_charge.shape,
    )


def _assign_store_full_energy(
    n: pypsa.Network,
    full_snapshots: pd.Index,
    e_csv_path: str | Path,
) -> None:
    """
    Overwrite Store energy trajectory with full-length reconstructed trajectory.
    """
    if n.stores.empty:
        logger.info("No Stores in network. Skipping full Store energy assignment.")
        return

    df = _read_full_timeseries_csv(e_csv_path)

    cols = n.stores.index.intersection(df.columns)
    if cols.empty:
        logger.warning("No matching Store columns found in %s", e_csv_path)
        return

    df = df[cols].copy()

    expected_len = len(full_snapshots) + 1
    if len(df.index) != expected_len:
        raise ValueError(
            f"Store full energy CSV has length {len(df.index)}, expected {expected_len} "
            "(T+1 because it includes initial state)."
        )

    # Drop t=0 initial state, keep states aligned to snapshots 1..T
    df = df.iloc[1:].copy()
    df.index = full_snapshots

    df = df.reindex(columns=n.stores.index)

    n.stores_t.e = df
    logger.info(
        "Assigned expanded Store energy with shape %s",
        n.stores_t.e.shape,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    network_gt_path = snakemake.input.network_gt
    days_assignment_path = snakemake.input.days_assignment
    full_timeline_path = snakemake.input.full_timeline
    storage_units_t_full_soc_path = snakemake.input.storage_units_t_full_soc
    stores_t_full_e_path = snakemake.input.stores_t_full_e
    out_path = snakemake.output.network

    logger.info("Loading clustered optimized network: %s", network_gt_path)
    n = pypsa.Network(network_gt_path)

    original_snapshots = n.snapshots.copy()

    logger.info("Reading days_assignment: %s", days_assignment_path)
    df = _read_days_assignment(days_assignment_path)
    day_sequence, n_days = _infer_day_sequence(df)

    logger.info("Reading full timeline metadata: %s", full_timeline_path)
    full_timeline = _load_full_timeline(full_timeline_path)

    full_snapshots = _build_full_snapshots_like_clustered(original_snapshots, day_sequence)
    map_full_to_clustered = _build_full_to_clustered_map(original_snapshots, day_sequence)

    if len(full_timeline) != len(full_snapshots) + 1:
        logger.warning(
            "full_timeline length is %s while expected T+1 is %s. "
            "Proceeding anyway because full energy CSVs are the source of truth for storage levels.",
            len(full_timeline),
            len(full_snapshots) + 1,
        )

    logger.info(
        "Expanding all clustered time series from %s to %s snapshots",
        len(original_snapshots),
        len(full_snapshots),
    )

    _expand_time_series_tables(
        n=n,
        original_snapshots=original_snapshots,
        full_snapshots=full_snapshots,
        map_full_to_clustered=map_full_to_clustered,
    )

    # Overwrite storage energy levels with the reconstructed full-length seasonal results
    _assign_storageunit_full_soc(
        n=n,
        full_snapshots=full_snapshots,
        soc_csv_path=storage_units_t_full_soc_path,
    )

    _assign_store_full_energy(
        n=n,
        full_snapshots=full_snapshots,
        e_csv_path=stores_t_full_e_path,
    )

    logger.info("Saving expanded network: %s", out_path)
    n.export_to_netcdf(out_path)


if __name__ == "__main__":
    main()