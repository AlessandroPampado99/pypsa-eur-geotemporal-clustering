# scripts/geo_temporal_clustering/seasonal_storage.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------
# CSV parsing helpers
# ----------------------------

def _read_days_assignment(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _infer_day_sequence(df: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """
    Build day_sequence: for each original day (0..n_days-1), returns representative day id (rep_day_index).
    Robust to column naming differences.
    """
    cols = {c.lower(): c for c in df.columns}

    # candidate columns
    day_candidates = ["day", "day_index", "original_day", "day_in_year", "d"]
    rep_candidates = ["rep_day_index", "rep_day", "representative_day", "cluster_day", "rep", "representative"]

    day_col = next((cols[c] for c in day_candidates if c in cols), None)
    rep_col = next((cols[c] for c in rep_candidates if c in cols), None)

    if day_col is None or rep_col is None:
        raise ValueError(
            f"days_assignment.csv columns not recognized. "
            f"Need something like day_index + rep_day_index. Got: {list(df.columns)}"
        )

    tmp = df[[day_col, rep_col]].copy()
    tmp = tmp.sort_values(day_col)
    day_vals = tmp[day_col].to_numpy()
    rep_vals = tmp[rep_col].to_numpy()

    # normalize day indexing (0-based contiguous)
    # if day starts at 1, shift
    if day_vals.min() == 1 and np.all(np.unique(day_vals) == np.arange(1, day_vals.max() + 1)):
        day_vals = day_vals - 1

    n_days = int(day_vals.max() + 1)
    day_sequence = np.empty(n_days, dtype=int)
    day_sequence[day_vals.astype(int)] = rep_vals.astype(int)
    return day_sequence, n_days


# ----------------------------
# Linopy/PyPSA model helpers
# ----------------------------

def _drop_constraints_if_exist(n, names: Iterable[str]) -> None:
    if not hasattr(n, "model") or n.model is None:
        logger.warning("Network model not built yet; cannot drop constraints.")
        return

    cons = getattr(n.model, "constraints", None)
    if cons is None:
        logger.warning("No n.model.constraints found; skipping constraint removal.")
        return

    for nm in names:
        if nm in cons:
            logger.info(f"Dropping constraint group: {nm}")
            cons.pop(nm, None)
        else:
            logger.debug(f"Constraint group not present (ok): {nm}")


def _get_var(n, candidates: Iterable[str]):
    """
    Fetch a linopy variable by trying multiple keys, since naming differs across versions.
    Returns None if not found.
    """
    if not hasattr(n, "model") or n.model is None:
        return None

    vars_ = getattr(n.model, "variables", None)
    if vars_ is None:
        return None

    for key in candidates:
        if key in vars_:
            return vars_[key]

    # Some versions nest variables differently; try attribute access as fallback
    for key in candidates:
        try:
            v = vars_.get(key)
            if v is not None:
                return v
        except Exception:
            pass
    return None


def _add_soc_full_for_storageunits(n, snapshots, full_to_clustered: pd.Index, cyclic: bool) -> None:
    """
    Adds soc_full_su[su, t_full] and constraints linking it to clustered p_store/p_dispatch.
    """
    if n.storage_units.empty:
        logger.info("No StorageUnits in network; skipping StorageUnit seasonal SOC.")
        return

    # Try typical variable keys across PyPSA versions
    p_store = _get_var(n, ["StorageUnit-p_store", "StorageUnit-p_store_t", "p_store"])
    p_dispatch = _get_var(n, ["StorageUnit-p_dispatch", "StorageUnit-p_dispatch_t", "p_dispatch"])

    if p_store is None or p_dispatch is None:
        logger.warning(
            "Could not find StorageUnit p_store/p_dispatch variables in linopy model. "
            "Skipping StorageUnit seasonal SOC constraints."
        )
        return

    # Create full SOC variable
    su_i = n.storage_units.index
    T_full = len(full_to_clustered)

    soc_full = n.model.add_variables(
        lower=0,
        name="soc_full_su",
        coords=[("StorageUnit", su_i), ("t_full", np.arange(T_full))]
    )

    # parameters
    su = n.storage_units
    eta_store = su.get("efficiency_store", pd.Series(1.0, index=su_i)).reindex(su_i).fillna(1.0)
    eta_dispatch = su.get("efficiency_dispatch", pd.Series(1.0, index=su_i)).reindex(su_i).fillna(1.0)

    # initial SOC (use state_of_charge_initial if present else 0)
    soc0 = su.get("state_of_charge_initial", pd.Series(0.0, index=su_i)).reindex(su_i).fillna(0.0)

    # Map each full hour -> clustered snapshot label
    # full_to_clustered is Index of clustered snapshots aligned with t_full
    cl_snap = pd.Index(full_to_clustered)

    # Build constraints:
    # soc_full[t] = soc_full[t-1] + eta_store * p_store[cl] - (1/eta_dispatch) * p_dispatch[cl]
    # Use snapshot_weightings? Here we assume hourly steps already.
    cons = []

    for t in range(T_full):
        cl = cl_snap[t]
        if t == 0:
            lhs = soc_full.sel(t_full=0)
            rhs = soc0 + eta_store * p_store.sel(snapshot=cl) - (1.0 / eta_dispatch) * p_dispatch.sel(snapshot=cl)
        else:
            lhs = soc_full.sel(t_full=t)
            rhs = soc_full.sel(t_full=t - 1) + eta_store * p_store.sel(snapshot=cl) - (1.0 / eta_dispatch) * p_dispatch.sel(snapshot=cl)
        cons.append(lhs - rhs)

    n.model.add_constraints(cons, name="StorageUnit-seasonal_soc_full")

    if cyclic:
        n.model.add_constraints(
            soc_full.sel(t_full=T_full - 1) - soc_full.sel(t_full=0),
            name="StorageUnit-seasonal_soc_full_cyclic"
        )


def _add_soc_full_for_stores(n, snapshots, full_to_clustered: pd.Index, cyclic: bool) -> None:
    """
    Adds soc_full_store[store, t_full] and constraints linking it to clustered Store-p.
    This is the robust baseline. If your stores are modeled via Links, you can extend here.
    """
    if n.stores.empty:
        logger.info("No Stores in network; skipping Store seasonal SOC.")
        return

    p_store = _get_var(n, ["Store-p", "Store-p_t", "p"])
    if p_store is None:
        logger.warning(
            "Could not find Store-p variable in linopy model. "
            "Skipping Store seasonal SOC constraints."
        )
        return

    store_i = n.stores.index
    T_full = len(full_to_clustered)

    soc_full = n.model.add_variables(
        lower=0,
        name="soc_full_store",
        coords=[("Store", store_i), ("t_full", np.arange(T_full))]
    )

    st = n.stores
    # PyPSA Store uses efficiency_store/dispatch sometimes; if absent assume 1
    eta_store = st.get("efficiency_store", pd.Series(1.0, index=store_i)).reindex(store_i).fillna(1.0)
    eta_dispatch = st.get("efficiency_dispatch", pd.Series(1.0, index=store_i)).reindex(store_i).fillna(1.0)
    soc0 = st.get("e_initial", pd.Series(0.0, index=store_i)).reindex(store_i).fillna(0.0)

    cl_snap = pd.Index(full_to_clustered)

    cons = []
    for t in range(T_full):
        cl = cl_snap[t]
        # Convention: Store-p is net dispatch (positive discharging, negative charging)
        # energy change = -p (if p>0 discharge decreases SOC), but this depends on convention.
        # PyPSA store energy balance typically: e_t = e_{t-1} + eta_store * (-p_negative) - (1/eta_dispatch) * p_positive
        # We approximate by splitting sign via two nonnegative vars if present; but we likely only have one p.
        # So use simplest: e_t = e_{t-1} - p[cl]
        if t == 0:
            lhs = soc_full.sel(t_full=0)
            rhs = soc0 - p_store.sel(snapshot=cl)
        else:
            lhs = soc_full.sel(t_full=t)
            rhs = soc_full.sel(t_full=t - 1) - p_store.sel(snapshot=cl)
        cons.append(lhs - rhs)

    n.model.add_constraints(cons, name="Store-seasonal_soc_full")

    if cyclic:
        n.model.add_constraints(
            soc_full.sel(t_full=T_full - 1) - soc_full.sel(t_full=0),
            name="Store-seasonal_soc_full_cyclic"
        )


def add_seasonal_storage_constraints(n, snapshots, snakemake) -> None:
    """
    Seasonal storage SOC on full chronology (hole-free) while optimizing on clustered snapshots.

    Expected:
      - snakemake.input.days_assignment points to days_assignment.csv (original day -> representative day)
      - clustered snapshots index in n.snapshots is compatible with (rep_day_index, hour) or similar

    Steps:
      1) read day_sequence from days_assignment.csv
      2) remove built-in SOC constraints for StorageUnit + Store
      3) add soc_full_* variables and full chronology constraints using clustered dispatch vars
      4) optionally cyclic soc(T)=soc(0)
    """
    if not hasattr(snakemake, "input") or not hasattr(snakemake.input, "days_assignment"):
        raise ValueError("snakemake.input.days_assignment missing; cannot build seasonal-storage constraints.")

    days_path = snakemake.input.days_assignment
    logger.info(f"Reading days_assignment from: {days_path}")
    df = _read_days_assignment(days_path)
    day_sequence, n_days = _infer_day_sequence(df)

    # Full timeline length
    T_full = int(24 * n_days)

    # Determine how clustered snapshots are indexed.
    # We build, for each full hour, the corresponding clustered snapshot label.
    # Default assumption: clustered snapshots can be selected by (rep_day_index, hour)
    # If not MultiIndex, try to match by constructing a string key.
    n_snap = n.snapshots
    is_mi = isinstance(n_snap, pd.MultiIndex)

    rep_days_full = np.repeat(day_sequence, 24)
    hours_full = np.tile(np.arange(24), n_days)

    if is_mi and n_snap.nlevels >= 2:
        # assume levels: (rep_day, hour, ...) take first two levels
        full_to_clustered = pd.MultiIndex.from_arrays([rep_days_full, hours_full], names=n_snap.names[:2])
        # If model uses extra levels, try to align by taking first two
        # Selection below uses .sel(snapshot=label) where label must exist;
        # we therefore map to an Index that matches exactly an existing key.
        # Best effort: if snapshot index has >2 levels, we try to fill remaining with first element.
        if n_snap.nlevels > 2:
            fillers = []
            for lv in range(2, n_snap.nlevels):
                fillers.append(np.repeat(n_snap.levels[lv][0], T_full))
            arrays = [rep_days_full, hours_full] + fillers
            full_to_clustered = pd.MultiIndex.from_arrays(arrays, names=n_snap.names)
    else:
        # fallback: if snapshots are strings like "d{rep}_h{hour}" you can adapt this.
        # Here we attempt to match by exact formatting "rep_day hour" not guaranteed.
        full_to_clustered = pd.Index([f"{d}-{h}" for d, h in zip(rep_days_full, hours_full)], name="snapshot")
        logger.warning(
            "Clustered snapshots are not a MultiIndex. "
            "Fallback mapping uses 'rep-hour' strings; you may need to adapt mapping."
        )

    # Remove built-in SOC constraints
    _drop_constraints_if_exist(n, [
        "StorageUnit-energy_balance",
        "StorageUnit-fix-state_of_charge-lower",
        "StorageUnit-fix-state_of_charge-upper",
        "Store-energy_balance",
        "Store-fix-e-lower",
        "Store-fix-e-upper",
    ])

    # cyclic flag from config (optional)
    cyclic = False
    try:
        cyclic = bool(snakemake.config.get("geotemporal", {}).get("seasonal_storage", {}).get("cyclic", False))
    except Exception:
        cyclic = False

    logger.info(f"Adding seasonal SOC full chronology constraints (T_full={T_full}, cyclic={cyclic})")

    # Add constraints for StorageUnits and Stores
    _add_soc_full_for_storageunits(n, snapshots, full_to_clustered=full_to_clustered, cyclic=cyclic)
    _add_soc_full_for_stores(n, snapshots, full_to_clustered=full_to_clustered, cyclic=cyclic)

    logger.info("Seasonal storage constraints added.")