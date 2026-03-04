# scripts/geo_temporal_clustering/seasonal_storage.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------
# CSV helpers (days_assignment.csv -> day_sequence)
# --------------------------

def _read_days_assignment(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _infer_day_sequence(df: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """
    Returns:
      day_sequence: np.ndarray of length n_days; entry d is rep_day_index for original day d.
      n_days: number of original days.
    Accepts common column names: day_index + rep_day_index.
    """
    cols = {c.lower(): c for c in df.columns}

    day_candidates = ["day_index", "day", "original_day", "day_in_year", "d"]
    rep_candidates = ["rep_day_index", "rep_day", "representative_day", "cluster_day", "rep", "representative"]

    day_col = next((cols[c] for c in day_candidates if c in cols), None)
    rep_col = next((cols[c] for c in rep_candidates if c in cols), None)

    if day_col is None or rep_col is None:
        raise ValueError(
            f"days_assignment.csv columns not recognized. "
            f"Need day_index and rep_day_index (or equivalents). Got: {list(df.columns)}"
        )

    tmp = df[[day_col, rep_col]].copy().sort_values(day_col)
    day_vals = tmp[day_col].to_numpy()
    rep_vals = tmp[rep_col].to_numpy()

    # Normalize to 0-based if day indexing starts at 1
    if day_vals.min() == 1 and np.all(np.unique(day_vals) == np.arange(1, day_vals.max() + 1)):
        day_vals = day_vals - 1

    n_days = int(day_vals.max() + 1)
    day_sequence = np.empty(n_days, dtype=int)
    day_sequence[day_vals.astype(int)] = rep_vals.astype(int)
    return day_sequence, n_days


# --------------------------
# Model helpers (match the toy example logic)
# --------------------------

def _remove_constraints_if_exist(model, names: Iterable[str]) -> List[str]:
    """
    Exactly like the toy example:
      if cname in m.constraints: m.remove_constraints(cname)
    """
    removed = []
    for cname in names:
        try:
            if cname in model.constraints:
                model.remove_constraints(cname)
                removed.append(cname)
        except Exception as e:
            logger.warning(f"Could not remove constraint '{cname}': {e}")
    return removed


def _get_required_var(model, name: str):
    if name not in model.variables:
        raise KeyError(f"Expected '{name}' not found in model variables.")
    return model.variables[name]


def _squeeze_optional_dims(var, extras=("period", "timestep")):
    """
    Toy example removes optional dims period/timestep by selecting index 0.
    Works for linopy Variables which support .dims and .isel.
    """
    for extra in extras:
        if hasattr(var, "dims") and extra in var.dims:
            var = var.isel({extra: 0})
    return var


def _find_dim(var, candidates: Iterable[str]) -> str:
    """
    Robust dimension detection across PyPSA/linopy versions.

    In some versions, component dimensions are called 'name' (e.g. dims=('snapshot','name')),
    not 'StorageUnit' or 'Link'. This function:
      1) prefers the unique non-snapshot dimension if there is exactly one
      2) otherwise tries to match by candidate names (case-insensitive)
    """
    if not hasattr(var, "dims"):
        raise KeyError("Variable has no .dims; cannot infer dimension names.")

    dims = list(var.dims)

    # Prefer the unique non-snapshot dimension (common case: ('snapshot','name'))
    non_snap = [d for d in dims if d.lower() not in ("snapshot", "snapshots")]
    if len(non_snap) == 1:
        return non_snap[0]

    # Otherwise try candidates
    cand_lower = {c.lower() for c in candidates}
    for d in dims:
        if d.lower() in cand_lower:
            return d

    raise KeyError(f"Cannot find expected dim among {list(candidates)} in dims={dims}")


def _map_full_hour_to_clustered_snapshot(n_snapshots: pd.Index, rep_day_index: int, hour: int):
    """
    Map (rep_day_index, hour) to the snapshot label used in the clustered network.
    - If snapshots is MultiIndex: return (rep_day_index, hour) tuple (plus any constant fillers if needed).
    - If snapshots is DatetimeIndex: construct timestamp base + rep_day_index days + hour hours.
    """
    if isinstance(n_snapshots, pd.MultiIndex):
        # Most common GT format is 2-level (rep_day, hour). If more levels exist, fill with first value.
        if n_snapshots.nlevels == 2:
            return (int(rep_day_index), int(hour))

        # Fill remaining levels with first level value (best-effort)
        tup = [int(rep_day_index), int(hour)]
        for lv in range(2, n_snapshots.nlevels):
            tup.append(n_snapshots.levels[lv][0])
        return tuple(tup)

    if isinstance(n_snapshots, pd.DatetimeIndex):
        year = int(n_snapshots.min().year)
        base = pd.Timestamp(f"{year}-01-01 00:00:00")
        return base + pd.Timedelta(days=int(rep_day_index), hours=int(hour))

    # Fallback (rare): if snapshots are strings, you must adapt formatting to your case
    return f"{int(rep_day_index)}-{int(hour)}"


# --------------------------
# Main entrypoint called from solve_network.py extra_functionality
# --------------------------

def add_seasonal_storage_constraints(n, snapshots, snakemake) -> None:
    """
    Implements the SAME LOGIC as the toy example, but with:
      - day_sequence read from days_assignment.csv
      - snapshot mapping compatible with GT clustered snapshots (MultiIndex or DatetimeIndex)

    Adds:
      - soc_full_su(StorageUnit, t=0..T)
      - soc_full_store(Store, t=0..T)
    """
    if not hasattr(snakemake, "input") or not hasattr(snakemake.input, "days_assignment"):
        raise ValueError("snakemake.input.days_assignment missing; cannot build seasonal-storage constraints.")

    # 0) Read day_sequence (chronological mapping original day -> representative day)
    days_path = snakemake.input.days_assignment
    logger.info(f"Reading days_assignment from: {days_path}")
    df = _read_days_assignment(days_path)
    day_sequence, n_days = _infer_day_sequence(df)

    # 1) Global timeline (no duplicate day boundaries)
    # Same as toy: T = N_days*24, time index has T+1 points (including t=0)
    T = int(n_days * 24)
    t_index = pd.Index(range(T + 1), name="t")

    # 2) Model handle
    m = n.model

    # 3) Optionally read cyclic flag from config
    use_year_cyclic = False
    try:
        use_year_cyclic = bool(
            snakemake.config.get("geotemporal", {}).get("seasonal_storage", {}).get("cyclic", False)
        )
    except Exception:
        use_year_cyclic = False

    # ------------------------------------------------------------------
    # 1) STORAGEUNITS: remove built-in SOC constraints and impose expanded SOC
    # ------------------------------------------------------------------
    if not n.storage_units.empty:
        removed = _remove_constraints_if_exist(
            m,
            [
                "StorageUnit-energy_balance",
                "StorageUnit-fix-state_of_charge-lower",
                "StorageUnit-fix-state_of_charge-upper",
            ],
        )
        logger.info(f"Removed StorageUnit SOC constraints: {removed if removed else '(none found)'}")

        # get variables (same names as toy)
        p_store = _get_required_var(m, "StorageUnit-p_store")
        p_disp = _get_required_var(m, "StorageUnit-p_dispatch")

        p_store = _squeeze_optional_dims(p_store)
        p_disp = _squeeze_optional_dims(p_disp)

        su_dim = _find_dim(p_store, ["StorageUnit", "storageunit", "storage_units", "name"])
        su_names = list(n.storage_units.index)

        # Bounds from p_nom*max_hours like toy
        Emax = (n.storage_units.p_nom * n.storage_units.max_hours).astype(float).reindex(su_names)
        Emax = Emax.fillna(1e9)

        # Create expanded SOC variable (lower=0, upper=Emax) across t=0..T
        # In linopy you can pass scalars; we keep it simple (like toy) by using lower=0 and upper via constraints.
        soc_full_su = m.add_variables(
            lower=0,
            name="soc_full_su",
            coords=[("StorageUnit", su_names), ("t", t_index)],
        )

        # Upper bound constraints (soc <= Emax) for each su and time
        for su in su_names:
            m.add_constraints(
                soc_full_su.sel(StorageUnit=su, t=t_index) <= float(Emax.loc[su]),
                name=f"soc_full_su_upper_{su}",
            )

        # Initial SOC constraints (toy logic: if <=1 treat as fraction else absolute)
        for su in su_names:
            init = np.nan
            if "state_of_charge_initial" in n.storage_units.columns:
                init = n.storage_units.at[su, "state_of_charge_initial"]
            if pd.isna(init):
                init_val = 0.5 * float(Emax.loc[su])
            else:
                init = float(init)
                init_val = init * float(Emax.loc[su]) if init <= 1.0 else init

            m.add_constraints(
                soc_full_su.sel(StorageUnit=su, t=0) == init_val,
                name=f"soc_full_su_init_{su}",
            )

        # Recursion constraints across full horizon
        snap_index = n.snapshots
        for su in su_names:
            eta_ch = float(n.storage_units.at[su, "efficiency_store"]) if "efficiency_store" in n.storage_units.columns else 1.0
            eta_dis = float(n.storage_units.at[su, "efficiency_dispatch"]) if "efficiency_dispatch" in n.storage_units.columns else 1.0

            p_store_su = p_store.sel({su_dim: su})
            p_disp_su = p_disp.sel({su_dim: su})

            for k in range(T):
                d = k // 24
                hh = k % 24
                rep_day = int(day_sequence[d])
                rep = _map_full_hour_to_clustered_snapshot(snap_index, rep_day, hh)

                m.add_constraints(
                    soc_full_su.sel(StorageUnit=su, t=k + 1)
                    == soc_full_su.sel(StorageUnit=su, t=k)
                    + eta_ch * p_store_su.sel(snapshot=rep)
                    - (1.0 / eta_dis) * p_disp_su.sel(snapshot=rep),
                    name=f"soc_full_su_balance_{su}_{k}",
                )

            if use_year_cyclic:
                m.add_constraints(
                    soc_full_su.sel(StorageUnit=su, t=T) == soc_full_su.sel(StorageUnit=su, t=0),
                    name=f"soc_full_su_cyclic_{su}",
                )
    else:
        logger.info("No StorageUnits: skipping StorageUnit expanded SOC.")

    # ------------------------------------------------------------------
    # 2) STORES (+ LINKS): remove Store energy balance and impose expanded SOC
    # ------------------------------------------------------------------
    if not n.stores.empty:
        removed = _remove_constraints_if_exist(
            m,
            [
                "Store-energy_balance",
                "Store-fix-e-lower",
                "Store-fix-e-upper",
            ],
        )
        logger.info(f"Removed Store SOC constraints: {removed if removed else '(none found)'}")

        # Link power variable (same as toy)
        link_p = _get_required_var(m, "Link-p")
        link_p = _squeeze_optional_dims(link_p)

        link_dim = _find_dim(link_p, ["Link", "link", "links", "name"])
        store_names = list(n.stores.index)

        # Expanded Store SOC variable
        soc_full_store = m.add_variables(
            lower=0,
            name="soc_full_store",
            coords=[("Store", store_names), ("t", t_index)],
        )

        # Upper bounds e_nom
        for s in store_names:
            e_nom = n.stores.at[s, "e_nom"] if "e_nom" in n.stores.columns else np.nan
            e_max = float(e_nom) if pd.notna(e_nom) else 1e9
            m.add_constraints(
                soc_full_store.sel(Store=s, t=t_index) <= e_max,
                name=f"soc_full_store_upper_{s}",
            )

        # Build mapping store -> (charge_links, discharge_links) based on store bus (same as toy)
        snap_index = n.snapshots
        for s in store_names:
            sbus = n.stores.at[s, "bus"]

            charge_links = n.links.index[n.links.bus1 == sbus].tolist()
            discharge_links = n.links.index[n.links.bus0 == sbus].tolist()

            charge_links = [l for l in charge_links if n.links.at[l, "bus0"] != sbus]
            discharge_links = [l for l in discharge_links if n.links.at[l, "bus1"] != sbus]

            if len(charge_links) == 0 and len(discharge_links) == 0:
                logger.info(f"Store '{s}' has no charge/discharge links detected -> skipping.")
                continue

            # Initial energy
            e_init = n.stores.at[s, "e_initial"] if "e_initial" in n.stores.columns else np.nan
            if pd.isna(e_init):
                e_nom = n.stores.at[s, "e_nom"] if "e_nom" in n.stores.columns else np.nan
                e_init_val = 0.5 * float(e_nom) if pd.notna(e_nom) else 0.0
            else:
                e_init_val = float(e_init)

            m.add_constraints(
                soc_full_store.sel(Store=s, t=0) == e_init_val,
                name=f"soc_full_store_init_{s}",
            )

            # Recursion
            for k in range(T):
                d = k // 24
                hh = k % 24
                rep_day = int(day_sequence[d])
                rep = _map_full_hour_to_clustered_snapshot(snap_index, rep_day, hh)

                # charging: + eff * p_link
                expr_charge = 0
                for l in charge_links:
                    eff = float(n.links.at[l, "efficiency"]) if "efficiency" in n.links.columns else 1.0
                    expr_charge = expr_charge + eff * link_p.sel({link_dim: l}).sel(snapshot=rep)

                # discharging: - p_link (power leaving store bus)
                expr_dis = 0
                for l in discharge_links:
                    expr_dis = expr_dis + link_p.sel({link_dim: l}).sel(snapshot=rep)

                m.add_constraints(
                    soc_full_store.sel(Store=s, t=k + 1)
                    == soc_full_store.sel(Store=s, t=k)
                    + expr_charge
                    - expr_dis,
                    name=f"soc_full_store_balance_{s}_{k}",
                )

            if use_year_cyclic:
                m.add_constraints(
                    soc_full_store.sel(Store=s, t=T) == soc_full_store.sel(Store=s, t=0),
                    name=f"soc_full_store_cyclic_{s}",
                )
    else:
        logger.info("No Stores: skipping Store expanded SOC.")

    logger.info("Seasonal storage constraints (toy-style) added successfully.")