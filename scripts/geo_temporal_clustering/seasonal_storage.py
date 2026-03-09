# scripts/geo_temporal_clustering/seasonal_storage.py

import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import xarray as xr
import time

logger = logging.getLogger(__name__)


# =============================================================================
# CSV helpers
# =============================================================================

def _read_days_assignment(path: str | Path) -> pd.DataFrame:
    """Read days_assignment.csv with stripped column names."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _infer_day_sequence(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    """
    Build the chronological mapping original_day -> representative_day_label.

    Expected columns include:
    - day_index
    - rep_day_index

    Returns
    -------
    day_sequence : np.ndarray
        Length = number of original days.
        day_sequence[d] gives the representative day label for original day d.
    n_days : int
        Number of original days.
    """
    cols = {c.lower(): c for c in df.columns}

    day_candidates = ["day_index", "day", "original_day", "day_in_year", "d"]
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
            "Need day_index and rep_day_index (or equivalents). "
            f"Got: {list(df.columns)}"
        )

    tmp = df[[day_col, rep_col]].copy().sort_values(day_col)
    day_vals = tmp[day_col].to_numpy()
    rep_vals = tmp[rep_col].to_numpy()

    # Normalize day index to 0-based if needed
    if day_vals.min() == 1 and np.all(np.unique(day_vals) == np.arange(1, day_vals.max() + 1)):
        day_vals = day_vals - 1

    n_days = int(day_vals.max() + 1)
    day_sequence = np.empty(n_days, dtype=int)
    day_sequence[day_vals.astype(int)] = rep_vals.astype(int)

    return day_sequence, n_days


# =============================================================================
# Linopy / model helpers
# =============================================================================

def _remove_constraints_if_exist(model, names):
    removed = []
    for name in names:
        try:
            if name in model.constraints:
                model.remove_constraints(name)
                removed.append(name)
        except Exception as exc:
            logger.warning("Could not remove constraint '%s': %s", name, exc)
    return removed


def _remove_variables_if_exist(model, names):
    removed = []
    for name in names:
        try:
            if name in model.variables:
                model.remove_variables(name)
                removed.append(name)
        except Exception as exc:
            logger.warning("Could not remove variable '%s': %s", name, exc)
    return removed


def _get_required_var(model, name: str):
    """Return a model variable or raise a clear error."""
    if name not in model.variables:
        raise KeyError(f"Expected variable '{name}' not found in model.")
    return model.variables[name]


def _get_optional_var(model, name: str):
    """Return a model variable if present, else None."""
    return model.variables[name] if name in model.variables else None


def _squeeze_optional_dims(var, extras=("period", "timestep")):
    """
    Drop optional singleton dims if present.

    This mirrors the practical behavior needed across different PyPSA / linopy versions.
    """
    for extra in extras:
        if hasattr(var, "dims") and extra in var.dims:
            var = var.isel({extra: 0})
    return var


def _find_dim(var, candidates: Iterable[str]) -> str:
    """
    Robust component-dimension detection across PyPSA / linopy versions.
    """
    if not hasattr(var, "dims"):
        raise KeyError("Variable has no .dims; cannot infer dimension names.")

    dims = list(var.dims)

    non_snap = [d for d in dims if d.lower() not in ("snapshot", "snapshots")]
    if len(non_snap) == 1:
        return non_snap[0]

    cand_lower = {c.lower() for c in candidates}
    for d in dims:
        if d.lower() in cand_lower:
            return d

    raise KeyError(f"Cannot find expected dim among {list(candidates)} in dims={dims}")


def _rename_and_reindex_time(obj, old_dim: str, new_dim: str, new_index: pd.Index):
    """
    Rename a time-like dimension and force coordinates to a new index.

    Works on linopy Variables / LinearExpressions backed by xarray-like objects.
    """
    out = obj.rename({old_dim: new_dim})
    if hasattr(out, "assign_coords"):
        out = out.assign_coords({new_dim: new_index})
    return out


# =============================================================================
# Snapshot mapping helpers
# =============================================================================

def _map_full_hour_to_clustered_snapshot(
    n_snapshots: pd.Index,
    rep_day_label: int,
    hour: int,
):
    """
    Map (representative_day_label, hour) to the snapshot label used in the clustered network.

    Supported cases:
    - MultiIndex with first level = representative day label, second level = hour
    - DatetimeIndex (fallback)
    - string-like labels (last-resort fallback)
    """
    if isinstance(n_snapshots, pd.MultiIndex):
        if n_snapshots.nlevels == 2:
            return (int(rep_day_label), int(hour))

        # Best-effort fallback for >2 levels
        tpl = [int(rep_day_label), int(hour)]
        for lv in range(2, n_snapshots.nlevels):
            tpl.append(n_snapshots.levels[lv][0])
        return tuple(tpl)

    if isinstance(n_snapshots, pd.DatetimeIndex):
        year = int(n_snapshots.min().year)
        base = pd.Timestamp(f"{year}-01-01 00:00:00")
        return base + pd.Timedelta(days=int(rep_day_label), hours=int(hour))

    return f"{int(rep_day_label)}-{int(hour)}"


def _build_full_representative_snapshot_index(
    n_snapshots: pd.Index,
    day_sequence: np.ndarray,
) -> pd.Index:
    """
    Build the annual chronological snapshot index of length T = n_days * 24,
    where each hour points to the corresponding representative snapshot.
    """
    full_labels = []
    n_days = len(day_sequence)

    for d in range(n_days):
        rep_day_label = int(day_sequence[d])
        for hour in range(24):
            full_labels.append(
                _map_full_hour_to_clustered_snapshot(
                    n_snapshots=n_snapshots,
                    rep_day_label=rep_day_label,
                    hour=hour,
                )
            )

    if isinstance(n_snapshots, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(full_labels, names=n_snapshots.names)

    return pd.Index(full_labels, name=n_snapshots.name)


def _validate_snapshot_mapping(n_snapshots: pd.Index, rep_snapshots_full: pd.Index) -> None:
    """
    Validate that all reconstructed representative snapshots exist in the clustered network.
    """
    missing = rep_snapshots_full.difference(n_snapshots)
    if len(missing) > 0:
        sample = list(missing[:10])
        raise KeyError(
            "Some reconstructed representative snapshots are not present in n.snapshots. "
            f"First missing entries: {sample}"
        )


# =============================================================================
# Data / variable expansion helpers
# =============================================================================

def _expand_variable_over_full_timeline(
    var,
    component_dim: str,
    component_name: str,
    component_index: pd.Index,
    rep_snapshots_full: pd.Index,
    step_index: pd.Index,
):
    """
    Select a snapshot-based model variable over the full chronological timeline
    and rename dims to (step, component_name).
    """
    out = var.sel(snapshot=rep_snapshots_full)

    rename_dict = {"snapshot": "step"}
    if component_dim != component_name:
        rename_dict[component_dim] = component_name

    out = out.rename(rename_dict)

    if hasattr(out, "assign_coords"):
        out = out.assign_coords(
            step=step_index,
            **{component_name: component_index},
        )

    return out


def _expand_dataframe_over_full_timeline(
    df: pd.DataFrame,
    rep_snapshots_full: pd.Index,
    component_name: str,
    component_index: pd.Index,
    step_index: pd.Index,
    fill_value: float = 0.0,
) -> xr.DataArray:
    """
    Expand a snapshot-indexed pandas DataFrame to the full chronological timeline.
    """
    if df.empty:
        values = np.zeros((len(step_index), len(component_index)), dtype=float)
        return xr.DataArray(
            values,
            dims=("step", component_name),
            coords={"step": step_index, component_name: component_index},
        )

    tmp = df.reindex(index=rep_snapshots_full, columns=component_index, fill_value=fill_value)
    return xr.DataArray(
        tmp.to_numpy(dtype=float),
        dims=("step", component_name),
        coords={"step": step_index, component_name: component_index},
    )


def _expand_series_to_dataarray(
    s: pd.Series,
    dim_name: str,
    index: pd.Index,
    fill_value: float = 0.0,
) -> xr.DataArray:
    """
    Convert a pandas Series to an xarray DataArray on a target index.
    """
    tmp = s.reindex(index).fillna(fill_value).astype(float)
    return xr.DataArray(tmp.to_numpy(), dims=(dim_name,), coords={dim_name: index})


# =============================================================================
# StorageUnit block
# =============================================================================

def _add_storageunit_seasonal_constraints(
    n,
    m,
    rep_snapshots_full: pd.Index,
    t_index: pd.Index,
    step_index: pd.Index,
    use_year_cyclic: bool,
) -> None:
    """
    Replace StorageUnit SOC dynamics with an annual chronological SOC
    while keeping power variables on representative snapshots.
    """
    if n.storage_units.empty:
        logger.info("No StorageUnits found. Skipping StorageUnit seasonal storage block.")
        return

    removed_cons = _remove_constraints_if_exist(
        m,
        [
            "StorageUnit-energy_balance",
            "StorageUnit-fix-state_of_charge-lower",
            "StorageUnit-fix-state_of_charge-upper",
        ],
    )
    logger.info("Removed StorageUnit constraints: %s", removed_cons if removed_cons else "(none)")

    removed_vars = _remove_variables_if_exist(
        m,
        [
            "StorageUnit-state_of_charge",
        ],
    )
    logger.info("Removed StorageUnit variables: %s", removed_vars if removed_vars else "(none)")

    p_store = _squeeze_optional_dims(_get_required_var(m, "StorageUnit-p_store"))
    p_disp = _squeeze_optional_dims(_get_required_var(m, "StorageUnit-p_dispatch"))
    spill = _get_optional_var(m, "StorageUnit-spill")
    spill = _squeeze_optional_dims(spill) if spill is not None else None

    su_dim = _find_dim(p_store, ["StorageUnit", "storageunit", "storage_units", "name"])
    su_names = pd.Index(n.storage_units.index, name="StorageUnit")

    # New annual SOC variable
    soc = m.add_variables(
        lower=0,
        name="StorageUnit-soc_full",
        coords=[su_names, t_index],
    )

    # -------------------------------------------------------------------------
    # Capacity upper bounds: fixed and extendable separately
    # -------------------------------------------------------------------------
    max_hours = _expand_series_to_dataarray(
        n.storage_units["max_hours"],
        dim_name="StorageUnit",
        index=su_names,
        fill_value=0.0,
    )

    ext_mask = n.storage_units["p_nom_extendable"].reindex(su_names).fillna(False)
    fix_su = su_names[~ext_mask]
    ext_su = su_names[ext_mask]

    if len(fix_su) > 0:
        emax_fix = (
            n.storage_units.loc[fix_su, "p_nom"].astype(float)
            * n.storage_units.loc[fix_su, "max_hours"].astype(float)
        )
        emax_fix = xr.DataArray(
            emax_fix.to_numpy(),
            dims=("StorageUnit",),
            coords={"StorageUnit": fix_su},
        )
        m.add_constraints(
            soc.sel(StorageUnit=fix_su) <= emax_fix,
            name="StorageUnit-soc_full-upper-fix",
        )

    p_nom_var = _get_optional_var(m, "StorageUnit-p_nom")
    if len(ext_su) > 0:
        if p_nom_var is None:
            raise KeyError(
                "Found extendable StorageUnits but model variable 'StorageUnit-p_nom' is missing."
            )

        p_nom_var = _squeeze_optional_dims(p_nom_var)
        p_nom_dim = _find_dim(p_nom_var, ["StorageUnit", "storageunit", "storage_units", "name"])
        p_nom_ext = p_nom_var.sel({p_nom_dim: ext_su})
        if p_nom_dim != "StorageUnit":
            p_nom_ext = p_nom_ext.rename({p_nom_dim: "StorageUnit"})
        if hasattr(p_nom_ext, "assign_coords"):
            p_nom_ext = p_nom_ext.assign_coords(StorageUnit=ext_su)

        emax_ext = p_nom_ext * max_hours.sel(StorageUnit=ext_su)
        m.add_constraints(
            soc.sel(StorageUnit=ext_su) <= emax_ext,
            name="StorageUnit-soc_full-upper-ext",
        )

    # -------------------------------------------------------------------------
    # Initial SOC
    # -------------------------------------------------------------------------
    if "state_of_charge_initial" in n.storage_units.columns:
        init_raw = n.storage_units["state_of_charge_initial"].reindex(su_names)
    else:
        init_raw = pd.Series(np.nan, index=su_names)

    # Fixed units
    if len(fix_su) > 0:
        emax_fix_series = (
            n.storage_units.loc[fix_su, "p_nom"].astype(float)
            * n.storage_units.loc[fix_su, "max_hours"].astype(float)
        )
        init_fix = init_raw.loc[fix_su].copy()

        init_fix_abs = pd.Series(index=fix_su, dtype=float)
        for su in fix_su:
            val = init_fix.loc[su]
            if pd.isna(val):
                init_fix_abs.loc[su] = 0.5 * emax_fix_series.loc[su]
            else:
                val = float(val)
                init_fix_abs.loc[su] = val * emax_fix_series.loc[su] if val <= 1.0 else val

        init_fix_da = xr.DataArray(
            init_fix_abs.to_numpy(),
            dims=("StorageUnit",),
            coords={"StorageUnit": fix_su},
        )

        m.add_constraints(
            soc.sel(StorageUnit=fix_su, t=0) == init_fix_da,
            name="StorageUnit-soc_full-init-fix",
        )

    # Extendable units
    if len(ext_su) > 0:
        init_ext = init_raw.loc[ext_su].copy()

        frac_idx = init_ext.index[init_ext.notna() & (init_ext.astype(float) <= 1.0)]
        abs_idx = init_ext.index[init_ext.notna() & (init_ext.astype(float) > 1.0)]
        nan_idx = init_ext.index[init_ext.isna()]

        if len(frac_idx) > 0:
            frac = xr.DataArray(
                init_ext.loc[frac_idx].astype(float).to_numpy(),
                dims=("StorageUnit",),
                coords={"StorageUnit": frac_idx},
            )
            p_nom_frac = p_nom_ext.sel(StorageUnit=frac_idx)
            rhs = frac * p_nom_frac * max_hours.sel(StorageUnit=frac_idx)
            m.add_constraints(
                soc.sel(StorageUnit=frac_idx, t=0) == rhs,
                name="StorageUnit-soc_full-init-ext-frac",
            )

        if len(abs_idx) > 0:
            rhs = xr.DataArray(
                init_ext.loc[abs_idx].astype(float).to_numpy(),
                dims=("StorageUnit",),
                coords={"StorageUnit": abs_idx},
            )
            m.add_constraints(
                soc.sel(StorageUnit=abs_idx, t=0) == rhs,
                name="StorageUnit-soc_full-init-ext-abs",
            )

        if len(nan_idx) > 0:
            rhs = 0.5 * p_nom_ext.sel(StorageUnit=nan_idx) * max_hours.sel(StorageUnit=nan_idx)
            m.add_constraints(
                soc.sel(StorageUnit=nan_idx, t=0) == rhs,
                name="StorageUnit-soc_full-init-ext-default",
            )

    # -------------------------------------------------------------------------
    # Expanded full-year variables / series
    # -------------------------------------------------------------------------
    p_store_full = _expand_variable_over_full_timeline(
        var=p_store,
        component_dim=su_dim,
        component_name="StorageUnit",
        component_index=su_names,
        rep_snapshots_full=rep_snapshots_full,
        step_index=step_index,
    )

    p_disp_full = _expand_variable_over_full_timeline(
        var=p_disp,
        component_dim=su_dim,
        component_name="StorageUnit",
        component_index=su_names,
        rep_snapshots_full=rep_snapshots_full,
        step_index=step_index,
    )

    if spill is not None:
        spill_dim = _find_dim(spill, ["StorageUnit", "storageunit", "storage_units", "name"])
        spill_full = _expand_variable_over_full_timeline(
            var=spill,
            component_dim=spill_dim,
            component_name="StorageUnit",
            component_index=su_names,
            rep_snapshots_full=rep_snapshots_full,
            step_index=step_index,
        )
    else:
        spill_full = xr.DataArray(
            np.zeros((len(step_index), len(su_names)), dtype=float),
            dims=("step", "StorageUnit"),
            coords={"step": step_index, "StorageUnit": su_names},
        )

    if hasattr(n, "storage_units_t") and hasattr(n.storage_units_t, "inflow"):
        inflow_full = _expand_dataframe_over_full_timeline(
            df=n.storage_units_t.inflow,
            rep_snapshots_full=rep_snapshots_full,
            component_name="StorageUnit",
            component_index=su_names,
            step_index=step_index,
            fill_value=0.0,
        )
    else:
        inflow_full = xr.DataArray(
            np.zeros((len(step_index), len(su_names)), dtype=float),
            dims=("step", "StorageUnit"),
            coords={"step": step_index, "StorageUnit": su_names},
        )

    eta_store = _expand_series_to_dataarray(
        n.storage_units.get("efficiency_store", pd.Series(1.0, index=su_names)),
        dim_name="StorageUnit",
        index=su_names,
        fill_value=1.0,
    )
    eta_dispatch = _expand_series_to_dataarray(
        n.storage_units.get("efficiency_dispatch", pd.Series(1.0, index=su_names)),
        dim_name="StorageUnit",
        index=su_names,
        fill_value=1.0,
    )

    standing_loss = n.storage_units.get("standing_loss", pd.Series(0.0, index=su_names))
    eta_stand = _expand_series_to_dataarray(
        1.0 - standing_loss.astype(float),
        dim_name="StorageUnit",
        index=su_names,
        fill_value=1.0,
    )

    # -------------------------------------------------------------------------
    # Annual chronological balance
    # -------------------------------------------------------------------------
    soc_prev = soc.isel(t=slice(0, len(step_index))).rename({"t": "step"})
    soc_next = soc.isel(t=slice(1, len(step_index) + 1)).rename({"t": "step"})

    if hasattr(soc_prev, "assign_coords"):
        soc_prev = soc_prev.assign_coords(step=step_index)
    if hasattr(soc_next, "assign_coords"):
        soc_next = soc_next.assign_coords(step=step_index)

    rhs = (
        eta_stand * soc_prev
        + eta_store * p_store_full
        - (1.0 / eta_dispatch) * p_disp_full
        + inflow_full
        - spill_full
    )

    m.add_constraints(
        soc_next == rhs,
        name="StorageUnit-soc_full-balance",
    )

    # -------------------------------------------------------------------------
    # Optional annual cyclicity
    # -------------------------------------------------------------------------
    if use_year_cyclic:
        m.add_constraints(
            soc.sel(t=t_index[-1]) == soc.sel(t=t_index[0]),
            name="StorageUnit-soc_full-cyclic",
        )

    logger.info("Added vectorized seasonal-storage constraints for StorageUnits.")


# =============================================================================
# Store block
# =============================================================================


def _add_store_seasonal_constraints(
    n,
    m,
    rep_snapshots_full: pd.Index,
    t_index: pd.Index,
    step_index: pd.Index,
    use_year_cyclic: bool,
) -> None:
    """
    Replace Store energy balance with an annual chronological energy trajectory
    while keeping Store-p on representative snapshots.

    Important:
    - the native PyPSA Store balance uses Store-p, not Link-p
    - Store-p remains on clustered representative snapshots
    - Store-e_full is the only annual full-length energy variable
    """
    if n.stores.empty:
        logger.info("No Stores found. Skipping Store seasonal storage block.")
        return

    removed_cons = _remove_constraints_if_exist(
        m,
        [
            "Store-energy_balance",
            "Store-fix-e-lower",
            "Store-fix-e-upper",
            "Store-ext-e-lower",
            "Store-ext-e-upper",
        ],
    )
    logger.info("Removed Store constraints: %s", removed_cons if removed_cons else "(none)")

    removed_vars = _remove_variables_if_exist(
        m,
        [
            "Store-e",
        ],
    )
    logger.info("Removed Store variables: %s", removed_vars if removed_vars else "(none)")

    store_p = _squeeze_optional_dims(_get_required_var(m, "Store-p"))
    store_p_dim = _find_dim(store_p, ["Store", "store", "stores", "name"])

    store_names = pd.Index(n.stores.index, name="Store")
    store_ext_mask = n.stores["e_nom_extendable"].reindex(store_names).fillna(False)
    fix_store = store_names[~store_ext_mask]
    ext_store = store_names[store_ext_mask]

    # New annual energy variable
    e = m.add_variables(
        lower=0,
        name="Store-e_full",
        coords=[store_names, t_index],
    )

    # -------------------------------------------------------------------------
    # Capacity upper bounds: fixed and extendable separately
    # -------------------------------------------------------------------------
    if len(fix_store) > 0:
        e_nom_fix = xr.DataArray(
            n.stores.loc[fix_store, "e_nom"].astype(float).to_numpy(),
            dims=("Store",),
            coords={"Store": fix_store},
        )
        m.add_constraints(
            e.sel(Store=fix_store) <= e_nom_fix,
            name="Store-e_full-upper-fix",
        )

    e_nom_var = _get_optional_var(m, "Store-e_nom")
    e_nom_ext = None
    if len(ext_store) > 0:
        if e_nom_var is None:
            raise KeyError(
                "Found extendable Stores but model variable 'Store-e_nom' is missing."
            )

        e_nom_var = _squeeze_optional_dims(e_nom_var)
        e_nom_dim = _find_dim(e_nom_var, ["Store", "store", "stores", "name"])
        e_nom_ext = e_nom_var.sel({e_nom_dim: ext_store})
        if e_nom_dim != "Store":
            e_nom_ext = e_nom_ext.rename({e_nom_dim: "Store"})
        if hasattr(e_nom_ext, "assign_coords"):
            e_nom_ext = e_nom_ext.assign_coords(Store=ext_store)

        m.add_constraints(
            e.sel(Store=ext_store) <= e_nom_ext,
            name="Store-e_full-upper-ext",
        )

    # -------------------------------------------------------------------------
    # Initial energy
    # -------------------------------------------------------------------------
    if "e_initial" in n.stores.columns:
        init_raw = n.stores["e_initial"].reindex(store_names)
    else:
        init_raw = pd.Series(np.nan, index=store_names)

    if len(fix_store) > 0:
        e_nom_fix_series = n.stores.loc[fix_store, "e_nom"].astype(float)
        init_fix = init_raw.loc[fix_store].copy()

        init_fix_abs = pd.Series(index=fix_store, dtype=float)
        for s in fix_store:
            val = init_fix.loc[s]
            if pd.isna(val):
                init_fix_abs.loc[s] = 0.5 * e_nom_fix_series.loc[s]
            else:
                init_fix_abs.loc[s] = float(val)

        rhs = xr.DataArray(
            init_fix_abs.to_numpy(),
            dims=("Store",),
            coords={"Store": fix_store},
        )
        m.add_constraints(
            e.sel(Store=fix_store, t=0) == rhs,
            name="Store-e_full-init-fix",
        )

    if len(ext_store) > 0:
        init_ext = init_raw.loc[ext_store].copy()

        abs_idx = init_ext.index[init_ext.notna()]
        nan_idx = init_ext.index[init_ext.isna()]

        if len(abs_idx) > 0:
            rhs = xr.DataArray(
                init_ext.loc[abs_idx].astype(float).to_numpy(),
                dims=("Store",),
                coords={"Store": abs_idx},
            )
            m.add_constraints(
                e.sel(Store=abs_idx, t=0) == rhs,
                name="Store-e_full-init-ext-abs",
            )

        if len(nan_idx) > 0:
            rhs = 0.5 * e_nom_ext.sel(Store=nan_idx)
            m.add_constraints(
                e.sel(Store=nan_idx, t=0) == rhs,
                name="Store-e_full-init-ext-default",
            )

    # -------------------------------------------------------------------------
    # Expanded Store-p on the full annual timeline
    # -------------------------------------------------------------------------
    store_p_full = _expand_variable_over_full_timeline(
        var=store_p,
        component_dim=store_p_dim,
        component_name="Store",
        component_index=store_names,
        rep_snapshots_full=rep_snapshots_full,
        step_index=step_index,
    )

    # Standing losses for Store
    standing_loss = n.stores.get("standing_loss", pd.Series(0.0, index=store_names))
    eta_stand = _expand_series_to_dataarray(
        1.0 - standing_loss.astype(float),
        dim_name="Store",
        index=store_names,
        fill_value=1.0,
    )

    # -------------------------------------------------------------------------
    # Annual chronological balance
    #
    # Native PyPSA Store balance is:
    #   e_t = eta_stand * e_{t-1} - p_t
    #
    # where Store-p > 0 means energy leaves the store.
    # -------------------------------------------------------------------------
    e_prev = e.isel(t=slice(0, len(step_index))).rename({"t": "step"})
    e_next = e.isel(t=slice(1, len(step_index) + 1)).rename({"t": "step"})

    if hasattr(e_prev, "assign_coords"):
        e_prev = e_prev.assign_coords(step=step_index)
    if hasattr(e_next, "assign_coords"):
        e_next = e_next.assign_coords(step=step_index)

    rhs = eta_stand * e_prev - store_p_full

    m.add_constraints(
        e_next == rhs,
        name="Store-e_full-balance",
    )

    # -------------------------------------------------------------------------
    # Optional annual cyclicity
    # -------------------------------------------------------------------------
    if use_year_cyclic:
        m.add_constraints(
            e.sel(t=t_index[-1]) == e.sel(t=t_index[0]),
            name="Store-e_full-cyclic",
        )

    logger.info("Added vectorized seasonal-storage constraints for Stores.")

# =============================================================================
# Public entrypoint
# =============================================================================

def add_seasonal_storage_constraints(n, snapshots, days_assignment_path: str | Path, cyclic: bool = False) -> None:
    """
    Add vectorized seasonal-storage constraints for a geo-temporally clustered network.

    Parameters
    ----------
    n : pypsa.Network
        Clustered PyPSA network with representative snapshots.
    snapshots :
        Unused explicit argument kept for compatibility with PyPSA extra_functionality signature.
    days_assignment_path : str | Path
        Path to days_assignment.csv
    cyclic : bool, default False
        Whether to impose annual cyclicity on the expanded SOC variables.
    """
    t0_total = time.perf_counter()

    logger.info("Reading days_assignment from: %s", days_assignment_path)
    logger.info(
        "Seasonal storage targets: %s StorageUnits, %s Stores",
        len(n.storage_units.index),
        len(n.stores.index),
    )
    df = _read_days_assignment(days_assignment_path)
    day_sequence, n_days = _infer_day_sequence(df)

    T = int(n_days * 24)
    t_index = pd.Index(range(T + 1), name="t")
    step_index = pd.Index(range(T), name="step")

    rep_snapshots_full = _build_full_representative_snapshot_index(n.snapshots, day_sequence)
    _validate_snapshot_mapping(n.snapshots, rep_snapshots_full)

    logger.info(
        "Seasonal storage setup: n_days=%s, T=%s, clustered_snapshots=%s, cyclic=%s",
        n_days,
        T,
        len(n.snapshots),
        cyclic,
    )

    m = n.model

    t0_su = time.perf_counter()
    _add_storageunit_seasonal_constraints(
        n=n,
        m=m,
        rep_snapshots_full=rep_snapshots_full,
        t_index=t_index,
        step_index=step_index,
        use_year_cyclic=cyclic,
    )
    dt_su = time.perf_counter() - t0_su
    logger.info("Seasonal storage: StorageUnit constraints added in %.2f s", dt_su)

    t0_store = time.perf_counter()
    _add_store_seasonal_constraints(
        n=n,
        m=m,
        rep_snapshots_full=rep_snapshots_full,
        t_index=t_index,
        step_index=step_index,
        use_year_cyclic=cyclic,
    )
    dt_store = time.perf_counter() - t0_store
    logger.info("Seasonal storage: Store constraints added in %.2f s", dt_store)

    dt_total = time.perf_counter() - t0_total
    logger.info(
        "Seasonal storage constraints added successfully in %.2f s "
        "(StorageUnits: %.2f s, Stores: %.2f s).",
        dt_total,
        dt_su,
        dt_store,
    )

    n._seasonal_storage_meta = {
    "day_sequence": day_sequence.copy(),
    "n_days": int(n_days),
    "T": int(T),
    "t_index": list(t_index),
    "rep_snapshots_full": rep_snapshots_full,
    "storageunit_var_name": "StorageUnit-soc_full",
    "store_var_name": "Store-e_full",
}