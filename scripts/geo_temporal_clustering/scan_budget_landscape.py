# -*- coding: utf-8 -*-
"""
Standalone geo-temporal clustering scan.

This script loads a pre-clustering PyPSA network and repeatedly runs only the
geo-temporal reducer with different initial (K_nodes, K_days) pairs.

It does not export clustered networks and does not call PyPSA spatial clustering.
Its purpose is to diagnose the reducer landscape and the path dependence of the
budget-based geo-temporal heuristic.
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from pathlib import Path as _Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pypsa


# =============================================================================
# Path setup
# =============================================================================

ROOT = _Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.geo_temporal_clustering.core import (
    AlternatingSpatioTemporalReducer,
    build_tensor_X,
    select_buses_from_loads,
    bus_coords_latlon,
    loads_by_bus_timeseries,
    cf_by_bus_timeseries,
)


# =============================================================================
# USER SETTINGS
# =============================================================================

NETWORK_PATH = Path("/home/pampado/clustering/pypsa-eur/resources/clustering_new_algorithm_tuning/gtb-0.15-900-bal-0.05/networks/base_s_adm_elec_Gt.nc")

OUT_DIR = Path("resources/geotemporal_clustering_scan/beta_opposite_1")

# Main scan parameters
TARGET_BUDGET = 900
MIN_INITIAL_STEPS = 850
MAX_INITIAL_STEPS = 900

# Pair generation mode:
# - "frontier": one pair per K_nodes, with K_days = floor(TARGET_BUDGET / K_nodes)
# - "band": all pairs satisfying MIN_INITIAL_STEPS <= K_nodes * K_days <= MAX_INITIAL_STEPS
PAIR_MODE = "band"

# Optional filters on initial pairs
MIN_INIT_NODES = 1
MAX_INIT_NODES = None  # None means use all available nodes
MIN_INIT_DAYS = 1
MAX_INIT_DAYS = None  # None means use all available days

# Include a run starting from full resolution
RUN_FULL_BASELINE = True

# Seeds to test for each initial pair
RANDOM_STATES = [0]

# Feature extraction
HOURS_PER_DAY = 24
EXCLUDE_BUS_SUBSTRINGS = (" H2", "battery")

FEATURE_MODE = "daily_stats"
STATS = ("mean", "max", "min", "std", "ramp_max", "energy")

INCLUDE_LOAD = True

PV_CARRIER = "solar"
PV_WEIGHT_BY = "p_nom"

WIND_CARRIER = "onwind"
WIND_WEIGHT_BY = "p_nom"

# Objective weights
# Supported forms:
# - exact feature name, e.g. "load_mean"
# - attribute prefix, e.g. "load", "pv_cf", "wind_cf"
FEATURE_WEIGHTS_CFG = {
    "load": 2.0,
    "pv_cf": 1.0,
    "wind_cf": 1.0,
}

# Supported: None, "none", "mean_load", "peak_load"
NODE_WEIGHTS_MODE = None

# Reducer parameters
REDUCER_BASE_CFG = {
    "lambda_ts": 0.05,
    "normalize": "zscore",
    "max_total_steps": TARGET_BUDGET,
    "loss_norm": "l2_squared",
    "beta": 0.05,
    "beta_growth": 1.5,
    "beta_max": 1.0,
    "max_iter": 50,
    "tol_no_change": 7,
    "objective_tol_rel": 1e-5,
    "verbose": False,
    "norm_q": 0.95,
    "use_pca_days": False,
    "pca_days_n_components": 0.95,
    "pca_days_random_state": 0,
    "standardize_day_matrix_cols": False,
    "kmedoids_max_iter": 100,
}


# =============================================================================
# Helpers
# =============================================================================

def build_feature_weights(feature_names: List[str], cfg_weights: Dict[str, float]) -> np.ndarray:
    """
    Build a feature-weight vector from a dictionary.

    Supported keys:
    - exact feature name, e.g. "load_mean"
    - attribute prefix, e.g. "load", "pv_cf", "wind_cf"
    """
    weights = np.ones(len(feature_names), dtype=float)

    if not cfg_weights:
        return weights

    for i, feat in enumerate(feature_names):
        assigned = False

        if feat in cfg_weights:
            weights[i] = float(cfg_weights[feat])
            assigned = True

        if not assigned:
            for prefix, value in cfg_weights.items():
                prefix = str(prefix)
                if feat.startswith(prefix + "_"):
                    weights[i] = float(value)
                    assigned = True
                    break

    return weights


def build_scan_pairs(
    *,
    n_nodes: int,
    n_days: int,
    target_budget: int,
    min_steps: int,
    max_steps: int,
    pair_mode: str,
    min_init_nodes: int = 1,
    max_init_nodes: Optional[int] = None,
    min_init_days: int = 1,
    max_init_days: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Build initial (K_nodes, K_days) pairs for the scan.
    """
    if max_init_nodes is None:
        max_init_nodes = n_nodes
    if max_init_days is None:
        max_init_days = n_days

    max_init_nodes = min(max_init_nodes, n_nodes)
    max_init_days = min(max_init_days, n_days)

    pairs: List[Tuple[int, int]] = []

    if pair_mode == "frontier":
        for kn in range(min_init_nodes, max_init_nodes + 1):
            kd = target_budget // kn
            kd = min(kd, max_init_days)

            if kd < min_init_days:
                continue

            steps = kn * kd
            if min_steps <= steps <= max_steps:
                pairs.append((int(kn), int(kd)))

    elif pair_mode == "band":
        for kn in range(min_init_nodes, max_init_nodes + 1):
            kd_min = max(min_init_days, int(np.ceil(min_steps / kn)))
            kd_max = min(max_init_days, int(np.floor(max_steps / kn)))

            if kd_min > kd_max:
                continue

            for kd in range(kd_min, kd_max + 1):
                pairs.append((int(kn), int(kd)))

    else:
        raise ValueError("pair_mode must be either 'frontier' or 'band'.")

    pairs = sorted(set(pairs), key=lambda x: (x[0] * x[1], x[0], x[1]), reverse=True)
    return pairs

def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """
    Return the first column from candidates that exists in df.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _json_safe_value(value: Any) -> Any:
    """
    Convert numpy/pandas scalar values to JSON-safe Python objects.
    """
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    return value


def enrich_history_with_evaluation_alternatives(
    history: pd.DataFrame,
    evaluations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add compact candidate-alternative information to the reducer history.

    The detailed candidate-level data remain in scan_evaluations.csv.
    This function adds, for each history iteration, a compact summary such as:
    - number of candidates tested;
    - list of candidate pairs;
    - best candidate pair;
    - best candidate objective;
    - rank of the accepted/current pair among candidates, when inferable.

    The function is intentionally tolerant to column-name differences in
    reducer.history and reducer.evaluations.
    """
    if history.empty or evaluations.empty:
        return history

    h = history.copy()
    e = evaluations.copy()

    iter_col_h = _first_existing_column(
        h,
        ["iteration", "iter", "it", "step", "iteration_id"],
    )
    iter_col_e = _first_existing_column(
        e,
        ["iteration", "iter", "it", "step", "iteration_id"],
    )

    cand_nodes_col = _first_existing_column(
        e,
        [
            "candidate_K_nodes",
            "candidate_k_nodes",
            "cand_K_nodes",
            "cand_k_nodes",
            "new_K_nodes",
            "new_k_nodes",
            "K_nodes_new",
            "k_nodes_new",
            "K_nodes",
            "k_nodes",
        ],
    )
    cand_days_col = _first_existing_column(
        e,
        [
            "candidate_K_days",
            "candidate_k_days",
            "cand_K_days",
            "cand_k_days",
            "new_K_days",
            "new_k_days",
            "K_days_new",
            "k_days_new",
            "K_days",
            "k_days",
        ],
    )
    cand_obj_col = _first_existing_column(
        e,
        [
            "candidate_objective",
            "objective_candidate",
            "new_objective",
            "objective_new",
            "objective",
            "loss",
            "score",
        ],
    )

    hist_nodes_col = _first_existing_column(
        h,
        [
            "K_nodes",
            "k_nodes",
            "current_K_nodes",
            "current_k_nodes",
            "final_K_nodes",
            "final_k_nodes",
        ],
    )
    hist_days_col = _first_existing_column(
        h,
        [
            "K_days",
            "k_days",
            "current_K_days",
            "current_k_days",
            "final_K_days",
            "final_k_days",
        ],
    )

    accepted_col = _first_existing_column(
        e,
        ["accepted", "is_accepted", "chosen", "selected"],
    )

    required = [iter_col_h, iter_col_e, cand_nodes_col, cand_days_col]
    if any(col is None for col in required):
        missing = {
            "history_iteration_col": iter_col_h,
            "evaluations_iteration_col": iter_col_e,
            "candidate_nodes_col": cand_nodes_col,
            "candidate_days_col": cand_days_col,
        }
        print(
            ">>> Warning: cannot enrich scan_history.csv with candidate alternatives. "
            f"Missing/inferred columns: {missing}"
        )
        return h

    group_cols_e = ["run_id", iter_col_e] if "run_id" in e.columns else [iter_col_e]
    group_cols_h = ["run_id", iter_col_h] if "run_id" in h.columns else [iter_col_h]

    rows = []

    for key, g in e.groupby(group_cols_e, dropna=False):
        g = g.copy()

        if cand_obj_col is not None:
            g = g.sort_values(cand_obj_col, ascending=True, kind="mergesort")
        else:
            g = g.sort_values([cand_nodes_col, cand_days_col], kind="mergesort")

        candidate_records = []
        candidate_pairs = []

        for _, row in g.iterrows():
            kn = _json_safe_value(row[cand_nodes_col])
            kd = _json_safe_value(row[cand_days_col])

            rec = {
                "K_nodes": kn,
                "K_days": kd,
                "total_steps": None if kn is None or kd is None else int(kn * kd),
            }

            if cand_obj_col is not None:
                rec["objective"] = _json_safe_value(row[cand_obj_col])

            candidate_records.append(rec)
            candidate_pairs.append(f"{kn}x{kd}")

        best = candidate_records[0] if candidate_records else {}

        accepted_rank = None
        accepted_K_nodes = None
        accepted_K_days = None
        accepted_objective = None

        if accepted_col is not None:
            accepted_mask = g[accepted_col].astype(bool)
            if accepted_mask.any():
                accepted_row = g.loc[accepted_mask].iloc[0]
                accepted_K_nodes = _json_safe_value(accepted_row[cand_nodes_col])
                accepted_K_days = _json_safe_value(accepted_row[cand_days_col])
                if cand_obj_col is not None:
                    accepted_objective = _json_safe_value(accepted_row[cand_obj_col])

                for rank, rec in enumerate(candidate_records, start=1):
                    if (
                        rec["K_nodes"] == accepted_K_nodes
                        and rec["K_days"] == accepted_K_days
                    ):
                        accepted_rank = rank
                        break

        out = {
            "n_candidates": int(len(g)),
            "candidate_pairs": ";".join(candidate_pairs),
            "candidate_details_json": json.dumps(candidate_records),
            "best_candidate_K_nodes": best.get("K_nodes"),
            "best_candidate_K_days": best.get("K_days"),
            "best_candidate_total_steps": best.get("total_steps"),
            "best_candidate_objective": best.get("objective"),
            "accepted_candidate_K_nodes": accepted_K_nodes,
            "accepted_candidate_K_days": accepted_K_days,
            "accepted_candidate_objective": accepted_objective,
            "accepted_candidate_rank": accepted_rank,
        }

        if isinstance(key, tuple):
            for col, value in zip(group_cols_e, key):
                out[col] = value
        else:
            out[group_cols_e[0]] = key

        rows.append(out)

    alt = pd.DataFrame(rows)

    if alt.empty:
        return h

    # Align evaluation iteration column name to history iteration column name.
    if iter_col_e != iter_col_h and iter_col_e in alt.columns:
        alt = alt.rename(columns={iter_col_e: iter_col_h})

    merge_cols = group_cols_h

    h = h.merge(
        alt,
        on=merge_cols,
        how="left",
    )

    # If the accepted candidate was not explicitly available in evaluations,
    # infer its rank by matching the current history pair against candidate pairs.
    if (
        "accepted_candidate_rank" in h.columns
        and hist_nodes_col is not None
        and hist_days_col is not None
    ):
        for idx, row in h.iterrows():
            if pd.notna(row.get("accepted_candidate_rank")):
                continue

            details = row.get("candidate_details_json")
            if not isinstance(details, str) or not details:
                continue

            try:
                candidates = json.loads(details)
            except json.JSONDecodeError:
                continue

            current_kn = row.get(hist_nodes_col)
            current_kd = row.get(hist_days_col)

            for rank, rec in enumerate(candidates, start=1):
                if rec.get("K_nodes") == current_kn and rec.get("K_days") == current_kd:
                    h.at[idx, "accepted_candidate_rank"] = rank
                    h.at[idx, "accepted_candidate_K_nodes"] = current_kn
                    h.at[idx, "accepted_candidate_K_days"] = current_kd
                    h.at[idx, "accepted_candidate_objective"] = rec.get("objective")
                    break

    return h

def prepare_clustering_inputs(network_path: Path) -> dict:
    """
    Load the PyPSA network and build all inputs required by the reducer.
    """
    print(f">>> Loading network: {network_path}")
    n = pypsa.Network(network_path)

    base_buses = select_buses_from_loads(
        n,
        exclude_bus_substrings=EXCLUDE_BUS_SUBSTRINGS,
    )
    lat, lon = bus_coords_latlon(n, base_buses)

    print(f">>> Selected base buses: {len(base_buses)}")

    snaps = n.snapshots
    data_hourly: Dict[str, np.ndarray] = {}

    if INCLUDE_LOAD:
        load_bus = loads_by_bus_timeseries(n, base_buses).reindex(index=snaps)
        data_hourly["load"] = load_bus[base_buses].to_numpy(dtype=float).T

    pv_cf_bus = cf_by_bus_timeseries(
        n,
        base_buses,
        carrier=PV_CARRIER,
        weight_by=PV_WEIGHT_BY,
    ).reindex(index=snaps)
    data_hourly["pv_cf"] = pv_cf_bus[base_buses].to_numpy(dtype=float).T

    wind_cf_bus = cf_by_bus_timeseries(
        n,
        base_buses,
        carrier=WIND_CARRIER,
        weight_by=WIND_WEIGHT_BY,
    ).reindex(index=snaps)
    data_hourly["wind_cf"] = wind_cf_bus[base_buses].to_numpy(dtype=float).T

    n_snapshots = len(snaps)
    if n_snapshots % HOURS_PER_DAY != 0:
        raise ValueError(
            f"Snapshots length {n_snapshots} is not divisible by HOURS_PER_DAY={HOURS_PER_DAY}."
        )

    n_days = n_snapshots // HOURS_PER_DAY

    X, feature_names = build_tensor_X(
        data_hourly,
        hours_per_day=HOURS_PER_DAY,
        feature_mode=FEATURE_MODE,
        stats=STATS,
    )

    feature_weights = build_feature_weights(feature_names, FEATURE_WEIGHTS_CFG)

    node_weights = None
    if NODE_WEIGHTS_MODE is not None:
        mode = str(NODE_WEIGHTS_MODE).lower()

        if mode == "none":
            node_weights = None

        elif mode == "mean_load":
            if "load" not in data_hourly:
                raise ValueError("NODE_WEIGHTS_MODE='mean_load' requires load to be included.")
            node_weights = data_hourly["load"].mean(axis=1).astype(float)

        elif mode == "peak_load":
            if "load" not in data_hourly:
                raise ValueError("NODE_WEIGHTS_MODE='peak_load' requires load to be included.")
            node_weights = data_hourly["load"].max(axis=1).astype(float)

        else:
            raise ValueError(
                f"Unsupported NODE_WEIGHTS_MODE={NODE_WEIGHTS_MODE}. "
                "Supported: None, 'none', 'mean_load', 'peak_load'."
            )

    print(f">>> Built X with shape {X.shape} = (nodes, days, features)")
    print(f">>> Features: {feature_names}")

    return {
        "network_path": str(network_path),
        "base_buses": base_buses,
        "lat": lat,
        "lon": lon,
        "X": X,
        "feature_names": feature_names,
        "feature_weights": feature_weights,
        "node_weights": node_weights,
        "n_nodes": int(X.shape[0]),
        "n_days": int(X.shape[1]),
        "n_features": int(X.shape[2]),
    }


def run_one_reducer(
    *,
    X: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    base_buses: List[str],
    feature_weights: np.ndarray,
    node_weights: Optional[np.ndarray],
    run_id: str,
    init_mode: str,
    init_nodes: Optional[int],
    init_days: Optional[int],
    random_state: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Run one reducer instance and return summary, history, and evaluations.
    """
    reducer = AlternatingSpatioTemporalReducer(
        lambda_ts=float(REDUCER_BASE_CFG["lambda_ts"]),
        normalize=str(REDUCER_BASE_CFG["normalize"]),
        max_total_steps=int(REDUCER_BASE_CFG["max_total_steps"]),
        init_mode=str(init_mode),
        init_nodes=init_nodes,
        init_days=init_days,
        beta=float(REDUCER_BASE_CFG["beta"]),
        beta_growth=float(REDUCER_BASE_CFG["beta_growth"]),
        beta_max=float(REDUCER_BASE_CFG["beta_max"]),
        max_iter=int(REDUCER_BASE_CFG["max_iter"]),
        tol_no_change=int(REDUCER_BASE_CFG["tol_no_change"]),
        objective_tol_rel=float(REDUCER_BASE_CFG["objective_tol_rel"]),
        verbose=bool(REDUCER_BASE_CFG["verbose"]),
        norm_q=float(REDUCER_BASE_CFG["norm_q"]),
        use_pca_days=bool(REDUCER_BASE_CFG["use_pca_days"]),
        pca_days_n_components=REDUCER_BASE_CFG["pca_days_n_components"],
        pca_days_random_state=int(REDUCER_BASE_CFG["pca_days_random_state"]),
        standardize_day_matrix_cols=bool(REDUCER_BASE_CFG["standardize_day_matrix_cols"]),
        kmedoids_max_iter=int(REDUCER_BASE_CFG["kmedoids_max_iter"]),
        random_state=int(random_state),
        feature_weights=feature_weights,
    )

    t0 = time.perf_counter()

    result = reducer.fit(
        X,
        lat,
        lon,
        buses=base_buses,
        node_weights=node_weights,
    )

    elapsed = time.perf_counter() - t0

    final_k_nodes = int(len(np.unique(result.labels_nodes)))
    final_k_days = int(len(np.unique(result.labels_days)))
    final_steps = int(final_k_nodes * final_k_days)

    summary = {
        "run_id": run_id,
        "init_mode": init_mode,
        "init_nodes": init_nodes,
        "init_days": init_days,
        "init_steps": None if init_nodes is None or init_days is None else int(init_nodes * init_days),
        "random_state": int(random_state),
        "final_K_nodes": final_k_nodes,
        "final_K_days": final_k_days,
        "final_total_steps": final_steps,
        "objective": float(result.objective),
        "elapsed_seconds": float(elapsed),
        "n_history_rows": int(len(result.history)),
        "n_evaluation_rows": int(len(result.evaluations)),
        "lambda_ts": float(REDUCER_BASE_CFG["lambda_ts"]),
        "max_total_steps": int(REDUCER_BASE_CFG["max_total_steps"]),
        "beta": float(REDUCER_BASE_CFG["beta"]),
        "beta_growth": float(REDUCER_BASE_CFG["beta_growth"]),
        "beta_max": float(REDUCER_BASE_CFG["beta_max"]),
        "tol_no_change": int(REDUCER_BASE_CFG["tol_no_change"]),
        "objective_tol_rel": float(REDUCER_BASE_CFG["objective_tol_rel"]),
        "normalize": str(REDUCER_BASE_CFG["normalize"]),
        "norm_q": float(REDUCER_BASE_CFG["norm_q"]),
    }

    history = pd.DataFrame(result.history)
    if not history.empty:
        history.insert(0, "run_id", run_id)
        history.insert(1, "init_mode", init_mode)
        history.insert(2, "init_nodes", init_nodes)
        history.insert(3, "init_days", init_days)
        history.insert(4, "random_state", int(random_state))

    evaluations = pd.DataFrame(result.evaluations)
    if not evaluations.empty:
        evaluations.insert(0, "run_id", run_id)
        evaluations.insert(1, "init_mode", init_mode)
        evaluations.insert(2, "init_nodes", init_nodes)
        evaluations.insert(3, "init_days", init_days)
        evaluations.insert(4, "random_state", int(random_state))

    if not history.empty and not evaluations.empty:
        history = enrich_history_with_evaluation_alternatives(history, evaluations)

    return summary, history, evaluations


def main() -> None:
    """
    Main scan routine.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs = prepare_clustering_inputs(NETWORK_PATH)

    X = inputs["X"]
    lat = inputs["lat"]
    lon = inputs["lon"]
    base_buses = inputs["base_buses"]
    feature_weights = inputs["feature_weights"]
    node_weights = inputs["node_weights"]

    n_nodes = int(inputs["n_nodes"])
    n_days = int(inputs["n_days"])

    max_init_nodes = MAX_INIT_NODES if MAX_INIT_NODES is not None else n_nodes
    max_init_days = MAX_INIT_DAYS if MAX_INIT_DAYS is not None else n_days

    pairs = build_scan_pairs(
        n_nodes=n_nodes,
        n_days=n_days,
        target_budget=TARGET_BUDGET,
        min_steps=MIN_INITIAL_STEPS,
        max_steps=MAX_INITIAL_STEPS,
        pair_mode=PAIR_MODE,
        min_init_nodes=MIN_INIT_NODES,
        max_init_nodes=max_init_nodes,
        min_init_days=MIN_INIT_DAYS,
        max_init_days=max_init_days,
    )

    print(f">>> Pair mode: {PAIR_MODE}")
    print(f">>> Number of initial pairs: {len(pairs)}")
    print(f">>> First 20 pairs: {pairs[:20]}")

    metadata = {
        "network_path": str(NETWORK_PATH),
        "out_dir": str(OUT_DIR),
        "target_budget": TARGET_BUDGET,
        "min_initial_steps": MIN_INITIAL_STEPS,
        "max_initial_steps": MAX_INITIAL_STEPS,
        "pair_mode": PAIR_MODE,
        "run_full_baseline": RUN_FULL_BASELINE,
        "random_states": RANDOM_STATES,
        "n_nodes": n_nodes,
        "n_days": n_days,
        "n_features": int(inputs["n_features"]),
        "feature_names": inputs["feature_names"],
        "feature_weights": feature_weights.tolist(),
        "node_weights_mode": NODE_WEIGHTS_MODE,
        "reducer_base_cfg": REDUCER_BASE_CFG,
    }

    with open(OUT_DIR / "scan_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    summaries: List[dict] = []
    histories: List[pd.DataFrame] = []
    evaluations: List[pd.DataFrame] = []

    total_runs = len(pairs) * len(RANDOM_STATES)
    if RUN_FULL_BASELINE:
        total_runs += len(RANDOM_STATES)

    run_counter = 0

    # -------------------------------------------------------------------------
    # Full baseline
    # -------------------------------------------------------------------------
    if RUN_FULL_BASELINE:
        for seed in RANDOM_STATES:
            run_counter += 1
            run_id = f"full_seed{seed}"

            print(
                f">>> [{run_counter}/{total_runs}] Running {run_id}: "
                f"init_mode=full, seed={seed}"
            )

            summary, history, evals = run_one_reducer(
                X=X,
                lat=lat,
                lon=lon,
                base_buses=base_buses,
                feature_weights=feature_weights,
                node_weights=node_weights,
                run_id=run_id,
                init_mode="full",
                init_nodes=None,
                init_days=None,
                random_state=seed,
            )

            summaries.append(summary)
            if not history.empty:
                histories.append(history)
            if not evals.empty:
                evaluations.append(evals)

            pd.DataFrame(summaries).to_csv(OUT_DIR / "scan_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # Initial-pair scan
    # -------------------------------------------------------------------------
    for init_nodes, init_days in pairs:
        for seed in RANDOM_STATES:
            run_counter += 1
            init_steps = int(init_nodes * init_days)
            run_id = f"init_n{init_nodes}_d{init_days}_s{init_steps}_seed{seed}"

            print(
                f">>> [{run_counter}/{total_runs}] Running {run_id}: "
                f"init=({init_nodes}, {init_days}), steps={init_steps}, seed={seed}"
            )

            summary, history, evals = run_one_reducer(
                X=X,
                lat=lat,
                lon=lon,
                base_buses=base_buses,
                feature_weights=feature_weights,
                node_weights=node_weights,
                run_id=run_id,
                init_mode="balanced",
                init_nodes=int(init_nodes),
                init_days=int(init_days),
                random_state=seed,
            )

            summaries.append(summary)
            if not history.empty:
                histories.append(history)
            if not evals.empty:
                evaluations.append(evals)

            # Incremental output, useful if the scan is interrupted.
            pd.DataFrame(summaries).to_csv(OUT_DIR / "scan_summary.csv", index=False)

            if histories:
                pd.concat(histories, ignore_index=True).to_csv(
                    OUT_DIR / "scan_history.csv",
                    index=False,
                )

            if evaluations:
                pd.concat(evaluations, ignore_index=True).to_csv(
                    OUT_DIR / "scan_evaluations.csv",
                    index=False,
                )

    df_summary = pd.DataFrame(summaries).sort_values("objective")
    df_summary.to_csv(OUT_DIR / "scan_summary.csv", index=False)

    if histories:
        df_history = pd.concat(histories, ignore_index=True)
        df_history.to_csv(OUT_DIR / "scan_history.csv", index=False)

    if evaluations:
        df_evaluations = pd.concat(evaluations, ignore_index=True)
        df_evaluations.to_csv(OUT_DIR / "scan_evaluations.csv", index=False)

    best_runs = df_summary.head(30).copy()
    best_runs.to_csv(OUT_DIR / "best_runs.csv", index=False)

    final_shape_summary = (
        df_summary
        .groupby(["final_K_nodes", "final_K_days", "final_total_steps"], as_index=False)
        .agg(
            objective_best=("objective", "min"),
            objective_mean=("objective", "mean"),
            objective_std=("objective", "std"),
            n_runs=("objective", "size"),
            elapsed_mean_seconds=("elapsed_seconds", "mean"),
        )
        .sort_values("objective_best")
    )
    final_shape_summary.to_csv(OUT_DIR / "final_shape_summary.csv", index=False)

    print("\n>>> Best runs:")
    print(best_runs.head(15).to_string(index=False))

    print("\n>>> Best final shapes:")
    print(final_shape_summary.head(15).to_string(index=False))

    print(f"\n>>> Done. Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()