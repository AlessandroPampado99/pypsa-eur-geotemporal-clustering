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

import argparse
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
    electric_demand_weights_by_bus,
    cf_by_bus_timeseries,
    reconstruct_tensor,
    weighted_reconstruction_loss,
    zscore_global,
    minmax_global,
)
from scripts.geo_temporal_clustering.plot_scan_summary import generate_scan_summary_plots


# =============================================================================
# USER SETTINGS
# =============================================================================

NETWORK_PATH = Path("/home/pampado/clustering/pypsa-eur/resources/reference_nuts3_ES_kmeans_400/complete/networks/base_s_adm_elec_.nc")

OUT_DIR = Path("resources/geotemporal_clustering_scan/400_mean_ES")

# Standalone reducer mode:
# - "budget": reproduce the budget/local-search behaviour
# - "fixed_pair": evaluate exactly each (K_nodes, K_days) pair
RUN_MODE = "budget"

# Main scan parameters
TARGET_BUDGET = 380
MIN_INITIAL_STEPS = 400
MAX_INITIAL_STEPS = 420

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
RUN_FULL_BASELINE = False

# Seeds to test for each initial pair
RANDOM_STATES = [0]

# Feature extraction
HOURS_PER_DAY = 24
EXCLUDE_BUS_SUBSTRINGS = (" H2", "battery")

FEATURE_MODE = "daily_stats"
STATS = ("mean", "max", "std", "ramp_max")

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

# Supported: None, "none", "mean_load", "peak_load", "electric_demand"
# "electric_demand" uses the total electric demand per bus, including snapshot weightings,
# and normalizes weights to mean 1.
NODE_WEIGHTS_MODE = "electric_demand"

# Algorithm choices to compare. The scan runs the full cross-product of these
# lists. Keep each list length 1 to run a single configuration.
SPATIAL_CLUSTERING_ALGORITHMS = ["kmeans"]  # "kmedoids" or "kmeans"
TEMPORAL_CLUSTERING_ALGORITHMS = ["kmeans"]  # "kmedoids" or "kmeans"

# Objective tensor reconstruction used by diagnostics and by the reducer loss.
# Supported: "medoid", "mean", "clustering".
# "clustering" maps kmedoids -> medoid and kmeans -> mean per axis.
OBJECTIVE_SPATIAL_RECONSTRUCTION = "clustering"
OBJECTIVE_TEMPORAL_RECONSTRUCTION = "clustering"

# Run plot_scan_summary automatically at the end of a successful scan.
PLOT_AFTER_SCAN = True
SCAN_PLOTS_DIRNAME = "plots_summary"

# Reducer parameters
REDUCER_BASE_CFG = {
    "lambda_ts": 0.10,
    "normalize": "zscore",
    "max_total_steps": TARGET_BUDGET,
    "loss_norm": "l2_squared",
    "beta": 0.05,
    "beta_growth": 1.3,
    "beta_max": 1,
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
    "candidate_seed_mode": "full",
}

# Temporal representation strategy used when the clustered network is actually
# built. In this standalone scan REPRESENTATION is stored for traceability; the
# reducer objective is controlled separately by OBJECTIVE_*_RECONSTRUCTION above.
# Supported by core.py: "medoid", "mean", "medoid_scaled".
REPRESENTATION = "mean"


# =============================================================================
# Helpers
# =============================================================================



def resolve_objective_reconstruction_method(value: str, algorithm: str, *, name: str) -> str:
    """Resolve medoid/mean/clustering objective reconstruction settings."""
    value = str(value).lower()
    algorithm = str(algorithm).lower()

    if value in {"medoid", "mean"}:
        return value

    if value in {"clustering", "algorithm"}:
        if algorithm == "kmedoids":
            return "medoid"
        if algorithm == "kmeans":
            return "mean"
        raise ValueError(f"Unsupported {name} clustering algorithm: {algorithm!r}.")

    raise ValueError(
        f"OBJECTIVE_{name.upper()}_RECONSTRUCTION must be one of "
        "'medoid', 'mean', or 'clustering'. "
        f"Got {value!r}."
    )


def build_algorithm_scenarios() -> List[dict]:
    """Build the algorithm scenarios scanned for every initial pair."""
    scenarios = []

    for spatial_algorithm in SPATIAL_CLUSTERING_ALGORITHMS:
        for temporal_algorithm in TEMPORAL_CLUSTERING_ALGORITHMS:
            spatial_algorithm = str(spatial_algorithm).lower()
            temporal_algorithm = str(temporal_algorithm).lower()

            objective_spatial = resolve_objective_reconstruction_method(
                OBJECTIVE_SPATIAL_RECONSTRUCTION,
                spatial_algorithm,
                name="spatial",
            )
            objective_temporal = resolve_objective_reconstruction_method(
                OBJECTIVE_TEMPORAL_RECONSTRUCTION,
                temporal_algorithm,
                name="temporal",
            )

            scenarios.append(
                {
                    "name": f"sp{spatial_algorithm}_tm{temporal_algorithm}_obj{objective_spatial}_{objective_temporal}",
                    "spatial_algorithm": spatial_algorithm,
                    "temporal_algorithm": temporal_algorithm,
                    "objective_spatial_reconstruction": objective_spatial,
                    "objective_temporal_reconstruction": objective_temporal,
                }
            )

    return scenarios

def validate_settings() -> None:
    """Validate user-facing settings before starting the scan."""
    valid_representations = {"medoid", "mean", "medoid_scaled"}
    if REPRESENTATION not in valid_representations:
        raise ValueError(
            f"REPRESENTATION must be one of {sorted(valid_representations)}, "
            f"got {REPRESENTATION!r}."
        )

    valid_run_modes = {"budget", "fixed_pair"}
    if RUN_MODE not in valid_run_modes:
        raise ValueError(
            f"RUN_MODE must be one of {sorted(valid_run_modes)}, "
            f"got {RUN_MODE!r}."
        )

    if RUN_MODE == "fixed_pair" and RUN_FULL_BASELINE:
        raise ValueError(
            "RUN_FULL_BASELINE must be False when RUN_MODE='fixed_pair', "
            "because the full baseline has no init_nodes/init_days pair."
        )

    valid_node_weight_modes = {None, "none", "mean_load", "peak_load", "electric_demand"}
    if NODE_WEIGHTS_MODE not in valid_node_weight_modes:
        raise ValueError(
            f"NODE_WEIGHTS_MODE must be one of {sorted(map(str, valid_node_weight_modes))}, "
            f"got {NODE_WEIGHTS_MODE!r}."
        )

    valid_algorithms = {"kmedoids", "kmeans"}
    for algorithm in SPATIAL_CLUSTERING_ALGORITHMS:
        if str(algorithm).lower() not in valid_algorithms:
            raise ValueError(f"Unsupported spatial algorithm: {algorithm!r}.")
    for algorithm in TEMPORAL_CLUSTERING_ALGORITHMS:
        if str(algorithm).lower() not in valid_algorithms:
            raise ValueError(f"Unsupported temporal algorithm: {algorithm!r}.")

    valid_objective_reconstruction = {"medoid", "mean", "clustering", "algorithm"}
    if str(OBJECTIVE_SPATIAL_RECONSTRUCTION).lower() not in valid_objective_reconstruction:
        raise ValueError(
            "OBJECTIVE_SPATIAL_RECONSTRUCTION must be one of "
            f"{sorted(valid_objective_reconstruction)}, got "
            f"{OBJECTIVE_SPATIAL_RECONSTRUCTION!r}."
        )
    if str(OBJECTIVE_TEMPORAL_RECONSTRUCTION).lower() not in valid_objective_reconstruction:
        raise ValueError(
            "OBJECTIVE_TEMPORAL_RECONSTRUCTION must be one of "
            f"{sorted(valid_objective_reconstruction)}, got "
            f"{OBJECTIVE_TEMPORAL_RECONSTRUCTION!r}."
        )

    valid_loss_norms = {"l1", "l2_squared"}
    if str(REDUCER_BASE_CFG.get("loss_norm")) not in valid_loss_norms:
        raise ValueError(
            f"REDUCER_BASE_CFG['loss_norm'] must be one of {sorted(valid_loss_norms)}, "
            f"got {REDUCER_BASE_CFG.get('loss_norm')!r}."
        )


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

    return float(np.sum(wn * err_by_node))

def compute_space_time_loss_decomposition(
    *,
    X: np.ndarray,
    feature_weights: np.ndarray,
    node_weights: Optional[np.ndarray],
    labels_nodes: np.ndarray,
    labels_days: np.ndarray,
    rep_nodes: np.ndarray,
    rep_days: np.ndarray,
    spatial_method: str,
    temporal_method: str,
    normalize: str,
    loss_norm: str,
) -> dict:
    """
    Decompose reconstruction loss consistently with the configured reducer objective.
    """
    X = np.asarray(X, dtype=float)
    N, D, F = X.shape

    if normalize == "zscore":
        Xn = zscore_global(X)
    elif normalize == "minmax":
        Xn = minmax_global(X)
    else:
        raise ValueError("normalize must be either 'zscore' or 'minmax'.")

    labels_nodes = np.asarray(labels_nodes, dtype=int)
    labels_days = np.asarray(labels_days, dtype=int)
    rep_nodes = np.asarray(rep_nodes, dtype=int)
    rep_days = np.asarray(rep_days, dtype=int)

    # Full reconstruction: same logic as the reducer objective.
    X_rec_full = reconstruct_tensor(
        Xn,
        rep_nodes=rep_nodes,
        labels_nodes=labels_nodes,
        rep_days=rep_days,
        labels_days=labels_days,
        spatial_method=spatial_method,
        temporal_method=temporal_method,
    )

    # Spatial-only reconstruction: apply configured spatial reconstruction and
    # keep all original days unchanged.
    X_rec_space = reconstruct_tensor(
        Xn,
        rep_nodes=rep_nodes,
        labels_nodes=labels_nodes,
        rep_days=np.arange(D, dtype=int),
        labels_days=np.arange(D, dtype=int),
        spatial_method=spatial_method,
        temporal_method="medoid",
    )

    # Temporal-only reconstruction: keep original nodes and apply configured
    # temporal reconstruction.
    X_rec_time = reconstruct_tensor(
        Xn,
        rep_nodes=np.arange(N, dtype=int),
        labels_nodes=np.arange(N, dtype=int),
        rep_days=rep_days,
        labels_days=labels_days,
        spatial_method="medoid",
        temporal_method=temporal_method,
    )

    loss_full = weighted_reconstruction_loss(
        Xn,
        X_rec_full,
        feature_weights=feature_weights,
        node_loss_weights=node_weights,
        loss_norm=loss_norm,
    )

    loss_space = weighted_reconstruction_loss(
        Xn,
        X_rec_space,
        feature_weights=feature_weights,
        node_loss_weights=node_weights,
        loss_norm=loss_norm,
    )

    loss_time = weighted_reconstruction_loss(
        Xn,
        X_rec_time,
        feature_weights=feature_weights,
        node_loss_weights=node_weights,
        loss_norm=loss_norm,
    )

    return {
        "loss_full": float(loss_full),
        "loss_space_only": float(loss_space),
        "loss_time_only": float(loss_time),
        "loss_interaction": float(loss_full - loss_space - loss_time),
        "space_share_vs_full": float(loss_space / (loss_full + 1e-12)),
        "time_share_vs_full": float(loss_time / (loss_full + 1e-12)),
        "space_to_time_ratio": float(loss_space / (loss_time + 1e-12)),
    }

def compute_axis_loss_breakdown(
    *,
    X: np.ndarray,
    feature_names: List[str],
    feature_weights: np.ndarray,
    node_weights: Optional[np.ndarray],
    labels_nodes: np.ndarray,
    labels_days: np.ndarray,
    rep_nodes: np.ndarray,
    rep_days: np.ndarray,
    spatial_method: str,
    temporal_method: str,
    normalize: str,
    loss_norm: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute reconstruction loss by original node and by original day.

    This is consistent with spatial medoids + temporal cluster means.
    rep_days is kept only for traceability in the output.
    """
    X = np.asarray(X, dtype=float)
    N, D, F = X.shape

    if normalize == "zscore":
        Xn = zscore_global(X)
    elif normalize == "minmax":
        Xn = minmax_global(X)
    else:
        raise ValueError("normalize must be either 'zscore' or 'minmax'.")

    X_rec = reconstruct_tensor(
        Xn,
        rep_nodes=np.asarray(rep_nodes, dtype=int),
        labels_nodes=np.asarray(labels_nodes, dtype=int),
        rep_days=np.asarray(rep_days, dtype=int),
        labels_days=np.asarray(labels_days, dtype=int),
        spatial_method=spatial_method,
        temporal_method=temporal_method,
    )

    if loss_norm == "l2_squared":
        err = (Xn - X_rec) ** 2
    elif loss_norm == "l1":
        err = np.abs(Xn - X_rec)
    else:
        raise ValueError("loss_norm must be either 'l1' or 'l2_squared'.")

    wf = np.asarray(feature_weights, dtype=float)
    if wf.shape != (F,):
        raise ValueError(f"feature_weights must have shape ({F},), got {wf.shape}.")
    wf = wf / (wf.mean() + 1e-12)

    if node_weights is None:
        wn = np.ones(N, dtype=float)
    else:
        wn = np.asarray(node_weights, dtype=float)
        if wn.shape != (N,):
            raise ValueError(f"node_weights must have shape ({N},), got {wn.shape}.")
        wn = wn / (wn.mean() + 1e-12)

    err_weighted = err * wf[None, None, :]

    node_loss = err_weighted.sum(axis=(1, 2)) * wn
    day_loss = (err_weighted * wn[:, None, None]).sum(axis=(0, 2))

    total = float(node_loss.sum())

    labels_nodes = np.asarray(labels_nodes, dtype=int)
    labels_days = np.asarray(labels_days, dtype=int)
    rep_nodes = np.asarray(rep_nodes, dtype=int)
    rep_days = np.asarray(rep_days, dtype=int)

    df_node = pd.DataFrame(
        {
            "node_index": np.arange(N, dtype=int),
            "node_cluster": labels_nodes,
            "rep_node_index": rep_nodes[labels_nodes],
            "node_weight_normalized": wn,
            "loss": node_loss,
            "loss_share": node_loss / (total + 1e-12),
        }
    ).sort_values("loss", ascending=False)

    df_day = pd.DataFrame(
        {
            "day_index": np.arange(D, dtype=int),
            "day_cluster": labels_days,
            "rep_day_index": rep_days[labels_days],
            "loss": day_loss,
            "loss_share": day_loss / (total + 1e-12),
        }
    ).sort_values("loss", ascending=False)

    return df_node.reset_index(drop=True), df_day.reset_index(drop=True)

def compute_feature_loss_breakdown(
    *,
    X: np.ndarray,
    feature_names: List[str],
    feature_weights: np.ndarray,
    node_weights: Optional[np.ndarray],
    labels_nodes: np.ndarray,
    labels_days: np.ndarray,
    rep_nodes: np.ndarray,
    rep_days: np.ndarray,
    spatial_method: str,
    temporal_method: str,
    normalize: str,
    loss_norm: str,
) -> pd.DataFrame:
    """
    Compute the reducer reconstruction loss contribution feature by feature.

    This mirrors the reducer objective:
    - X is normalized first using the same normalization mode;
    - reconstruction is medoid-based;
    - feature weights are normalized to mean 1;
    - node weights are normalized to mean 1;
    - loss can be l2_squared or l1.
    """
    X = np.asarray(X, dtype=float)
    N, D, F = X.shape

    if normalize == "zscore":
        Xn = zscore_global(X)
    elif normalize == "minmax":
        Xn = minmax_global(X)
    else:
        raise ValueError("normalize must be either 'zscore' or 'minmax'.")

    X_rec = reconstruct_tensor(
        Xn,
        rep_nodes=np.asarray(rep_nodes, dtype=int),
        labels_nodes=np.asarray(labels_nodes, dtype=int),
        rep_days=np.asarray(rep_days, dtype=int),
        labels_days=np.asarray(labels_days, dtype=int),
        spatial_method=spatial_method,
        temporal_method=temporal_method,
    )

    if loss_norm == "l2_squared":
        err = (Xn - X_rec) ** 2
    elif loss_norm == "l1":
        err = np.abs(Xn - X_rec)
    else:
        raise ValueError("loss_norm must be either 'l1' or 'l2_squared'.")

    wf = np.asarray(feature_weights, dtype=float)
    if wf.shape != (F,):
        raise ValueError(f"feature_weights must have shape ({F},), got {wf.shape}.")
    if np.any(wf < 0):
        raise ValueError("feature_weights must be non-negative.")
    wf_norm = wf / (wf.mean() + 1e-12)

    if node_weights is None:
        wn_norm = np.ones(N, dtype=float)
    else:
        wn = np.asarray(node_weights, dtype=float)
        if wn.shape != (N,):
            raise ValueError(f"node_weights must have shape ({N},), got {wn.shape}.")
        if np.any(wn < 0):
            raise ValueError("node_weights must be non-negative.")
        wn_norm = wn / (wn.mean() + 1e-12)

    rows = []
    for f, name in enumerate(feature_names):
        # Loss contribution of feature f:
        # sum_n w_n * sum_d err[n,d,f] * normalized_feature_weight[f]
        loss_unweighted = float(err[:, :, f].sum())
        loss_node_weighted = float((err[:, :, f].sum(axis=1) * wn_norm).sum())
        loss_weighted = float(loss_node_weighted * wf_norm[f])

        rows.append(
            {
                "feature": str(name),
                "feature_weight_raw": float(wf[f]),
                "feature_weight_normalized": float(wf_norm[f]),
                "loss_unweighted": loss_unweighted,
                "loss_node_weighted": loss_node_weighted,
                "loss_weighted": loss_weighted,
            }
        )

    out = pd.DataFrame(rows)
    total = float(out["loss_weighted"].sum())
    out["loss_share"] = out["loss_weighted"] / (total + 1e-12)

    # Useful grouped fields, e.g. load_mean -> variable=load, stat=mean.
    # Feature names without an underscore are accepted for synthetic tests or
    # custom feature tensors.
    parts = out["feature"].str.rsplit("_", n=1, expand=True)
    out["feature_family"] = parts[0]
    out["stat"] = parts[1] if parts.shape[1] > 1 else "value"

    return out.sort_values("loss_weighted", ascending=False).reset_index(drop=True)

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

    print(f">>> Exported pre-clustering tensor to: {OUT_DIR / 'pre_clustering_tensor.npz'}")

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

        elif mode == "electric_demand":
            node_weights = electric_demand_weights_by_bus(
                n,
                base_buses,
                use_snapshot_weightings=True,
                normalization="mean",
                fallback_to_uniform=True,
            ).astype(float)

        else:
            raise ValueError(
                f"Unsupported NODE_WEIGHTS_MODE={NODE_WEIGHTS_MODE}. "
                "Supported: None, 'none', 'mean_load', 'peak_load', 'electric_demand'."
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
    feature_names: List[str],
    feature_weights: np.ndarray,
    node_weights: Optional[np.ndarray],
    run_id: str,
    init_mode: str,
    init_nodes: Optional[int],
    init_days: Optional[int],
    random_state: int,
    algorithm_scenario: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run one reducer instance and return summary, history, and evaluations.
    """
    spatial_algorithm = str(algorithm_scenario["spatial_algorithm"])
    temporal_algorithm = str(algorithm_scenario["temporal_algorithm"])
    objective_spatial_reconstruction = str(algorithm_scenario["objective_spatial_reconstruction"])
    objective_temporal_reconstruction = str(algorithm_scenario["objective_temporal_reconstruction"])
    algorithm_scenario_name = str(algorithm_scenario["name"])

    if RUN_MODE == "fixed_pair":
        if init_nodes is None or init_days is None:
            raise ValueError(
                "RUN_MODE='fixed_pair' requires init_nodes and init_days. "
                "Set RUN_FULL_BASELINE=False."
            )

        reducer = AlternatingSpatioTemporalReducer(
            lambda_ts=float(REDUCER_BASE_CFG["lambda_ts"]),
            normalize=str(REDUCER_BASE_CFG["normalize"]),

            # In fixed-pair mode this is not used to search, but keep it coherent.
            max_total_steps=int(init_nodes * init_days),

            reduction_mode="fixed_pair",
            fixed_nodes=int(init_nodes),
            fixed_days=int(init_days),
            candidate_seed_mode=str(
                REDUCER_BASE_CFG.get("candidate_seed_mode", "current")
            ),
            loss_norm=str(REDUCER_BASE_CFG["loss_norm"]),
            verbose=bool(REDUCER_BASE_CFG["verbose"]),
            norm_q=float(REDUCER_BASE_CFG["norm_q"]),
            use_pca_days=bool(REDUCER_BASE_CFG["use_pca_days"]),
            pca_days_n_components=REDUCER_BASE_CFG["pca_days_n_components"],
            pca_days_random_state=int(REDUCER_BASE_CFG["pca_days_random_state"]),
            standardize_day_matrix_cols=bool(REDUCER_BASE_CFG["standardize_day_matrix_cols"]),
            kmedoids_max_iter=int(REDUCER_BASE_CFG["kmedoids_max_iter"]),
            spatial_clustering_algorithm=spatial_algorithm,
            temporal_clustering_algorithm=temporal_algorithm,
            objective_spatial_reconstruction=objective_spatial_reconstruction,
            objective_temporal_reconstruction=objective_temporal_reconstruction,
            random_state=int(random_state),
            feature_weights=feature_weights,
        )

    elif RUN_MODE == "budget":
        reducer = AlternatingSpatioTemporalReducer(
            lambda_ts=float(REDUCER_BASE_CFG["lambda_ts"]),
            normalize=str(REDUCER_BASE_CFG["normalize"]),
            max_total_steps=int(REDUCER_BASE_CFG["max_total_steps"]),
            loss_norm=str(REDUCER_BASE_CFG["loss_norm"]),
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
            spatial_clustering_algorithm=spatial_algorithm,
            temporal_clustering_algorithm=temporal_algorithm,
            objective_spatial_reconstruction=objective_spatial_reconstruction,
            objective_temporal_reconstruction=objective_temporal_reconstruction,
            random_state=int(random_state),
            feature_weights=feature_weights,
            candidate_seed_mode=str(
                REDUCER_BASE_CFG.get("candidate_seed_mode", "current")
            ),
        )

    else:
        raise ValueError(f"Unsupported RUN_MODE={RUN_MODE!r}.")

    t0 = time.perf_counter()

    result = reducer.fit(
        X,
        lat,
        lon,
        buses=base_buses,
        node_weights=node_weights,
    )

    axis_decomp = compute_space_time_loss_decomposition(
        X=X,
        feature_weights=feature_weights,
        node_weights=node_weights,
        labels_nodes=result.labels_nodes,
        labels_days=result.labels_days,
        rep_nodes=result.rep_nodes,
        rep_days=result.rep_days,
        spatial_method=objective_spatial_reconstruction,
        temporal_method=objective_temporal_reconstruction,
        normalize=str(REDUCER_BASE_CFG["normalize"]),
        loss_norm=str(REDUCER_BASE_CFG["loss_norm"]),
    )

    axis_decomp["loss_full_objective_abs_diff"] = abs(
        float(axis_decomp["loss_full"]) - float(result.objective)
    )

    axis_decomp["loss_full_objective_rel_diff"] = (
        axis_decomp["loss_full_objective_abs_diff"]
        / max(abs(float(result.objective)), 1e-12)
    )

    feature_losses = compute_feature_loss_breakdown(
        X=X,
        feature_names=feature_names,
        feature_weights=feature_weights,
        node_weights=node_weights,
        labels_nodes=result.labels_nodes,
        labels_days=result.labels_days,
        rep_nodes=result.rep_nodes,
        rep_days=result.rep_days,
        spatial_method=objective_spatial_reconstruction,
        temporal_method=objective_temporal_reconstruction,
        normalize=str(REDUCER_BASE_CFG["normalize"]),
        loss_norm=str(REDUCER_BASE_CFG["loss_norm"]),
    )

    node_losses, day_losses = compute_axis_loss_breakdown(
        X=X,
        feature_names=feature_names,
        feature_weights=feature_weights,
        node_weights=node_weights,
        labels_nodes=result.labels_nodes,
        labels_days=result.labels_days,
        rep_nodes=result.rep_nodes,
        rep_days=result.rep_days,
        spatial_method=objective_spatial_reconstruction,
        temporal_method=objective_temporal_reconstruction,
        normalize=str(REDUCER_BASE_CFG["normalize"]),
        loss_norm=str(REDUCER_BASE_CFG["loss_norm"]),
    )

    node_losses.insert(0, "run_id", run_id)
    node_losses.insert(1, "algorithm_scenario", algorithm_scenario_name)
    day_losses.insert(0, "run_id", run_id)
    day_losses.insert(1, "algorithm_scenario", algorithm_scenario_name)

    feature_losses.insert(0, "run_id", run_id)
    feature_losses.insert(1, "init_mode", init_mode)
    feature_losses.insert(2, "init_nodes", init_nodes)
    feature_losses.insert(3, "init_days", init_days)
    feature_losses.insert(4, "random_state", int(random_state))
    feature_losses.insert(5, "representation", REPRESENTATION)
    feature_losses.insert(6, "algorithm_scenario", algorithm_scenario_name)
    feature_losses.insert(7, "spatial_algorithm", spatial_algorithm)
    feature_losses.insert(8, "temporal_algorithm", temporal_algorithm)
    feature_losses.insert(9, "objective_spatial_reconstruction", objective_spatial_reconstruction)
    feature_losses.insert(10, "objective_temporal_reconstruction", objective_temporal_reconstruction)

    elapsed = time.perf_counter() - t0

    final_k_nodes = int(len(np.unique(result.labels_nodes)))
    final_k_days = int(len(np.unique(result.labels_days)))
    final_steps = int(final_k_nodes * final_k_days)

    summary = {
        "run_id": run_id,
        "run_mode": RUN_MODE,
        "init_mode": init_mode,
        "init_nodes": init_nodes,
        "init_days": init_days,
        "init_steps": None if init_nodes is None or init_days is None else int(init_nodes * init_days),
        "random_state": int(random_state),
        "representation": REPRESENTATION,
        "algorithm_scenario": algorithm_scenario_name,
        "spatial_algorithm": spatial_algorithm,
        "temporal_algorithm": temporal_algorithm,
        "objective_spatial_reconstruction": objective_spatial_reconstruction,
        "objective_temporal_reconstruction": objective_temporal_reconstruction,
        "node_weights_mode": NODE_WEIGHTS_MODE,
        "final_K_nodes": final_k_nodes,
        "final_K_days": final_k_days,
        "final_total_steps": final_steps,
        "objective": float(result.objective),
        "elapsed_seconds": float(elapsed),
        "n_history_rows": int(len(result.history)),
        "n_evaluation_rows": int(len(result.evaluations)),
        "lambda_ts": float(REDUCER_BASE_CFG["lambda_ts"]),
        "max_total_steps": int(REDUCER_BASE_CFG["max_total_steps"]),
        "loss_norm": str(REDUCER_BASE_CFG["loss_norm"]),
        "beta": float(REDUCER_BASE_CFG["beta"]),
        "beta_growth": float(REDUCER_BASE_CFG["beta_growth"]),
        "beta_max": float(REDUCER_BASE_CFG["beta_max"]),
        "tol_no_change": int(REDUCER_BASE_CFG["tol_no_change"]),
        "objective_tol_rel": float(REDUCER_BASE_CFG["objective_tol_rel"]),
        "normalize": str(REDUCER_BASE_CFG["normalize"]),
        "norm_q": float(REDUCER_BASE_CFG["norm_q"]),
    }

    summary.update(axis_decomp)

    history = pd.DataFrame(result.history)
    if not history.empty:
        history.insert(0, "run_id", run_id)
        history.insert(1, "init_mode", init_mode)
        history.insert(2, "init_nodes", init_nodes)
        history.insert(3, "init_days", init_days)
        history.insert(4, "random_state", int(random_state))
        history.insert(5, "representation", REPRESENTATION)
        history.insert(6, "algorithm_scenario", algorithm_scenario_name)
        history.insert(7, "spatial_algorithm", spatial_algorithm)
        history.insert(8, "temporal_algorithm", temporal_algorithm)
        history.insert(9, "objective_spatial_reconstruction", objective_spatial_reconstruction)
        history.insert(10, "objective_temporal_reconstruction", objective_temporal_reconstruction)

    evaluations = pd.DataFrame(result.evaluations)
    if not evaluations.empty:
        evaluations.insert(0, "run_id", run_id)
        evaluations.insert(1, "init_mode", init_mode)
        evaluations.insert(2, "init_nodes", init_nodes)
        evaluations.insert(3, "init_days", init_days)
        evaluations.insert(4, "random_state", int(random_state))
        evaluations.insert(5, "representation", REPRESENTATION)
        evaluations.insert(6, "algorithm_scenario", algorithm_scenario_name)
        evaluations.insert(7, "spatial_algorithm", spatial_algorithm)
        evaluations.insert(8, "temporal_algorithm", temporal_algorithm)
        evaluations.insert(9, "objective_spatial_reconstruction", objective_spatial_reconstruction)
        evaluations.insert(10, "objective_temporal_reconstruction", objective_temporal_reconstruction)

    if not history.empty and not evaluations.empty:
        history = enrich_history_with_evaluation_alternatives(history, evaluations)

    return summary, history, evaluations, feature_losses, node_losses, day_losses




def record_scan_outputs(
    *,
    out_dir: Path,
    summaries: List[dict],
    histories: List[pd.DataFrame],
    evaluations: List[pd.DataFrame],
    feature_losses_all: List[pd.DataFrame],
    node_losses_all: List[pd.DataFrame],
    day_losses_all: List[pd.DataFrame],
    summary: dict,
    history: pd.DataFrame,
    evals: pd.DataFrame,
    feature_losses: pd.DataFrame,
    node_losses: pd.DataFrame,
    day_losses: pd.DataFrame,
) -> None:
    """Append one run's outputs and refresh interrupt-safe CSVs."""
    summaries.append(summary)

    if not history.empty:
        histories.append(history)
    if not evals.empty:
        evaluations.append(evals)
    if not feature_losses.empty:
        feature_losses_all.append(feature_losses)
    if not node_losses.empty:
        node_losses_all.append(node_losses)
    if not day_losses.empty:
        day_losses_all.append(day_losses)

    pd.DataFrame(summaries).to_csv(out_dir / "scan_summary.csv", index=False)

    if histories:
        pd.concat(histories, ignore_index=True).to_csv(
            out_dir / "scan_history.csv",
            index=False,
        )

    if evaluations:
        pd.concat(evaluations, ignore_index=True).to_csv(
            out_dir / "scan_evaluations.csv",
            index=False,
        )

    if feature_losses_all:
        pd.concat(feature_losses_all, ignore_index=True).to_csv(
            out_dir / "scan_feature_losses.csv",
            index=False,
        )

    if node_losses_all:
        pd.concat(node_losses_all, ignore_index=True).to_csv(
            out_dir / "scan_node_losses.csv",
            index=False,
        )

    if day_losses_all:
        pd.concat(day_losses_all, ignore_index=True).to_csv(
            out_dir / "scan_day_losses.csv",
            index=False,
        )

def run_scan() -> None:
    """
    Main scan routine.
    """
    validate_settings()
    algorithm_scenarios = build_algorithm_scenarios()

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
        "run_mode": RUN_MODE,
        "random_states": RANDOM_STATES,
        "algorithm_scenarios": algorithm_scenarios,
        "spatial_clustering_algorithms": SPATIAL_CLUSTERING_ALGORITHMS,
        "temporal_clustering_algorithms": TEMPORAL_CLUSTERING_ALGORITHMS,
        "objective_spatial_reconstruction_setting": OBJECTIVE_SPATIAL_RECONSTRUCTION,
        "objective_temporal_reconstruction_setting": OBJECTIVE_TEMPORAL_RECONSTRUCTION,
        "n_nodes": n_nodes,
        "n_days": n_days,
        "n_features": int(inputs["n_features"]),
        "feature_names": inputs["feature_names"],
        "feature_weights": feature_weights.tolist(),
        "node_weights_mode": NODE_WEIGHTS_MODE,
        "representation": REPRESENTATION,
        "reducer_base_cfg": REDUCER_BASE_CFG,
    }

    with open(OUT_DIR / "scan_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    summaries: List[dict] = []
    histories: List[pd.DataFrame] = []
    evaluations: List[pd.DataFrame] = []
    feature_losses_all: List[pd.DataFrame] = []
    node_losses_all: List[pd.DataFrame] = []
    day_losses_all: List[pd.DataFrame] = []

    total_runs = len(pairs) * len(RANDOM_STATES) * len(algorithm_scenarios)
    if RUN_FULL_BASELINE:
        total_runs += len(RANDOM_STATES) * len(algorithm_scenarios)

    run_counter = 0

    # -------------------------------------------------------------------------
    # Full baseline
    # -------------------------------------------------------------------------
    if RUN_FULL_BASELINE:
        for scenario in algorithm_scenarios:
            for seed in RANDOM_STATES:
                run_counter += 1
                run_id = f"{scenario['name']}_full_seed{seed}"

                print(
                    f">>> [{run_counter}/{total_runs}] Running {run_id}: "
                    f"init_mode=full, seed={seed}"
                )

                summary, history, evals, feature_losses, node_losses, day_losses = run_one_reducer(
                    X=X,
                    lat=lat,
                    lon=lon,
                    base_buses=base_buses,
                    feature_names=inputs["feature_names"],
                    feature_weights=feature_weights,
                    node_weights=node_weights,
                    run_id=run_id,
                    init_mode="full",
                    init_nodes=None,
                    init_days=None,
                    random_state=seed,
                    algorithm_scenario=scenario,
                )

                record_scan_outputs(
                    out_dir=OUT_DIR,
                    summaries=summaries,
                    histories=histories,
                    evaluations=evaluations,
                    feature_losses_all=feature_losses_all,
                    node_losses_all=node_losses_all,
                    day_losses_all=day_losses_all,
                    summary=summary,
                    history=history,
                    evals=evals,
                    feature_losses=feature_losses,
                    node_losses=node_losses,
                    day_losses=day_losses,
                )

    # -------------------------------------------------------------------------
    # Initial-pair scan
    # -------------------------------------------------------------------------
    for scenario in algorithm_scenarios:
        for init_nodes, init_days in pairs:
            for seed in RANDOM_STATES:
                run_counter += 1
                init_steps = int(init_nodes * init_days)
                run_id = f"{scenario['name']}_{RUN_MODE}_n{init_nodes}_d{init_days}_s{init_steps}_seed{seed}"

                print(
                    f">>> [{run_counter}/{total_runs}] Running {run_id}: "
                    f"scenario={scenario['name']}, init=({init_nodes}, {init_days}), "
                    f"steps={init_steps}, seed={seed}"
                )

                summary, history, evals, feature_losses, node_losses, day_losses = run_one_reducer(
                    X=X,
                    lat=lat,
                    lon=lon,
                    base_buses=base_buses,
                    feature_names=inputs["feature_names"],
                    feature_weights=feature_weights,
                    node_weights=node_weights,
                    run_id=run_id,
                    init_mode="balanced",
                    init_nodes=int(init_nodes),
                    init_days=int(init_days),
                    random_state=seed,
                    algorithm_scenario=scenario,
                )

                record_scan_outputs(
                    out_dir=OUT_DIR,
                    summaries=summaries,
                    histories=histories,
                    evaluations=evaluations,
                    feature_losses_all=feature_losses_all,
                    node_losses_all=node_losses_all,
                    day_losses_all=day_losses_all,
                    summary=summary,
                    history=history,
                    evals=evals,
                    feature_losses=feature_losses,
                    node_losses=node_losses,
                    day_losses=day_losses,
                )

    df_summary = pd.DataFrame(summaries).sort_values("objective")
    df_summary.to_csv(OUT_DIR / "scan_summary.csv", index=False)

    if histories:
        df_history = pd.concat(histories, ignore_index=True)
        df_history.to_csv(OUT_DIR / "scan_history.csv", index=False)

    if evaluations:
        df_evaluations = pd.concat(evaluations, ignore_index=True)
        df_evaluations.to_csv(OUT_DIR / "scan_evaluations.csv", index=False)

    if feature_losses_all:
        df_feature_losses = pd.concat(feature_losses_all, ignore_index=True)
        df_feature_losses.to_csv(OUT_DIR / "scan_feature_losses.csv", index=False)

        feature_loss_summary = (
            df_feature_losses
            .groupby(["feature"], as_index=False)
            .agg(
                loss_weighted_mean=("loss_weighted", "mean"),
                loss_share_mean=("loss_share", "mean"),
                loss_share_max=("loss_share", "max"),
                n_runs=("loss_share", "size"),
            )
            .sort_values("loss_share_mean", ascending=False)
        )
        feature_loss_summary.to_csv(
            OUT_DIR / "feature_loss_summary.csv",
            index=False,
        )

        stat_loss_summary = (
            df_feature_losses
            .groupby(["stat"], as_index=False)
            .agg(
                loss_weighted_mean=("loss_weighted", "mean"),
                loss_share_mean=("loss_share", "mean"),
                loss_share_max=("loss_share", "max"),
                n_runs=("loss_share", "size"),
            )
            .sort_values("loss_share_mean", ascending=False)
        )
        stat_loss_summary.to_csv(
            OUT_DIR / "stat_loss_summary.csv",
            index=False,
        )

        family_loss_summary = (
            df_feature_losses
            .groupby(["feature_family"], as_index=False)
            .agg(
                loss_weighted_mean=("loss_weighted", "mean"),
                loss_share_mean=("loss_share", "mean"),
                loss_share_max=("loss_share", "max"),
                n_runs=("loss_share", "size"),
            )
            .sort_values("loss_share_mean", ascending=False)
        )
        family_loss_summary.to_csv(
            OUT_DIR / "feature_family_loss_summary.csv",
            index=False,
        )

    best_runs = df_summary.head(30).copy()
    best_runs.to_csv(OUT_DIR / "best_runs.csv", index=False)

    final_shape_summary = (
        df_summary
        .groupby(["algorithm_scenario", "final_K_nodes", "final_K_days", "final_total_steps"], as_index=False)
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

    if PLOT_AFTER_SCAN:
        plots_dir = OUT_DIR / SCAN_PLOTS_DIRNAME
        print(f"\n>>> Generating scan summary plots in: {plots_dir}")
        generate_scan_summary_plots(OUT_DIR, out_dir=plots_dir)

    print(f"\n>>> Done. Outputs written to: {OUT_DIR}")


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standalone geo-temporal clustering budget scan and, by "
            "default, generate summary plots from the same output directory."
        )
    )
    parser.add_argument(
        "--network-path",
        type=Path,
        default=NETWORK_PATH,
        help="Input pre-clustering PyPSA network.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for scan CSVs and plots.",
    )
    parser.add_argument(
        "--spatial-algorithms",
        default=",".join(SPATIAL_CLUSTERING_ALGORITHMS),
        help="Comma-separated spatial algorithms: kmedoids,kmeans.",
    )
    parser.add_argument(
        "--temporal-algorithms",
        default=",".join(TEMPORAL_CLUSTERING_ALGORITHMS),
        help="Comma-separated temporal algorithms: kmedoids,kmeans.",
    )
    parser.add_argument(
        "--objective-spatial-reconstruction",
        default=OBJECTIVE_SPATIAL_RECONSTRUCTION,
        choices=["medoid", "mean", "clustering", "algorithm"],
        help="Objective spatial reconstruction method.",
    )
    parser.add_argument(
        "--objective-temporal-reconstruction",
        default=OBJECTIVE_TEMPORAL_RECONSTRUCTION,
        choices=["medoid", "mean", "clustering", "algorithm"],
        help="Objective temporal reconstruction method.",
    )
    parser.add_argument(
        "--plot",
        dest="plot_after_scan",
        action="store_true",
        default=PLOT_AFTER_SCAN,
        help="Generate summary plots after the scan.",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot_after_scan",
        action="store_false",
        help="Do not generate summary plots after the scan.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip scanning and regenerate plots from --out-dir/scan_summary.csv.",
    )
    return parser.parse_args()


def apply_cli_args(args: argparse.Namespace) -> None:
    global NETWORK_PATH, OUT_DIR
    global SPATIAL_CLUSTERING_ALGORITHMS, TEMPORAL_CLUSTERING_ALGORITHMS
    global OBJECTIVE_SPATIAL_RECONSTRUCTION, OBJECTIVE_TEMPORAL_RECONSTRUCTION
    global PLOT_AFTER_SCAN

    NETWORK_PATH = Path(args.network_path)
    OUT_DIR = Path(args.out_dir)
    SPATIAL_CLUSTERING_ALGORITHMS = _split_csv(args.spatial_algorithms)
    TEMPORAL_CLUSTERING_ALGORITHMS = _split_csv(args.temporal_algorithms)
    OBJECTIVE_SPATIAL_RECONSTRUCTION = str(args.objective_spatial_reconstruction)
    OBJECTIVE_TEMPORAL_RECONSTRUCTION = str(args.objective_temporal_reconstruction)
    PLOT_AFTER_SCAN = bool(args.plot_after_scan)


def main() -> None:
    args = parse_args()
    apply_cli_args(args)

    if args.plot_only:
        generate_scan_summary_plots(OUT_DIR, out_dir=OUT_DIR / SCAN_PLOTS_DIRNAME)
        return

    run_scan()


if __name__ == "__main__":
    main()
