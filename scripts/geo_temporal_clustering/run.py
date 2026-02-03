# -*- coding: utf-8 -*-
"""
Snakemake entrypoint: geo-temporal clustering + reconstruction for a PyPSA network.

- Activated by requesting the output file ..._gt.nc via opts containing "Gt"
  (the logic lives in the solve rule's input function).
- Produces:
  * clustered network (spatial + temporal)
  * mapping CSVs (nodes/days assignments, representative selections, busmap/linemap)

All comments are in English by request.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pypsa

from pypsa.clustering.spatial import get_clustering_from_busmap

# Ensure repo root on sys.path (PyPSA-Eur scripts style)
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]  # repo root (scripts/geo_temporal_clustering/run.py -> parents[2])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import configure_logging

from scripts.geo_temporal_clustering.core import (
    AlternatingSpatioTemporalReducer,
    representative_medoids,
    build_node_ts_distance,
    haversine_pairwise_km,
    normalize_distance_matrix,
    zscore_global,
    minmax_global,
    build_tensor_X,
    select_buses_from_loads,
    bus_coords_latlon,
    loads_by_bus_timeseries,
    cf_by_bus_timeseries,
    build_full_busmap,
    apply_temporal_reduction,
)

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Get key from dict-like or AttrDict-like config objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def main() -> None:
    if "snakemake" not in globals():
        raise RuntimeError("This script is intended to be run by Snakemake.")

    configure_logging(snakemake)
    cfg = snakemake.params.get("geotemporal", {}) if hasattr(snakemake, "params") else {}

    hours_per_day = int(_cfg_get(_cfg_get(cfg, "features", {}), "hours_per_day", 24))

    exclude_substrings = tuple(_cfg_get(_cfg_get(cfg, "buses", {}), "exclude_bus_substrings", [" H2", "battery"]))
    feature_mode = str(_cfg_get(_cfg_get(cfg, "features", {}), "feature_mode", "daily_stats"))
    stats = tuple(_cfg_get(_cfg_get(cfg, "features", {}), "stats", ["mean", "max", "min", "std", "ramp_max", "energy"]))

    include = _cfg_get(_cfg_get(cfg, "features", {}), "include", {}) or {}
    include_load = bool(_cfg_get(include, "load", True))
    pv_cfg = _cfg_get(include, "pv_cf", {"carrier": "solar", "weight_by": "p_nom"}) or {}
    wind_cfg = _cfg_get(include, "wind_cf", {"carrier": "onwind", "weight_by": "p_nom"}) or {}

    # Reducer params
    r_cfg = _cfg_get(cfg, "reducer", {}) or {}
    reducer = AlternatingSpatioTemporalReducer(
        lambda_ts=float(_cfg_get(r_cfg, "lambda_ts", 0.5)),
        normalize=str(_cfg_get(r_cfg, "normalize", "zscore")),
        node_threshold=float(_cfg_get(r_cfg, "node_threshold", 0.30)),
        day_threshold=float(_cfg_get(r_cfg, "day_threshold", 0.30)),
        linkage=str(_cfg_get(r_cfg, "linkage", "average")),
        max_iter=int(_cfg_get(r_cfg, "max_iter", 10)),
        tol_no_change=int(_cfg_get(r_cfg, "tol_no_change", 2)),
        verbose=True,
        norm_q=float(_cfg_get(r_cfg, "norm_q", 0.95)),
        use_pca_days=bool(_cfg_get(_cfg_get(r_cfg, "pca_days", {}), "enable", False)),
        pca_days_n_components=_cfg_get(_cfg_get(r_cfg, "pca_days", {}), "n_components", 0.95),
        standardize_day_matrix_cols=bool(_cfg_get(_cfg_get(r_cfg, "pca_days", {}), "standardize_day_matrix_cols", False)),
    )

    # ---------------------------------------------------------------------
    # Load network
    # ---------------------------------------------------------------------
    in_network = snakemake.input["network"]
    out_network = snakemake.output["network"]

    logger.info("Loading network: %s", in_network)
    n = pypsa.Network(in_network)

    # ---------------------------------------------------------------------
    # Select base buses and build features
    # ---------------------------------------------------------------------
    base_buses = select_buses_from_loads(n, exclude_bus_substrings=exclude_substrings)
    lat, lon = bus_coords_latlon(n, base_buses)

    logger.info("Selected %d base buses for clustering.", len(base_buses))

    # Build hourly features (DataFrames with snapshots index, base_buses columns)
    snaps = n.snapshots
    data_hourly: Dict[str, np.ndarray] = {}

    if include_load:
        load_bus = loads_by_bus_timeseries(n, base_buses).reindex(index=snaps)
        data_hourly["load"] = load_bus[base_buses].to_numpy(dtype=float).T

    pv_carrier = str(_cfg_get(pv_cfg, "carrier", "solar"))
    pv_weight_by = str(_cfg_get(pv_cfg, "weight_by", "p_nom"))
    pv_cf_bus = cf_by_bus_timeseries(n, base_buses, carrier=pv_carrier, weight_by=pv_weight_by).reindex(index=snaps)
    data_hourly["pv_cf"] = pv_cf_bus[base_buses].to_numpy(dtype=float).T

    wind_carrier = str(_cfg_get(wind_cfg, "carrier", "onwind"))
    wind_weight_by = str(_cfg_get(wind_cfg, "weight_by", "p_nom"))
    wind_cf_bus = cf_by_bus_timeseries(n, base_buses, carrier=wind_carrier, weight_by=wind_weight_by).reindex(index=snaps)
    data_hourly["wind_cf"] = wind_cf_bus[base_buses].to_numpy(dtype=float).T

    T = len(snaps)
    if T % hours_per_day != 0:
        raise ValueError(f"Snapshots length {T} not divisible by hours_per_day={hours_per_day}.")
    D = T // hours_per_day
    logger.info("Snapshots=%d -> Days=%d (hours_per_day=%d).", T, D, hours_per_day)

    X, feat_names = build_tensor_X(
        data_hourly,
        hours_per_day=hours_per_day,
        feature_mode=feature_mode,
        stats=stats,
    )
    logger.info("Built tensor X with shape %s (N, D, F), F=%d.", X.shape, len(feat_names))

    # ---------------------------------------------------------------------
    # Run reducer
    # ---------------------------------------------------------------------
    res = reducer.fit(X, lat, lon, node_weights=None)

    # ---------------------------------------------------------------------
    # Representative nodes (medoids of final spatial clusters)
    # ---------------------------------------------------------------------
    if reducer.normalize == "zscore":
        Xn = zscore_global(X)
    else:
        Xn = minmax_global(X)

    D_geo = haversine_pairwise_km(lat, lon)
    D_geo_n = normalize_distance_matrix(D_geo, q=float(_cfg_get(r_cfg, "norm_q", 0.95)))

    X_for_nodes = Xn[:, res.rep_days, :]
    w_for_nodes = res.rep_weights.astype(float)

    D_ts_raw = build_node_ts_distance(X_for_nodes, w_for_nodes)
    D_ts_n = normalize_distance_matrix(D_ts_raw, q=float(_cfg_get(r_cfg, "norm_q", 0.95)))

    D_node = float(_cfg_get(r_cfg, "lambda_ts", 0.5)) * D_ts_n + (1.0 - float(_cfg_get(r_cfg, "lambda_ts", 0.5))) * D_geo_n

    rep_nodes_idx, rep_nodes_w = representative_medoids(D_node, res.labels_nodes.astype(int))
    rep_labels = res.labels_nodes[rep_nodes_idx].astype(int)

    # ---------------------------------------------------------------------
    # Build busmap for ALL buses (incl. auxiliary suffix buses)
    # ---------------------------------------------------------------------
    busmap = build_full_busmap(n, base_buses, res.labels_nodes.astype(int), rep_nodes_idx)

    # ---------------------------------------------------------------------
    # Spatial clustering / reconstruction via PyPSA
    # ---------------------------------------------------------------------
    logger.info("Reconstructing clustered network using PyPSA get_clustering_from_busmap.")
    clustering = get_clustering_from_busmap(
        n,
        busmap,
        bus_strategies={},     # keep defaults; we set coordinates afterwards
        line_strategies={},    # keep defaults (PyPSA merges corridors and drops internal edges)
        custom_line_groupers=["build_year"] if "build_year" in n.lines.columns else [],
    )
    nc = clustering.n

    # Set bus coordinates to representative bus coordinates when possible
    if "x" in n.buses.columns and "y" in n.buses.columns:
        for b_new in nc.buses.index.astype(str):
            base, suffix = (b_new[:-3], " H2") if b_new.endswith(" H2") else (b_new[:-8], " battery") if b_new.endswith(" battery") else (b_new, "")
            # base should exist in original buses if it is a representative
            if base in n.buses.index:
                nc.buses.at[b_new, "x"] = float(n.buses.at[base, "x"])
                nc.buses.at[b_new, "y"] = float(n.buses.at[base, "y"])

    # ---------------------------------------------------------------------
    # Temporal reduction
    # ---------------------------------------------------------------------
    logger.info("Applying temporal reduction: %d representative days.", len(res.rep_days))
    apply_temporal_reduction(
        nc,
        rep_days=res.rep_days,
        rep_weights=res.rep_weights,
        hours_per_day=hours_per_day,
    )

    # ---------------------------------------------------------------------
    # Write outputs (network + mapping CSVs)
    # ---------------------------------------------------------------------
    outdir = Path(str(snakemake.output.get("busmap", out_network))).parent
    outdir.mkdir(parents=True, exist_ok=True)

    # Base nodes assignment (only base_buses)
    df_nodes = pd.DataFrame(
        {
            "bus": base_buses,
            "lat": lat,
            "lon": lon,
            "node_cluster": res.labels_nodes.astype(int),
        }
    )
    # Add representative bus per cluster
    rep_bus_by_cluster = {int(c): base_buses[i] for i, c in zip(rep_nodes_idx, rep_labels)}
    df_nodes["rep_bus"] = df_nodes["node_cluster"].map(rep_bus_by_cluster)
    df_nodes.to_csv(snakemake.output["nodes_assignment"], index=False)

    # Representative nodes table
    df_rep_nodes = pd.DataFrame(
        {
            "rep_bus": [base_buses[i] for i in rep_nodes_idx],
            "rep_lat": lat[rep_nodes_idx],
            "rep_lon": lon[rep_nodes_idx],
            "rep_node_cluster": rep_labels,
            "cluster_size": rep_nodes_w.astype(int),
        }
    ).sort_values("rep_node_cluster")
    df_rep_nodes.to_csv(snakemake.output["representative_nodes"], index=False)

    # Day assignment
    df_days = pd.DataFrame(
        {"day_index": np.arange(X.shape[1], dtype=int), "day_cluster": res.labels_days.astype(int)}
    )
    rep_day_by_cluster = {int(k): int(d) for d, k in zip(res.rep_days, df_days.set_index("day_index").loc[res.rep_days, "day_cluster"].values)}
    # Map each day to its medoid via cluster label
    df_days["rep_day_index"] = df_days["day_cluster"].map({int(lbl): int(day) for lbl, day in zip(df_days.set_index("day_index").loc[res.rep_days, "day_cluster"].values, res.rep_days)})
    # Weights per rep day
    rep_weight_map = {int(d): int(w) for d, w in zip(res.rep_days, res.rep_weights)}
    df_days["rep_weight"] = df_days["rep_day_index"].map(rep_weight_map).fillna(0).astype(int)
    df_days.to_csv(snakemake.output["days_assignment"], index=False)

    # Representative days table
    df_rep_days = pd.DataFrame({"rep_day_index": res.rep_days.astype(int), "rep_weight": res.rep_weights.astype(int)})
    df_rep_days.sort_values("rep_day_index").to_csv(snakemake.output["representative_days"], index=False)

    # Busmap / Linemap from PyPSA clustering
    busmap.to_csv(snakemake.output["busmap"])
    clustering.linemap.to_csv(snakemake.output["linemap"])

    # Summary YAML-ish (simple, no dependency)
    summary = {
        "input_network": str(in_network),
        "output_network": str(out_network),
        "n_buses_in": int(len(n.buses)),
        "n_buses_base": int(len(base_buses)),
        "n_buses_out": int(len(nc.buses)),
        "n_lines_in": int(len(n.lines)),
        "n_lines_out": int(len(nc.lines)),
        "n_links_in": int(len(n.links)),
        "n_links_out": int(len(nc.links)),
        "n_days_in": int(D),
        "n_rep_days": int(len(res.rep_days)),
        "n_node_clusters": int(len(np.unique(res.labels_nodes))),
        "n_day_clusters": int(len(np.unique(res.labels_days))),
        "history": res.history,
        "day_pca_info": res.day_pca_info,
    }
    # Write as YAML-ish for readability (no pyyaml required)
    with open(snakemake.output["summary"], "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    logger.info("Exporting clustered network: %s", out_network)
    nc.export_to_netcdf(out_network)


if __name__ == "__main__":
    main()
