# -*- coding: utf-8 -*-
"""
Snakemake entrypoint: geo-temporal clustering + reconstruction for a PyPSA network.

- Activated by requesting the output file ..._gt.nc via opts containing "Gt"
  (the logic lives in the solve rule's input function).
- Produces:
  * clustered network (spatial + temporal)
  * mapping CSVs (nodes/days assignments, representative selections, busmap/linemap)

"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pypsa
import geopandas as gpd


# Ensure repo root on sys.path (PyPSA-Eur scripts style)
import sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import configure_logging
from scripts.geo_temporal_clustering.core import (
    AlternatingSpatioTemporalReducer,
    build_tensor_X,
    select_buses_from_loads,
    bus_coords_latlon,
    loads_by_bus_timeseries,
    electric_demand_weights_by_bus,
    cf_by_bus_timeseries,
    build_full_busmap,
    apply_temporal_reduction,
    representative_method_from_algorithm,
)

from scripts.geo_temporal_clustering.spatial_reconstruction import (
    reconstruct_spatially_clustered_network,
)

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Get key from dict-like or AttrDict-like config objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _smk_path(namedlist: Any, key: str, idx: int = 0) -> str:
    """Return a path from snakemake input/output regardless of named vs positional style."""
    try:
        return str(namedlist[key])
    except Exception:
        return str(namedlist[idx])





def _temporal_representation_to_objective_method(representation: str) -> str:
    """Map network temporal representation to tensor reconstruction method."""
    representation = str(representation).lower()
    if representation == "mean":
        return "mean"
    if representation in {"medoid", "medoid_scaled"}:
        return "medoid"
    raise ValueError("temporal representation must be one of: medoid, mean, medoid_scaled.")


def _resolve_objective_reconstruction_method(
    value: Any,
    *,
    clustering_algorithm: str,
    reconstruction_method: str,
    default: str,
    name: str,
) -> str:
    """
    Resolve objective reconstruction config.

    Accepted values:
    - "medoid" or "mean": explicit objective reconstruction method.
    - "clustering": derive from clustering algorithm
      (kmedoids -> medoid, kmeans -> mean).
    - "reconstruction": use the configured network reconstruction method.
    """
    if value is None:
        value = default

    value = str(value).lower()

    if value in {"medoid", "mean"}:
        return value

    if value in {"clustering", "algorithm"}:
        return representative_method_from_algorithm(clustering_algorithm)

    if value == "reconstruction":
        reconstruction_method = str(reconstruction_method).lower()
        if reconstruction_method not in {"medoid", "mean"}:
            raise ValueError(
                f"Cannot use reconstruction method {reconstruction_method!r} "
                f"for objective {name}; expected 'medoid' or 'mean'."
            )
        return reconstruction_method

    raise ValueError(
        f"Unsupported objective reconstruction setting for {name}: {value!r}. "
        "Use 'medoid', 'mean', 'clustering', or 'reconstruction'."
    )

def _build_feature_weights(feature_names, cfg_weights):
    """
    Build a feature-weight vector from a config dictionary.

    Supported keys:
    - exact feature name, e.g. "load_mean"
    - attribute prefix, e.g. "load", "pv_cf", "wind_cf"
      (applied to all derived features whose names start with "<key>_")
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


def main() -> None:
    configure_logging(snakemake)
    cfg = snakemake.params.get("geotemporal", {}) if hasattr(snakemake, "params") else {}

    # ---------------------------------------------------------------------
    # Config
    # ---------------------------------------------------------------------
    hours_per_day = int(_cfg_get(_cfg_get(cfg, "features", {}), "hours_per_day", 24))

    exclude_substrings = tuple(
        _cfg_get(_cfg_get(cfg, "buses", {}), "exclude_bus_substrings", [" H2", "battery"])
    )
    feature_mode = str(_cfg_get(_cfg_get(cfg, "features", {}), "feature_mode", "daily_stats"))
    stats = tuple(
        _cfg_get(
            _cfg_get(cfg, "features", {}),
            "stats",
            ["mean", "max", "min", "std", "ramp_max", "energy"],
        )
    )

    include = _cfg_get(_cfg_get(cfg, "features", {}), "include", {}) or {}
    include_load = bool(_cfg_get(include, "load", True))
    pv_cfg = _cfg_get(include, "pv_cf", {"carrier": "solar", "weight_by": "p_nom"}) or {}
    wind_cfg = _cfg_get(include, "wind_cf", {"carrier": "onwind", "weight_by": "p_nom"}) or {}

    objective_cfg = _cfg_get(cfg, "objective", {}) or {}
    feature_weights_cfg = _cfg_get(objective_cfg, "feature_weights", {}) or {}
    node_weights_cfg = _cfg_get(objective_cfg, "node_weights", None)

    # Reducer params
    r_cfg = _cfg_get(cfg, "reducer", {}) or {}
    algorithms_cfg = _cfg_get(r_cfg, "algorithms", {}) or {}
    spatial_algorithm = str(
        _cfg_get(
            algorithms_cfg,
            "spatial",
            _cfg_get(r_cfg, "spatial_algorithm", _cfg_get(r_cfg, "algorithm", "kmedoids")),
        )
    ).lower()
    temporal_algorithm = str(
        _cfg_get(
            algorithms_cfg,
            "temporal",
            _cfg_get(r_cfg, "temporal_algorithm", _cfg_get(r_cfg, "algorithm", "kmedoids")),
        )
    ).lower()

    enforce_spatial_adjacency = bool(_cfg_get(r_cfg, "enforce_spatial_adjacency", False))
    region_name_col = str(_cfg_get(r_cfg, "region_name_col", "name"))

    regions_gdf = None
    if enforce_spatial_adjacency:
        regions_path = _smk_path(snakemake.input, "regions")
        logger.info("Loading input regions for adjacency: %s", regions_path)
        regions_gdf = gpd.read_file(regions_path)

    # ---------------------------------------------------------------------
    # Load network
    # ---------------------------------------------------------------------
    in_network = _smk_path(snakemake.input, "network", 0)
    out_network = _smk_path(snakemake.output, "network", 0)

    logger.info("Loading network: %s", in_network)
    n = pypsa.Network(in_network)

    # ---------------------------------------------------------------------
    # Select base buses and build features
    # ---------------------------------------------------------------------
    base_buses = select_buses_from_loads(n, exclude_bus_substrings=exclude_substrings)
    lat, lon = bus_coords_latlon(n, base_buses)

    logger.info("Selected %d base buses for clustering.", len(base_buses))

    snaps = n.snapshots
    data_hourly: Dict[str, np.ndarray] = {}

    if include_load:
        load_bus = loads_by_bus_timeseries(n, base_buses).reindex(index=snaps)
        data_hourly["load"] = load_bus[base_buses].to_numpy(dtype=float).T

    pv_carrier = str(_cfg_get(pv_cfg, "carrier", "solar"))
    pv_weight_by = str(_cfg_get(pv_cfg, "weight_by", "p_nom"))
    pv_cf_bus = cf_by_bus_timeseries(
        n,
        base_buses,
        carrier=pv_carrier,
        weight_by=pv_weight_by,
    ).reindex(index=snaps)
    data_hourly["pv_cf"] = pv_cf_bus[base_buses].to_numpy(dtype=float).T

    wind_carrier = str(_cfg_get(wind_cfg, "carrier", "onwind"))
    wind_weight_by = str(_cfg_get(wind_cfg, "weight_by", "p_nom"))
    wind_cf_bus = cf_by_bus_timeseries(
        n,
        base_buses,
        carrier=wind_carrier,
        weight_by=wind_weight_by,
    ).reindex(index=snaps)
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

    feature_weights = _build_feature_weights(feat_names, feature_weights_cfg)

    # ---------------------------------------------------------------------
    # Optional node weights for spatial clustering and reconstruction loss
    # ---------------------------------------------------------------------
    node_weights = None

    if node_weights_cfg is not None:
        node_weights_mode = str(node_weights_cfg).lower()

        if node_weights_mode in {"electric_demand", "total_load", "demand"}:
            node_weights = electric_demand_weights_by_bus(
                n,
                base_buses,
                use_snapshot_weightings=True,
                normalization="mean",
            )

        elif node_weights_mode == "mean_load":
            if "load" not in data_hourly:
                raise ValueError("node_weights='mean_load' requires load to be included in features.")
            node_weights = data_hourly["load"].mean(axis=1).astype(float)
            node_weights = node_weights / (node_weights.mean() + 1e-12)

        elif node_weights_mode == "peak_load":
            if "load" not in data_hourly:
                raise ValueError("node_weights='peak_load' requires load to be included in features.")
            node_weights = data_hourly["load"].max(axis=1).astype(float)
            node_weights = node_weights / (node_weights.mean() + 1e-12)

        elif node_weights_mode == "none":
            node_weights = None

        else:
            raise ValueError(
                f"Unsupported node_weights setting: {node_weights_cfg}. "
                "Supported: None, 'none', 'electric_demand', 'total_load', "
                "'demand', 'mean_load', 'peak_load'."
            )
        
    temporal_cfg = _cfg_get(cfg, "temporal", {}) or {}
    temporal_representation = str(_cfg_get(temporal_cfg, "representation", "medoid")).lower()

    reconstruction_cfg = _cfg_get(cfg, "reconstruction", {}) or {}
    spatial_reconstruction_method = str(
        _cfg_get(reconstruction_cfg, "spatial", "medoid")
    ).lower()
    temporal_reconstruction_method = _temporal_representation_to_objective_method(
        temporal_representation
    )

    objective_reconstruction_cfg = _cfg_get(objective_cfg, "reconstruction", {}) or {}
    objective_spatial_reconstruction = _resolve_objective_reconstruction_method(
        _cfg_get(objective_reconstruction_cfg, "spatial", None),
        clustering_algorithm=spatial_algorithm,
        reconstruction_method=spatial_reconstruction_method,
        default="medoid",
        name="spatial",
    )
    objective_temporal_reconstruction = _resolve_objective_reconstruction_method(
        _cfg_get(objective_reconstruction_cfg, "temporal", None),
        clustering_algorithm=temporal_algorithm,
        reconstruction_method=temporal_reconstruction_method,
        default="mean",
        name="temporal",
    )

    logger.info(
        "GT algorithms: spatial=%s, temporal=%s; objective reconstruction: spatial=%s, temporal=%s; network temporal representation=%s.",
        spatial_algorithm,
        temporal_algorithm,
        objective_spatial_reconstruction,
        objective_temporal_reconstruction,
        temporal_representation,
    )

    reducer = AlternatingSpatioTemporalReducer(
        reduction_mode=str(_cfg_get(r_cfg, "reduction_mode", "budget")),
        fixed_nodes=float(_cfg_get(r_cfg, "fixed_nodes", 9)),
        fixed_days=float(_cfg_get(r_cfg, "fixed_days", 30)),
        loss_norm=str(_cfg_get(r_cfg, "loss_norm", "l2_squared")),
        lambda_ts=float(_cfg_get(r_cfg, "lambda_ts", 0.5)),
        normalize=str(_cfg_get(r_cfg, "normalize", "zscore")),
        max_total_steps=int(_cfg_get(r_cfg, "max_total_steps", 144)),
        init_mode=str(_cfg_get(r_cfg, "init_mode", "balanced")),
        init_nodes=_cfg_get(r_cfg, "init_nodes", None),
        init_days=_cfg_get(r_cfg, "init_days", None),
        beta=float(_cfg_get(r_cfg, "beta", 0.15)),
        beta_growth=float(_cfg_get(r_cfg, "beta_growth", 2.0)),
        beta_max=float(_cfg_get(r_cfg, "beta_max", 0.5)),
        max_iter=int(_cfg_get(r_cfg, "max_iter", 20)),
        tol_no_change=int(_cfg_get(r_cfg, "tol_no_change", 2)),
        objective_tol_rel=float(_cfg_get(r_cfg, "objective_tol_rel", 1e-5)),
        verbose=bool(_cfg_get(r_cfg, "verbose", True)),
        norm_q=float(_cfg_get(r_cfg, "norm_q", 0.95)),
        use_pca_days=bool(_cfg_get(_cfg_get(r_cfg, "pca_days", {}), "enable", False)),
        pca_days_n_components=_cfg_get(_cfg_get(r_cfg, "pca_days", {}), "n_components", 0.95),
        pca_days_random_state=int(_cfg_get(_cfg_get(r_cfg, "pca_days", {}), "random_state", 0)),
        standardize_day_matrix_cols=bool(
            _cfg_get(_cfg_get(r_cfg, "pca_days", {}), "standardize_day_matrix_cols", False)
        ),
        kmedoids_max_iter=int(_cfg_get(r_cfg, "kmedoids_max_iter", 100)),
        spatial_clustering_algorithm=spatial_algorithm,
        temporal_clustering_algorithm=temporal_algorithm,
        objective_spatial_reconstruction=objective_spatial_reconstruction,
        objective_temporal_reconstruction=objective_temporal_reconstruction,
        random_state=int(_cfg_get(r_cfg, "random_state", 0)),
        enforce_spatial_adjacency=enforce_spatial_adjacency,
        regions_gdf=regions_gdf,
        region_name_col=region_name_col,
        feature_weights=feature_weights,
    )

    # ---------------------------------------------------------------------
    # Run reducer
    # ---------------------------------------------------------------------
    result = reducer.fit(
        X,
        lat,
        lon,
        buses=base_buses,
        node_weights=node_weights,
    )

    logger.info(
        "Reducer completed: %d node clusters, %d day clusters, objective=%.6e.",
        len(np.unique(result.labels_nodes)),
        len(np.unique(result.labels_days)),
        result.objective,
    )

    # ---------------------------------------------------------------------
    # Build busmap for ALL buses (incl. auxiliary suffix buses)
    # ---------------------------------------------------------------------
    busmap = build_full_busmap(
        n,
        base_buses,
        result.labels_nodes.astype(int),
        result.rep_nodes.astype(int),
    )

    # ---------------------------------------------------------------------
    # Spatial clustering / reconstruction via PyPSA
    # ---------------------------------------------------------------------
    logger.info("Reconstructing clustered network using GT spatial reconstruction.")

    nc, linemap = reconstruct_spatially_clustered_network(
        n=n,
        busmap=busmap,
        aggregate_one_ports=True,
        aggregate_links=True,
        custom_line_groupers=[],
    )

    # ---------------------------------------------------------------------
    # Temporal reduction
    # ---------------------------------------------------------------------
    logger.info("Applying temporal reduction: %d representative days.", len(result.rep_days))
    apply_temporal_reduction(
        nc,
        rep_days=result.rep_days,
        rep_weights=result.rep_weights,
        labels_days=result.labels_days,
        hours_per_day=hours_per_day,
        representation=temporal_representation,
    )

    # ---------------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------------
    out_nodes_assignment = _smk_path(snakemake.output, "nodes_assignment", 1)
    out_days_assignment = _smk_path(snakemake.output, "days_assignment", 2)
    out_rep_days = _smk_path(snakemake.output, "representative_days", 3)
    out_rep_nodes = _smk_path(snakemake.output, "representative_nodes", 4)
    out_busmap = _smk_path(snakemake.output, "busmap", 5)
    out_linemap = _smk_path(snakemake.output, "linemap", 6)
    out_summary = _smk_path(snakemake.output, "summary", 7)

    outdir = Path(out_busmap).parent
    outdir.mkdir(parents=True, exist_ok=True)

    # Base nodes assignment
    df_nodes = pd.DataFrame(
        {
            "bus": base_buses,
            "lat": lat,
            "lon": lon,
            "node_cluster": result.labels_nodes.astype(int),
        }
    )

    rep_bus_by_cluster = {
        int(c): base_buses[i]
        for c, i in enumerate(result.rep_nodes.astype(int))
    }
    df_nodes["rep_bus"] = df_nodes["node_cluster"].map(rep_bus_by_cluster)

    if result.region_membership is not None:
        df_nodes["input_region"] = (
            result.region_membership.reindex(df_nodes["bus"]).astype(str).values
        )

    df_nodes.to_csv(out_nodes_assignment, index=False)

    # Representative nodes
    df_rep_nodes = pd.DataFrame(
        {
            "rep_bus": [base_buses[i] for i in result.rep_nodes.astype(int)],
            "rep_lat": lat[result.rep_nodes.astype(int)],
            "rep_lon": lon[result.rep_nodes.astype(int)],
            "rep_node_cluster": np.arange(len(result.rep_nodes), dtype=int),
            "cluster_size": result.rep_node_weights.astype(float),
        }
    ).sort_values("rep_node_cluster")
    df_rep_nodes.to_csv(out_rep_nodes, index=False)

    # Day assignment
    df_days = pd.DataFrame(
        {
            "day_index": np.arange(X.shape[1], dtype=int),
            "day_cluster": result.labels_days.astype(int),
        }
    )

    rep_day_map = {
        int(cluster_label): int(rep_day)
        for cluster_label, rep_day in enumerate(result.rep_days.astype(int))
    }
    df_days["rep_day_index"] = df_days["day_cluster"].map(rep_day_map)

    rep_weight_map = {
        int(d): int(w) for d, w in zip(result.rep_days, result.rep_weights)
    }
    df_days["rep_weight"] = df_days["rep_day_index"].map(rep_weight_map).fillna(0).astype(int)
    df_days.to_csv(out_days_assignment, index=False)

    # Representative days
    df_rep_days = pd.DataFrame(
        {
            "rep_day_index": result.rep_days.astype(int),
            "rep_weight": result.rep_weights.astype(int),
            "rep_day_cluster": np.arange(len(result.rep_days), dtype=int),
        }
    ).sort_values("rep_day_cluster")
    df_rep_days.to_csv(out_rep_days, index=False)

    # Busmap / Linemap
    busmap.to_csv(out_busmap)
    linemap.to_csv(out_linemap)

    # Extra history/evaluations CSVs
    if len(result.history) > 0:
        pd.DataFrame(result.history).to_csv(outdir / "history.csv", index=False)
    if len(result.evaluations) > 0:
        pd.DataFrame(result.evaluations).to_csv(outdir / "evaluations.csv", index=False)

    # Summary
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
        "n_rep_days": int(len(result.rep_days)),
        "n_node_clusters": int(len(np.unique(result.labels_nodes))),
        "n_day_clusters": int(len(np.unique(result.labels_days))),
        "max_total_steps": int(_cfg_get(r_cfg, "max_total_steps", 144)),
        "actual_total_steps": int(len(np.unique(result.labels_nodes)) * len(np.unique(result.labels_days))),
        "objective": float(result.objective),
        "spatial_clustering_algorithm": spatial_algorithm,
        "temporal_clustering_algorithm": temporal_algorithm,
        "objective_spatial_reconstruction": objective_spatial_reconstruction,
        "objective_temporal_reconstruction": objective_temporal_reconstruction,
        "temporal_representation": temporal_representation,
        "feature_names": feat_names,
        "feature_weights": feature_weights.tolist(),
        "spatial_adjacency_enabled": bool(enforce_spatial_adjacency),
        "region_name_col": region_name_col if enforce_spatial_adjacency else None,
        "history": result.history,
        "evaluations": result.evaluations,
        "day_pca_info": result.day_pca_info,
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\\n")

    logger.info("Exporting clustered network: %s", out_network)
    nc.export_to_netcdf(out_network)


if __name__ == "__main__":

    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "geo_temporal_cluster_network",
            clusters="adm",
            opts="Gt",
            configfiles=["config/clustering_validation/config_validation_weights.yaml"],
            run="gtf-n13-d66"
        )

    main()