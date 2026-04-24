# -*- coding: utf-8 -*-
"""
Geo-temporal clustering core utilities for PyPSA-Eur integration.

This module is intentionally self-contained and uses only public PyPSA APIs.

"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import geopandas as gpd

from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA


# =============================================================================
# Spatial adjacency utilities
# =============================================================================

def _prepare_regions_geodataframe(
    regions_gdf: gpd.GeoDataFrame,
    *,
    region_name_col: str = "name",
) -> gpd.GeoDataFrame:
    """
    Validate and standardize the input regions GeoDataFrame.

    Notes
    -----
    - The region identifier column is cast to string.
    - Duplicate region names are dissolved to ensure uniqueness.
    - The geometry is converted to EPSG:4326 for spatial joins.
    """
    if region_name_col not in regions_gdf.columns:
        raise ValueError(f"Column '{region_name_col}' not found in regions_gdf.")

    regions = regions_gdf[[region_name_col, "geometry"]].copy()

    if regions.empty:
        raise ValueError("regions_gdf is empty.")

    if regions.crs is None:
        regions = regions.set_crs("EPSG:4326")
    else:
        regions = regions.to_crs("EPSG:4326")

    regions[region_name_col] = regions[region_name_col].astype(str)

    # Dissolve duplicate region identifiers if needed
    if regions[region_name_col].duplicated().any():
        regions = regions.dissolve(by=region_name_col, as_index=False)

    # Drop empty geometries if any
    regions = regions[~regions.geometry.is_empty & regions.geometry.notna()].copy()

    if regions.empty:
        raise ValueError("No valid geometries left in regions_gdf after cleaning.")

    return regions.reset_index(drop=True)


def build_bus_region_membership(
    buses: List[str],
    bus_lon: np.ndarray,
    bus_lat: np.ndarray,
    regions_gdf: gpd.GeoDataFrame,
    *,
    region_name_col: str = "name",
) -> pd.Series:
    """
    Assign each bus to one input region polygon.

    Strategy
    --------
    1. Try strict containment ("within").
    2. For unmatched buses, try topological intersection ("intersects").
    3. For still unmatched buses, assign the nearest region centroid.

    Returns
    -------
    pd.Series
        Index = bus names, values = assigned region identifiers.
    """
    regions = _prepare_regions_geodataframe(
        regions_gdf,
        region_name_col=region_name_col,
    )

    buses = list(map(str, buses))
    bus_lon = np.asarray(bus_lon, dtype=float)
    bus_lat = np.asarray(bus_lat, dtype=float)

    if len(buses) != len(bus_lon) or len(buses) != len(bus_lat):
        raise ValueError("buses, bus_lon, and bus_lat must have the same length.")

    points = gpd.GeoDataFrame(
        {"bus": buses},
        geometry=gpd.points_from_xy(bus_lon, bus_lat),
        crs="EPSG:4326",
    )

    membership = pd.Series(index=buses, dtype=object, name=region_name_col)

    # Step 1: strict containment
    joined_within = gpd.sjoin(
        points,
        regions[[region_name_col, "geometry"]],
        how="left",
        predicate="within",
    )

    if not joined_within.empty:
        tmp = (
            joined_within[["bus", region_name_col]]
            .dropna(subset=[region_name_col])
            .drop_duplicates(subset=["bus"], keep="first")
            .set_index("bus")[region_name_col]
        )
        membership.loc[tmp.index] = tmp

    # Step 2: fallback to intersects for buses on borders
    missing = membership[membership.isna()].index.tolist()
    if missing:
        joined_intersects = gpd.sjoin(
            points.loc[points["bus"].isin(missing)],
            regions[[region_name_col, "geometry"]],
            how="left",
            predicate="intersects",
        )

        if not joined_intersects.empty:
            tmp = (
                joined_intersects[["bus", region_name_col]]
                .dropna(subset=[region_name_col])
                .drop_duplicates(subset=["bus"], keep="first")
                .set_index("bus")[region_name_col]
            )
            membership.loc[tmp.index] = tmp

    # Step 3: nearest centroid in projected CRS
    missing = membership[membership.isna()].index.tolist()
    if missing:
        regions_proj = regions.to_crs("EPSG:3035")
        points_proj = points.to_crs("EPSG:3035")

        centroids_proj = regions_proj.copy()
        centroids_proj["geometry"] = centroids_proj.geometry.centroid

        nearest = gpd.sjoin_nearest(
            points_proj.loc[points_proj["bus"].isin(missing)],
            centroids_proj[[region_name_col, "geometry"]],
            how="left",
        )

        if not nearest.empty:
            tmp = (
                nearest[["bus", region_name_col]]
                .dropna(subset=[region_name_col])
                .drop_duplicates(subset=["bus"], keep="first")
                .set_index("bus")[region_name_col]
            )
            membership.loc[tmp.index] = tmp

    if membership.isna().any():
        bad = membership[membership.isna()].index.tolist()
        raise ValueError(
            f"Could not assign all buses to regions. Missing: {bad[:10]}"
            f"{'...' if len(bad) > 10 else ''}"
        )

    return membership.reindex(buses)


def build_region_adjacency_matrix(
    regions_gdf: gpd.GeoDataFrame,
    *,
    region_name_col: str = "name",
    include_self: bool = True,
) -> pd.DataFrame:
    """
    Build a symmetric adjacency matrix between input regions.

    Two regions are considered adjacent if their polygons touch or intersect.
    The use of intersects in addition to touches makes the procedure more
    robust to imperfect geometries.
    """
    regions = _prepare_regions_geodataframe(
        regions_gdf,
        region_name_col=region_name_col,
    )

    names = regions[region_name_col].tolist()
    n = len(names)

    A = np.zeros((n, n), dtype=bool)

    geoms = regions.geometry.values

    for i in range(n):
        gi = geoms[i]
        for j in range(i + 1, n):
            gj = geoms[j]
            if gi.touches(gj) or gi.intersects(gj):
                A[i, j] = True
                A[j, i] = True

    if include_self:
        np.fill_diagonal(A, True)

    return pd.DataFrame(A, index=names, columns=names)


def build_bus_connectivity_from_regions(
    buses: List[str],
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    regions_gdf: gpd.GeoDataFrame,
    *,
    region_name_col: str = "name",
) -> Tuple[sp.csr_matrix, pd.Series]:
    """
    Build sparse bus-to-bus connectivity matrix from input region adjacency.

    Two buses are connectable if:
    - they belong to the same region, or
    - their assigned regions are adjacent.

    Returns
    -------
    A_bus : scipy.sparse.csr_matrix
        Sparse connectivity matrix of shape (N, N).
    membership : pd.Series
        Mapping bus -> assigned region.
    """
    buses = list(map(str, buses))
    membership = build_bus_region_membership(
        buses=buses,
        bus_lon=np.asarray(lon_deg, dtype=float),
        bus_lat=np.asarray(lat_deg, dtype=float),
        regions_gdf=regions_gdf,
        region_name_col=region_name_col,
    )

    A_regions = build_region_adjacency_matrix(
        regions_gdf,
        region_name_col=region_name_col,
        include_self=True,
    )

    bus_regions = membership.reindex(buses).astype(str).to_numpy()
    region_idx = {r: i for i, r in enumerate(A_regions.index.astype(str))}
    region_ids = np.array([region_idx[r] for r in bus_regions], dtype=int)

    A_regions_np = A_regions.to_numpy(dtype=bool)
    A_bus_np = A_regions_np[np.ix_(region_ids, region_ids)]

    A_bus = sp.csr_matrix(A_bus_np.astype(np.float64))
    return A_bus, membership


# =============================================================================
# Distance utilities
# =============================================================================

def haversine_pairwise_km(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Pairwise Haversine distance matrix in km."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))

    R = 6371.0
    return R * c


def normalize_distance_matrix(
    D: np.ndarray,
    *,
    q: float = 0.95,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize distances to ~[0,1] using a robust scale (q-quantile of off-diagonal),
    then clip to [0,1]. Thresholds are interpreted in this normalized space.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("Distance matrix must be square.")

    offdiag = D[~np.eye(n, dtype=bool)]
    scale = np.quantile(offdiag, q) + eps
    return np.clip(D / scale, 0.0, 1.0)


# =============================================================================
# Tensor building + feature scaling
# =============================================================================

def build_tensor_X(
    data_hourly: Dict[str, np.ndarray],
    *,
    hours_per_day: int = 24,
    feature_mode: Literal["flatten", "daily_stats"] = "daily_stats",
    stats: Tuple[str, ...] = ("mean", "max", "min", "std", "ramp_max", "energy"),
) -> Tuple[np.ndarray, List[str]]:
    """
    Build X[node, day, feature] from hourly data.

    data_hourly: dict[attr] -> array (N, T_hours)

    feature_mode:
    - "flatten": features = hour-by-hour values per attribute (H * A)
    - "daily_stats": features per attribute computed from 24h profile

    Returns:
    - X: (N, D, F)
    - feature_names: length F
    """
    keys = list(data_hourly.keys())
    if len(keys) == 0:
        raise ValueError("data_hourly is empty.")

    first = np.asarray(data_hourly[keys[0]])
    if first.ndim != 2:
        raise ValueError("Each attribute array must be 2D: (N_nodes, T_hours).")

    N, T = first.shape
    if T % hours_per_day != 0:
        raise ValueError(f"T_hours={T} is not divisible by hours_per_day={hours_per_day}.")
    D = T // hours_per_day

    for k in keys:
        arr = np.asarray(data_hourly[k])
        if arr.shape != (N, T):
            raise ValueError(f"Attribute '{k}' has shape {arr.shape}, expected {(N, T)}.")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"Attribute '{k}' contains NaN/inf. Clean first.")

    X_attr = {
        k: np.asarray(data_hourly[k], dtype=float).reshape(N, D, hours_per_day)
        for k in keys
    }

    feature_names: List[str] = []
    blocks: List[np.ndarray] = []

    if feature_mode == "flatten":
        for k in keys:
            blocks.append(X_attr[k])  # (N, D, H)
            feature_names += [f"{k}_h{h:02d}" for h in range(hours_per_day)]
        X = np.concatenate(blocks, axis=2)  # (N, D, H*A)
        return X, feature_names

    if feature_mode == "daily_stats":
        for k in keys:
            x = X_attr[k]  # (N, D, H)
            feats = []
            names = []

            if "mean" in stats:
                feats.append(x.mean(axis=2, keepdims=True))
                names.append(f"{k}_mean")
            if "max" in stats:
                feats.append(x.max(axis=2, keepdims=True))
                names.append(f"{k}_max")
            if "min" in stats:
                feats.append(x.min(axis=2, keepdims=True))
                names.append(f"{k}_min")
            if "std" in stats:
                feats.append(x.std(axis=2, ddof=0, keepdims=True))
                names.append(f"{k}_std")
            if "ramp_max" in stats:
                ramp = np.abs(np.diff(x, axis=2)).max(axis=2, keepdims=True)
                feats.append(ramp)
                names.append(f"{k}_ramp_max")
            if "energy" in stats:
                energy = x.sum(axis=2, keepdims=True)
                feats.append(energy)
                names.append(f"{k}_energy")

            if len(feats) == 0:
                raise ValueError("No stats selected in daily_stats mode.")

            blocks.append(np.concatenate(feats, axis=2))
            feature_names += names

        X = np.concatenate(blocks, axis=2)
        return X, feature_names

    raise ValueError("feature_mode must be 'flatten' or 'daily_stats'.")


def zscore_global(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Feature-wise global z-score over all nodes and days."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=(0, 1), keepdims=True)
    sd = X.std(axis=(0, 1), keepdims=True) + eps
    return (X - mu) / sd


def minmax_global(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Feature-wise global min-max over all nodes and days -> [0,1]."""
    X = np.asarray(X, dtype=float)
    xmin = X.min(axis=(0, 1), keepdims=True)
    xmax = X.max(axis=(0, 1), keepdims=True)
    denom = (xmax - xmin) + eps
    Xn = (X - xmin) / denom
    return np.clip(Xn, 0.0, 1.0)


# =============================================================================
# Medoids + weights
# =============================================================================

def medoid_index(D_within: np.ndarray) -> int:
    """Medoid = argmin_i sum_j d(i,j)."""
    return int(np.argmin(D_within.sum(axis=1)))


def representative_medoids(
    D_full: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a distance matrix among items and cluster labels,
    return medoid indices and weights (= cluster sizes).
    """
    labels = np.asarray(labels, dtype=int)
    clusters = np.unique(labels)

    medoids = []
    weights = []
    for c in clusters:
        idx = np.where(labels == c)[0]
        D_within = D_full[np.ix_(idx, idx)]
        m_local = medoid_index(D_within)
        medoids.append(idx[m_local])
        weights.append(len(idx))

    return np.asarray(medoids, dtype=int), np.asarray(weights, dtype=int)


# =============================================================================
# Node distances and clustering
# =============================================================================

def build_node_ts_distance(X: np.ndarray, day_weights: np.ndarray) -> np.ndarray:
    """
    Weighted node distance using squared Euclidean on day-feature vectors.

    X: (N, Dsub, F)
    day_weights: (Dsub,) non-negative
    d_ts(i,j) = sqrt(sum_d w_d * ||X[i,d,:] - X[j,d,:]||^2)

    Implementation:
    - normalize weights to sum 1 (pure rescaling)
    - multiply each day block by sqrt(w_d)
    - flatten to (N, Dsub*F)
    - Euclidean distance
    """
    X = np.asarray(X, dtype=float)
    N, Dsub, F = X.shape

    w = np.asarray(day_weights, dtype=float)
    if w.shape != (Dsub,):
        raise ValueError(f"day_weights must have shape ({Dsub},), got {w.shape}.")
    if np.any(w < 0):
        raise ValueError("day_weights must be non-negative.")
    w = w / (w.sum() + 1e-12)

    Xw = X * np.sqrt(w)[None, :, None]
    Z = Xw.reshape(N, Dsub * F)
    return pairwise_distances(Z, metric="euclidean")


def _agglomerative_precomputed(
    D: np.ndarray,
    *,
    distance_threshold: float,
    linkage: Literal["average", "complete"],
    connectivity=None,
) -> np.ndarray:
    """Agglomerative clustering on precomputed distances with optional connectivity."""
    kwargs = dict(
        n_clusters=None,
        linkage=linkage,
        distance_threshold=float(distance_threshold),
    )

    if connectivity is not None:
        kwargs["connectivity"] = connectivity

    try:
        model = AgglomerativeClustering(
            metric="precomputed",
            **kwargs,
        )
    except TypeError:
        model = AgglomerativeClustering(
            affinity="precomputed",
            **kwargs,
        )

    return model.fit_predict(D).astype(int)


# =============================================================================
# Day representation and distances (optional PCA)
# =============================================================================

def aggregate_nodes_by_cluster(
    X: np.ndarray,
    labels_nodes: np.ndarray,
    *,
    node_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate X[N,D,F] into X_agg[K,D,F] by averaging nodes within each node cluster.
    Optionally weighted by node_weights (e.g., load, population).
    """
    X = np.asarray(X, dtype=float)
    labels_nodes = np.asarray(labels_nodes, dtype=int)

    N, D, F = X.shape
    if labels_nodes.shape != (N,):
        raise ValueError("labels_nodes must have shape (N,).")

    clusters = np.unique(labels_nodes)
    K = len(clusters)

    if node_weights is None:
        w = np.ones(N, dtype=float)
    else:
        w = np.asarray(node_weights, dtype=float)
        if w.shape != (N,):
            raise ValueError("node_weights must have shape (N,).")
        w = np.clip(w, 0.0, np.inf)

    X_agg = np.zeros((K, D, F), dtype=float)
    cluster_sizes = np.zeros(K, dtype=int)

    for k, c in enumerate(clusters):
        idx = np.where(labels_nodes == c)[0]
        cluster_sizes[k] = len(idx)
        ww = w[idx]
        ww_sum = ww.sum()
        if ww_sum <= 0:
            ww = np.ones_like(ww)
            ww_sum = ww.sum()
        X_agg[k, :, :] = np.tensordot(ww / ww_sum, X[idx, :, :], axes=(0, 0))

    return X_agg, cluster_sizes


def build_day_matrix(
    X_agg: np.ndarray,
    *,
    cluster_sizes: Optional[np.ndarray] = None,
    standardize_cols: bool = False,
) -> np.ndarray:
    """
    Build day matrix Y[d, :] = vec(X_agg[:, d, :]) with optional node-cluster block weighting.

    X_agg: (K, D, F)
    Returns Y: (D, K*F)

    cluster_sizes:
      - if provided, apply block weights sqrt(cs_norm) replicated F times.
      - cs_norm = cs / sum(cs) (so weights are scale-stable).
    """
    X_agg = np.asarray(X_agg, dtype=float)
    K, D, F = X_agg.shape

    Y = np.transpose(X_agg, (1, 0, 2)).reshape(D, K * F)

    if cluster_sizes is not None:
        cs = np.asarray(cluster_sizes, dtype=float)
        if cs.shape != (K,):
            raise ValueError("cluster_sizes must have shape (K,).")
        cs_norm = cs / (cs.sum() + 1e-12)
        w_block = np.repeat(np.sqrt(cs_norm), F)
        Y = Y * w_block[None, :]

    if standardize_cols:
        mu = Y.mean(axis=0, keepdims=True)
        sd = Y.std(axis=0, keepdims=True) + 1e-12
        Y = (Y - mu) / sd

    return Y


def build_day_distance(
    X_agg: np.ndarray,
    *,
    cluster_sizes: Optional[np.ndarray] = None,
    use_pca: bool = False,
    pca_n_components: Union[int, float] = 0.95,
    pca_random_state: int = 0,
    standardize_day_matrix_cols: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Compute day-to-day distances on system-state vectors, with optional PCA.
    """
    Y = build_day_matrix(
        X_agg,
        cluster_sizes=cluster_sizes,
        standardize_cols=standardize_day_matrix_cols,
    )

    info: dict = {}

    if use_pca:
        if isinstance(pca_n_components, float):
            if not (0.0 < pca_n_components <= 1.0):
                raise ValueError("pca_n_components as float must be in (0,1].")
            pca = PCA(
                n_components=pca_n_components,
                svd_solver="full",
                random_state=pca_random_state,
            )
        else:
            if int(pca_n_components) < 1:
                raise ValueError("pca_n_components as int must be >= 1.")
            pca = PCA(
                n_components=int(pca_n_components),
                svd_solver="randomized",
                random_state=pca_random_state,
            )

        Z = pca.fit_transform(Y)
        D_day_raw = pairwise_distances(Z, metric="euclidean")

        evr = pca.explained_variance_ratio_
        info = dict(
            use_pca=True,
            pca_n_components_selected=int(getattr(pca, "n_components_", Z.shape[1])),
            pca_explained_variance_ratio_sum=float(np.sum(evr)),
        )
    else:
        D_day_raw = pairwise_distances(Y, metric="euclidean")
        info = dict(use_pca=False)

    return D_day_raw, info


# =============================================================================
# Alternating reducer
# =============================================================================


@dataclass
class ReductionResult:
    labels_nodes: np.ndarray
    labels_days: np.ndarray
    rep_nodes: np.ndarray
    rep_node_weights: np.ndarray
    rep_days: np.ndarray
    rep_weights: np.ndarray
    history: List[dict]
    evaluations: List[dict]
    day_pca_info: dict
    objective: float
    region_membership: Optional[pd.Series] = None


def _relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    """Relabel cluster ids to 0..K-1 preserving order of first appearance."""
    labels = np.asarray(labels, dtype=int)
    uniques = []
    mapping = {}
    out = np.empty_like(labels)
    for i, val in enumerate(labels):
        if int(val) not in mapping:
            mapping[int(val)] = len(mapping)
            uniques.append(int(val))
        out[i] = mapping[int(val)]
    return out.astype(int)


def _cluster_sizes_from_labels(labels: np.ndarray) -> np.ndarray:
    """Return cluster sizes for contiguous labels."""
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return np.zeros(0, dtype=int)
    return np.bincount(labels, minlength=int(labels.max()) + 1).astype(int)


def _medoids_from_labels(D: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return medoid indices and cluster sizes for a labeled partition."""
    labels = _relabel_contiguous(labels)
    K = int(labels.max()) + 1 if labels.size else 0
    medoids = np.zeros(K, dtype=int)
    sizes = np.zeros(K, dtype=int)

    for k in range(K):
        idx = np.where(labels == k)[0]
        sizes[k] = len(idx)
        if len(idx) == 1:
            medoids[k] = int(idx[0])
            continue
        D_within = D[np.ix_(idx, idx)]
        medoids[k] = int(idx[int(np.argmin(D_within.sum(axis=1)))])
    return medoids, sizes


def _farthest_point_from_set(D: np.ndarray, chosen: List[int]) -> int:
    """Pick the point farthest from the current chosen set."""
    n = D.shape[0]
    remaining = np.setdiff1d(np.arange(n, dtype=int), np.asarray(chosen, dtype=int), assume_unique=False)
    if len(remaining) == 0:
        return int(chosen[-1])
    if len(chosen) == 0:
        return int(remaining[0])
    dist_to_set = D[np.ix_(remaining, np.asarray(chosen, dtype=int))].min(axis=1)
    return int(remaining[int(np.argmax(dist_to_set))])


def _initialize_medoids_greedy(D: np.ndarray, k: int) -> np.ndarray:
    """Greedy farthest-point initialization for k-medoids on a precomputed distance matrix."""
    n = D.shape[0]
    if k >= n:
        return np.arange(n, dtype=int)

    total = D.sum(axis=1)
    medoids = [int(np.argmin(total))]
    while len(medoids) < k:
        medoids.append(_farthest_point_from_set(D, medoids))
    return np.asarray(medoids, dtype=int)


def kmedoids_precomputed(
    D: np.ndarray,
    k: int,
    *,
    max_iter: int = 100,
    random_state: int = 0,
    init_medoids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple PAM-style k-medoids on a precomputed distance matrix.

    Returns
    -------
    labels : np.ndarray
        Contiguous labels 0..k-1.
    medoids : np.ndarray
        Indices of the medoid points, aligned with label ids.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("Distance matrix must be square.")
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}.")

    rng = np.random.RandomState(random_state)

    if init_medoids is not None:
        medoids = np.asarray(init_medoids, dtype=int).copy()
        if len(np.unique(medoids)) != k:
            raise ValueError("init_medoids must contain k unique indices.")
    else:
        medoids = _initialize_medoids_greedy(D, k)
        if len(medoids) != k:
            remaining = np.setdiff1d(np.arange(n, dtype=int), medoids, assume_unique=False)
            rng.shuffle(remaining)
            medoids = np.concatenate([medoids, remaining[: k - len(medoids)]])

    medoids = np.asarray(medoids, dtype=int)

    for _ in range(max_iter):
        d_to_medoids = D[:, medoids]
        labels = np.argmin(d_to_medoids, axis=1).astype(int)

        # Fix potential empty clusters by re-seeding with farthest points
        counts = np.bincount(labels, minlength=k)
        if np.any(counts == 0):
            missing = np.where(counts == 0)[0]
            used_points = set(medoids.tolist())
            for cl in missing:
                new_medoid = _farthest_point_from_set(D, list(used_points))
                medoids[cl] = new_medoid
                used_points.add(int(new_medoid))
            d_to_medoids = D[:, medoids]
            labels = np.argmin(d_to_medoids, axis=1).astype(int)

        new_medoids = medoids.copy()
        changed = False

        for cl in range(k):
            idx = np.where(labels == cl)[0]
            if len(idx) == 0:
                continue
            D_within = D[np.ix_(idx, idx)]
            best_local = int(np.argmin(D_within.sum(axis=1)))
            best_point = int(idx[best_local])
            if best_point != medoids[cl]:
                new_medoids[cl] = best_point
                changed = True

        medoids = new_medoids

        if not changed:
            break

    labels = np.argmin(D[:, medoids], axis=1).astype(int)

    # Reorder clusters by medoid index for determinism
    order = np.argsort(medoids)
    medoids = medoids[order]
    inv = np.empty_like(order)
    inv[order] = np.arange(k)
    labels = inv[labels]

    return labels.astype(int), medoids.astype(int)


def reconstruct_tensor_from_medoids(
    X: np.ndarray,
    *,
    rep_nodes: np.ndarray,
    labels_nodes: np.ndarray,
    rep_days: np.ndarray,
    labels_days: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct X[N,D,F] by repeating spatial and temporal medoids.

    For each original pair (n, d), the reconstructed value is taken from:
    X[rep_nodes[label_nodes[n]], rep_days[label_days[d]], :].
    """
    X = np.asarray(X, dtype=float)
    rep_nodes = np.asarray(rep_nodes, dtype=int)
    labels_nodes = np.asarray(labels_nodes, dtype=int)
    rep_days = np.asarray(rep_days, dtype=int)
    labels_days = np.asarray(labels_days, dtype=int)

    node_src = rep_nodes[labels_nodes]
    day_src = rep_days[labels_days]

    return X[node_src[:, None], day_src[None, :], :]


def weighted_reconstruction_loss(
    X: np.ndarray,
    X_rec: np.ndarray,
    *,
    feature_weights: Optional[np.ndarray] = None,
    node_loss_weights: Optional[np.ndarray] = None,
) -> float:
    """
    Weighted reconstruction loss on X[N,D,F].

    The loss is:
        sum_n w_n * sum_f w_f * sum_d (X - X_rec)^2

    Feature weights and node weights are internally normalized to have mean 1.
    """
    X = np.asarray(X, dtype=float)
    X_rec = np.asarray(X_rec, dtype=float)

    if X.shape != X_rec.shape:
        raise ValueError("X and X_rec must have the same shape.")

    N, _, F = X.shape
    err2 = (X - X_rec) ** 2

    if feature_weights is None:
        wf = np.ones(F, dtype=float)
    else:
        wf = np.asarray(feature_weights, dtype=float)
        if wf.shape != (F,):
            raise ValueError(f"feature_weights must have shape ({F},), got {wf.shape}.")
        if np.any(wf < 0):
            raise ValueError("feature_weights must be non-negative.")
        wf = wf / (wf.mean() + 1e-12)

    if node_loss_weights is None:
        wn = np.ones(N, dtype=float)
    else:
        wn = np.asarray(node_loss_weights, dtype=float)
        if wn.shape != (N,):
            raise ValueError(f"node_loss_weights must have shape ({N},), got {wn.shape}.")
        if np.any(wn < 0):
            raise ValueError("node_loss_weights must be non-negative.")
        wn = wn / (wn.mean() + 1e-12)

    err2 = err2 * wf[None, None, :]
    err2 = err2.sum(axis=(1, 2))
    return float(np.sum(wn * err2))


class AlternatingSpatioTemporalReducer:
    """
    Budget-based alternating geo-temporal reducer.

    Main ideas
    ----------
    - The total geo-temporal budget is controlled through max_total_steps.
    - lambda_ts remains an external hyperparameter and only affects the
      spatial distance matrix.
    - Spatial and temporal clustering use k-medoids on precomputed distances.
    - The accepted solution at each iteration is the one with the lowest
      reconstruction loss among the tested candidates.
    - Two initialization modes are supported:
      * "balanced": start directly from a near-balanced pair (K_nodes, K_days)
        under the target budget.
      * "full": start from full resolution and progressively reduce the budget
        using beta until the target budget is reached.
    """

    def __init__(
        self,
        *,
        lambda_ts: float = 0.85,
        normalize: Literal["zscore", "minmax"] = "zscore",
        max_total_steps: int = 144,
        init_mode: Literal["balanced", "full"] = "balanced",
        init_nodes: Optional[int] = None,
        init_days: Optional[int] = None,
        beta: float = 0.15,
        max_iter: int = 20,
        tol_no_change: int = 2,
        objective_tol_rel: float = 1e-5,
        verbose: bool = True,
        norm_q: float = 0.95,
        use_pca_days: bool = False,
        pca_days_n_components: Union[int, float] = 0.95,
        pca_days_random_state: int = 0,
        standardize_day_matrix_cols: bool = False,
        kmedoids_max_iter: int = 100,
        random_state: int = 0,
        enforce_spatial_adjacency: bool = False,
        regions_gdf: Optional[gpd.GeoDataFrame] = None,
        region_name_col: str = "name",
        feature_weights: Optional[np.ndarray] = None,
    ):
        self.lambda_ts = float(lambda_ts)
        self.normalize = normalize
        self.max_total_steps = int(max_total_steps)
        self.init_mode = str(init_mode)
        self.init_nodes = None if init_nodes is None else int(init_nodes)
        self.init_days = None if init_days is None else int(init_days)
        self.beta = float(beta)
        self.max_iter = int(max_iter)
        self.tol_no_change = int(tol_no_change)
        self.objective_tol_rel = float(objective_tol_rel)
        self.verbose = bool(verbose)
        self.norm_q = float(norm_q)

        self.use_pca_days = bool(use_pca_days)
        self.pca_days_n_components = pca_days_n_components
        self.pca_days_random_state = int(pca_days_random_state)
        self.standardize_day_matrix_cols = bool(standardize_day_matrix_cols)

        self.kmedoids_max_iter = int(kmedoids_max_iter)
        self.random_state = int(random_state)

        self.enforce_spatial_adjacency = bool(enforce_spatial_adjacency)
        self.regions_gdf = regions_gdf
        self.region_name_col = str(region_name_col)

        self.feature_weights = feature_weights

    def _balanced_pair(self, N: int, D: int, budget: int) -> Tuple[int, int]:
        """Find a near-balanced integer pair (K_nodes, K_days) under the budget."""
        best_pair = (1, 1)
        best_budget = 1
        best_score = np.inf

        max_kn = min(N, max(1, budget))
        for kn in range(1, max_kn + 1):
            kd = min(D, max(1, budget // kn))
            used = kn * kd
            score = abs(np.log(kn + 1e-12) - np.log(kd + 1e-12))
            if used > best_budget or (used == best_budget and score < best_score):
                best_pair = (kn, kd)
                best_budget = used
                best_score = score

        return best_pair

    def _initialize_pair(self, N: int, D: int) -> Tuple[int, int]:
        """Choose the initial pair (K_nodes, K_days)."""
        full_budget = N * D
        target_budget = min(self.max_total_steps, full_budget)

        if self.init_mode == "full":
            return N, D

        if self.init_nodes is not None and self.init_days is not None:
            kn = min(N, max(1, self.init_nodes))
            kd = min(D, max(1, self.init_days))
            if kn * kd > target_budget:
                kd = max(1, min(D, target_budget // kn))
            return kn, kd

        if self.init_nodes is not None:
            kn = min(N, max(1, self.init_nodes))
            kd = max(1, min(D, target_budget // kn))
            return kn, kd

        if self.init_days is not None:
            kd = min(D, max(1, self.init_days))
            kn = max(1, min(N, target_budget // kd))
            return kn, kd

        return self._balanced_pair(N, D, target_budget)

    def _build_spatial_distance(
        self,
        Xn: np.ndarray,
        D_geo_n: np.ndarray,
        rep_days: np.ndarray,
        rep_weights: np.ndarray,
    ) -> np.ndarray:
        """Build the combined node distance matrix conditioned on the current temporal mapping."""
        X_for_nodes = Xn[:, rep_days, :]
        D_ts_raw = build_node_ts_distance(X_for_nodes, rep_weights.astype(float))
        D_ts_n = normalize_distance_matrix(D_ts_raw, q=self.norm_q)
        return self.lambda_ts * D_ts_n + (1.0 - self.lambda_ts) * D_geo_n

    def _solve_for_pair(
        self,
        Xn: np.ndarray,
        D_geo_n: np.ndarray,
        rep_days_seed: np.ndarray,
        rep_weights_seed: np.ndarray,
        K_nodes: int,
        K_days: int,
        *,
        node_loss_weights: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Solve one alternating step for a fixed pair (K_nodes, K_days).

        Spatial clustering is conditioned on the provided temporal seed
        (representative days and weights). Temporal clustering is then computed
        on the medoid-based spatial representation.
        """
        D_node = self._build_spatial_distance(Xn, D_geo_n, rep_days_seed, rep_weights_seed)
        labels_nodes, rep_nodes = kmedoids_precomputed(
            D_node,
            K_nodes,
            max_iter=self.kmedoids_max_iter,
            random_state=self.random_state,
        )
        labels_nodes = _relabel_contiguous(labels_nodes)
        rep_node_weights = _cluster_sizes_from_labels(labels_nodes)

        X_rep_nodes = Xn[rep_nodes, :, :]
        D_day_raw, day_pca_info = build_day_distance(
            X_rep_nodes,
            cluster_sizes=rep_node_weights,
            use_pca=self.use_pca_days,
            pca_n_components=self.pca_days_n_components,
            pca_random_state=self.pca_days_random_state,
            standardize_day_matrix_cols=self.standardize_day_matrix_cols,
        )
        D_day_n = normalize_distance_matrix(D_day_raw, q=self.norm_q)

        labels_days, rep_days = kmedoids_precomputed(
            D_day_n,
            K_days,
            max_iter=self.kmedoids_max_iter,
            random_state=self.random_state,
        )
        labels_days = _relabel_contiguous(labels_days)
        rep_weights = _cluster_sizes_from_labels(labels_days)

        X_rec = reconstruct_tensor_from_medoids(
            Xn,
            rep_nodes=rep_nodes,
            labels_nodes=labels_nodes,
            rep_days=rep_days,
            labels_days=labels_days,
        )
        objective = weighted_reconstruction_loss(
            Xn,
            X_rec,
            feature_weights=self.feature_weights,
            node_loss_weights=node_loss_weights,
        )

        return dict(
            K_nodes=int(K_nodes),
            K_days=int(K_days),
            labels_nodes=labels_nodes,
            labels_days=labels_days,
            rep_nodes=rep_nodes.astype(int),
            rep_node_weights=rep_node_weights.astype(int),
            rep_days=rep_days.astype(int),
            rep_weights=rep_weights.astype(int),
            objective=float(objective),
            day_pca_info=day_pca_info,
            total_steps=int(K_nodes * K_days),
        )

    def _candidate_pairs_reduce_budget(
        self,
        K_nodes: int,
        K_days: int,
        next_budget: int,
        N: int,
        D: int,
    ) -> List[Tuple[int, int, str]]:
        """Generate candidate pairs when the current budget must be reduced."""
        candidates: List[Tuple[int, int, str]] = []

        kn_red = max(1, min(N, next_budget // max(1, K_days)))
        if kn_red < K_nodes:
            candidates.append((kn_red, K_days, "reduce_space"))

        kd_red = max(1, min(D, next_budget // max(1, K_nodes)))
        if kd_red < K_days:
            candidates.append((K_nodes, kd_red, "reduce_time"))

        if not candidates:
            kn_bal, kd_bal = self._balanced_pair(N, D, next_budget)
            candidates.append((kn_bal, kd_bal, "reduce_balanced"))

        return candidates

    def _candidate_pairs_rebalance(
        self,
        K_nodes: int,
        K_days: int,
        target_budget: int,
        N: int,
        D: int,
    ) -> List[Tuple[int, int, str]]:
        """Generate candidate pairs around the current one while staying under the target budget."""
        candidates: List[Tuple[int, int, str]] = []

        kn_up = min(N, max(K_nodes + 1, int(np.ceil(K_nodes * (1.0 + self.beta)))))
        kd_for_kn_up = max(1, min(D, target_budget // kn_up))
        if (kn_up, kd_for_kn_up) != (K_nodes, K_days):
            candidates.append((kn_up, kd_for_kn_up, "more_space"))

        kd_up = min(D, max(K_days + 1, int(np.ceil(K_days * (1.0 + self.beta)))))
        kn_for_kd_up = max(1, min(N, target_budget // kd_up))
        if (kn_for_kd_up, kd_up) != (K_nodes, K_days):
            candidates.append((kn_for_kd_up, kd_up, "more_time"))

        return candidates

    def fit(
        self,
        X: np.ndarray,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        *,
        buses: Optional[List[str]] = None,
        node_weights: Optional[np.ndarray] = None,
    ) -> ReductionResult:
        X = np.asarray(X, dtype=float)
        N, D, F = X.shape

        lat_deg = np.asarray(lat_deg, dtype=float)
        lon_deg = np.asarray(lon_deg, dtype=float)
        if lat_deg.shape != (N,) or lon_deg.shape != (N,):
            raise ValueError("lat_deg and lon_deg must have shape (N,).")

        if self.feature_weights is not None:
            fw = np.asarray(self.feature_weights, dtype=float)
            if fw.shape != (F,):
                raise ValueError(f"feature_weights must have shape ({F},), got {fw.shape}.")

        if buses is None:
            raise ValueError("buses must be provided when fitting the spatial reducer.")
        if len(buses) != N:
            raise ValueError("buses must have length N.")

        if self.enforce_spatial_adjacency:
            raise NotImplementedError(
                "Spatial adjacency constraints are not yet implemented for the new "
                "k-medoids budget-based reducer."
            )

        if self.normalize == "zscore":
            Xn = zscore_global(X)
        elif self.normalize == "minmax":
            Xn = minmax_global(X)
        else:
            raise ValueError("normalize must be 'zscore' or 'minmax'.")

        D_geo = haversine_pairwise_km(lat_deg, lon_deg)
        D_geo_n = normalize_distance_matrix(D_geo, q=self.norm_q)

        if node_weights is None:
            node_loss_weights = None
        else:
            node_loss_weights = np.asarray(node_weights, dtype=float)
            if node_loss_weights.shape != (N,):
                raise ValueError("node_weights must have shape (N,).")

        target_budget = min(self.max_total_steps, N * D)

        K_nodes, K_days = self._initialize_pair(N, D)
        current_budget = int(K_nodes * K_days)

        evaluations: List[dict] = []
        history: List[dict] = []

        rep_days_seed = np.arange(D, dtype=int)
        rep_weights_seed = np.ones(D, dtype=int)

        if current_budget == N * D:
            current_solution = dict(
                K_nodes=int(N),
                K_days=int(D),
                labels_nodes=np.arange(N, dtype=int),
                labels_days=np.arange(D, dtype=int),
                rep_nodes=np.arange(N, dtype=int),
                rep_node_weights=np.ones(N, dtype=int),
                rep_days=np.arange(D, dtype=int),
                rep_weights=np.ones(D, dtype=int),
                objective=0.0,
                day_pca_info=dict(use_pca=False),
                total_steps=int(N * D),
            )
        else:
            current_solution = self._solve_for_pair(
                Xn,
                D_geo_n,
                rep_days_seed,
                rep_weights_seed,
                K_nodes,
                K_days,
                node_loss_weights=node_loss_weights,
            )

        no_improve_counter = 0
        it_global = 0
        it_feasible = 0

        # Safety guard only for pathological cases during infeasible reduction
        max_infeasible_iters = max(10 * self.max_iter, 50)

        # ------------------------------------------------------------------
        # Phase 1: mandatory reduction until the solution is feasible
        # ------------------------------------------------------------------
        while int(current_solution["total_steps"]) > target_budget:
            current_budget = int(current_solution["total_steps"])
            current_obj = float(current_solution["objective"])

            history.append(
                dict(
                    iter=int(it_global),
                    phase="reduce_to_budget",
                    status="accepted_infeasible",
                    K_nodes=int(current_solution["K_nodes"]),
                    K_days=int(current_solution["K_days"]),
                    total_steps=int(current_solution["total_steps"]),
                    objective=float(current_obj),
                )
            )

            if self.verbose:
                print(
                    f"[Iter {it_global}] accepted_infeasible | "
                    f"K_nodes={current_solution['K_nodes']} | "
                    f"K_days={current_solution['K_days']} | "
                    f"steps={current_solution['total_steps']} | "
                    f"objective={current_obj:.6e}"
                )

            next_budget = max(
                target_budget,
                int(np.floor(current_budget * (1.0 - self.beta)))
            )

            candidate_pairs = self._candidate_pairs_reduce_budget(
                int(current_solution["K_nodes"]),
                int(current_solution["K_days"]),
                next_budget,
                N,
                D,
            )

            if len(candidate_pairs) == 0:
                # Hard fallback: jump directly to a balanced feasible pair
                kn_bal, kd_bal = self._balanced_pair(N, D, target_budget)
                candidate_pairs = [(kn_bal, kd_bal, "force_feasible_balanced")]

            seed_days = current_solution["rep_days"]
            seed_weights = current_solution["rep_weights"]

            candidate_solutions = []
            for cand_kn, cand_kd, move_type in candidate_pairs:
                cand = self._solve_for_pair(
                    Xn,
                    D_geo_n,
                    seed_days,
                    seed_weights,
                    cand_kn,
                    cand_kd,
                    node_loss_weights=node_loss_weights,
                )
                cand["move_type"] = move_type
                candidate_solutions.append(cand)

                evaluations.append(
                    dict(
                        iter=int(it_global),
                        phase="reduce_to_budget",
                        move_type=str(move_type),
                        K_nodes=int(cand["K_nodes"]),
                        K_days=int(cand["K_days"]),
                        total_steps=int(cand["total_steps"]),
                        objective=float(cand["objective"]),
                    )
                )

                if self.verbose:
                    print(
                        f"    candidate={move_type:>22s} | "
                        f"K_nodes={cand['K_nodes']} | "
                        f"K_days={cand['K_days']} | "
                        f"steps={cand['total_steps']} | "
                        f"objective={cand['objective']:.6e}"
                    )

            # Among reduction candidates, prefer feasibility first, then lower objective
            feasible_candidates = [
                c for c in candidate_solutions if int(c["total_steps"]) <= target_budget
            ]

            if feasible_candidates:
                best_candidate = min(feasible_candidates, key=lambda s: float(s["objective"]))
            else:
                # If none is feasible yet, keep reducing. Prefer the smallest total_steps,
                # then the lowest objective among those.
                best_candidate = min(
                    candidate_solutions,
                    key=lambda s: (int(s["total_steps"]), float(s["objective"])),
                )

            current_solution = best_candidate
            it_global += 1

            if it_global >= max_infeasible_iters and int(current_solution["total_steps"]) > target_budget:
                raise RuntimeError(
                    "The reducer could not reach a feasible solution under max_total_steps. "
                    f"Current total_steps={current_solution['total_steps']}, "
                    f"target={target_budget}."
                )

        # Log the first feasible solution before starting local rebalancing
        current_budget = int(current_solution["total_steps"])
        current_obj = float(current_solution["objective"])
        history.append(
            dict(
                iter=int(it_global),
                phase="feasible_search",
                status="accepted_feasible",
                K_nodes=int(current_solution["K_nodes"]),
                K_days=int(current_solution["K_days"]),
                total_steps=int(current_solution["total_steps"]),
                objective=float(current_obj),
            )
        )

        if self.verbose:
            print(
                f"[Iter {it_global}] accepted_feasible | "
                f"K_nodes={current_solution['K_nodes']} | "
                f"K_days={current_solution['K_days']} | "
                f"steps={current_solution['total_steps']} | "
                f"objective={current_obj:.6e}"
            )

        # ------------------------------------------------------------------
        # Phase 2: local search within the feasible region
        # ------------------------------------------------------------------
        while it_feasible < self.max_iter:
            current_budget = int(current_solution["total_steps"])
            current_obj = float(current_solution["objective"])

            candidate_pairs = self._candidate_pairs_rebalance(
                int(current_solution["K_nodes"]),
                int(current_solution["K_days"]),
                target_budget,
                N,
                D,
            )

            if len(candidate_pairs) == 0:
                if self.verbose:
                    print("[Stop] No new feasible candidate pairs available.")
                break

            seed_days = current_solution["rep_days"]
            seed_weights = current_solution["rep_weights"]

            candidate_solutions = []
            for cand_kn, cand_kd, move_type in candidate_pairs:
                cand = self._solve_for_pair(
                    Xn,
                    D_geo_n,
                    seed_days,
                    seed_weights,
                    cand_kn,
                    cand_kd,
                    node_loss_weights=node_loss_weights,
                )
                cand["move_type"] = move_type
                candidate_solutions.append(cand)

                evaluations.append(
                    dict(
                        iter=int(it_global + 1),
                        phase="feasible_search",
                        move_type=str(move_type),
                        K_nodes=int(cand["K_nodes"]),
                        K_days=int(cand["K_days"]),
                        total_steps=int(cand["total_steps"]),
                        objective=float(cand["objective"]),
                    )
                )

                if self.verbose:
                    print(
                        f"    candidate={move_type:>14s} | "
                        f"K_nodes={cand['K_nodes']} | "
                        f"K_days={cand['K_days']} | "
                        f"steps={cand['total_steps']} | "
                        f"objective={cand['objective']:.6e}"
                    )

            best_candidate = min(candidate_solutions, key=lambda s: float(s["objective"]))
            best_obj = float(best_candidate["objective"])

            rel_improvement = (current_obj - best_obj) / max(abs(current_obj), 1e-12)

            if best_obj < current_obj and rel_improvement > self.objective_tol_rel:
                current_solution = best_candidate
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                if self.verbose:
                    print(
                        f"[No improvement] best_candidate={best_obj:.6e} | "
                        f"current={current_obj:.6e} | "
                        f"rel_improvement={rel_improvement:.3e}"
                    )
                if no_improve_counter >= self.tol_no_change:
                    if self.verbose:
                        print(
                            f"[Converged] No relevant improvement for "
                            f"{no_improve_counter} consecutive feasible iterations."
                        )
                    break

            it_global += 1
            it_feasible += 1

            history.append(
                dict(
                    iter=int(it_global),
                    phase="feasible_search",
                    status="accepted_feasible",
                    K_nodes=int(current_solution["K_nodes"]),
                    K_days=int(current_solution["K_days"]),
                    total_steps=int(current_solution["total_steps"]),
                    objective=float(current_solution["objective"]),
                )
            )

            if self.verbose:
                print(
                    f"[Iter {it_global}] accepted_feasible | "
                    f"K_nodes={current_solution['K_nodes']} | "
                    f"K_days={current_solution['K_days']} | "
                    f"steps={current_solution['total_steps']} | "
                    f"objective={current_solution['objective']:.6e}"
                )

        # Final hard guarantee
        if int(current_solution["total_steps"]) > target_budget:
            raise RuntimeError(
                "The final solution is infeasible: "
                f"total_steps={current_solution['total_steps']} > max_total_steps={target_budget}."
            )

        return ReductionResult(
            labels_nodes=current_solution["labels_nodes"].astype(int),
            labels_days=current_solution["labels_days"].astype(int),
            rep_nodes=current_solution["rep_nodes"].astype(int),
            rep_node_weights=current_solution["rep_node_weights"].astype(int),
            rep_days=current_solution["rep_days"].astype(int),
            rep_weights=current_solution["rep_weights"].astype(int),
            history=history,
            evaluations=evaluations,
            day_pca_info=current_solution["day_pca_info"],
            objective=float(current_solution["objective"]),
            region_membership=None,
        )


# =============================================================================
# PyPSA helpers (feature extraction + mapping)
# =============================================================================
# =============================================================================

def select_buses_from_loads(
    n,
    *,
    exclude_bus_substrings: Tuple[str, ...] = (" H2", "battery"),
) -> List[str]:
    """Return sorted list of buses referenced by loads, excluding auxiliary buses by name."""
    if n.loads.empty:
        raise ValueError("n.loads is empty. Cannot infer buses from loads.")
    buses = n.loads["bus"].astype(str).unique().tolist()
    keep = []
    for b in buses:
        if any(s in b for s in exclude_bus_substrings):
            continue
        keep.append(b)
    keep = sorted(set(keep))
    if len(keep) == 0:
        raise ValueError("No buses left after filtering.")
    return keep


def bus_coords_latlon(n, buses: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return lat, lon arrays for buses.
    In PyPSA-Eur, n.buses has x=lon and y=lat.
    """
    missing = [b for b in buses if b not in n.buses.index]
    if missing:
        raise KeyError(f"Missing buses in n.buses: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    lon = n.buses.loc[buses, "x"].astype(float).to_numpy()
    lat = n.buses.loc[buses, "y"].astype(float).to_numpy()
    return lat, lon


def loads_by_bus_timeseries(n, buses: List[str]) -> pd.DataFrame:
    """Build load time series per bus (MW): sum all loads connected to each bus."""
    if n.loads.empty:
        raise ValueError("n.loads is empty.")
    if not hasattr(n, "loads_t") or getattr(n.loads_t, "p_set", None) is None:
        raise ValueError("n.loads_t.p_set not available.")

    snaps = n.snapshots
    L = n.loads.copy()
    L = L[L["bus"].isin(buses)]
    if L.empty:
        raise ValueError("No loads found on selected buses.")

    p_set = n.loads_t.p_set.reindex(index=snaps, columns=L.index)

    out = pd.DataFrame(index=snaps, columns=buses, dtype=float)
    for b in buses:
        load_names = L.index[L["bus"] == b]
        if len(load_names) == 0:
            out[b] = 0.0
        else:
            out[b] = p_set[load_names].sum(axis=1)

    return out


def cf_by_bus_timeseries(
    n,
    buses: List[str],
    *,
    carrier: str,
    weight_by: str = "p_nom",
) -> pd.DataFrame:
    """
    Build capacity factor time series per bus by aggregating generators with a given carrier.

    CF(bus,t) = sum_g w_g * p_max_pu_g(t) / sum_g w_g for gens g at that bus.

    For buses with no such gens -> 0.
    """
    if getattr(n, "generators", None) is None or n.generators.empty:
        return pd.DataFrame(0.0, index=n.snapshots, columns=buses)

    if not hasattr(n, "generators_t") or getattr(n.generators_t, "p_max_pu", None) is None:
        raise ValueError("n.generators_t.p_max_pu not available.")

    snaps = n.snapshots
    G = n.generators.copy()
    G = G[(G["bus"].isin(buses)) & (G["carrier"].astype(str) == carrier)]

    p_max_pu = n.generators_t.p_max_pu.reindex(index=snaps)

    out = pd.DataFrame(index=snaps, columns=buses, dtype=float)

    if G.empty:
        out.loc[:, :] = 0.0
        return out

    if weight_by not in G.columns:
        weights = pd.Series(1.0, index=G.index)
    else:
        weights = G[weight_by].astype(float).copy()
        if np.isclose(weights.sum(), 0.0):
            weights[:] = 1.0

    for b in buses:
        gens = G.index[G["bus"] == b]
        if len(gens) == 0:
            out[b] = 0.0
            continue

        cf = p_max_pu.reindex(columns=gens)
        w = weights.reindex(gens).to_numpy(dtype=float)

        wsum = float(np.sum(w))
        if wsum <= 0.0:
            out[b] = cf.mean(axis=1)
        else:
            out[b] = (cf.to_numpy() * w[None, :]).sum(axis=1) / wsum

    return out.clip(lower=0.0, upper=1.0)


def strip_suffix(bus: str) -> Tuple[str, str]:
    """Return (base, suffix) where suffix is '', ' H2', or ' battery'."""
    if bus.endswith(" H2"):
        return bus[:-3], " H2"
    if bus.endswith(" battery"):
        return bus[:-8], " battery"
    return bus, ""


def build_full_busmap(
    n,
    base_buses: List[str],
    labels_nodes: np.ndarray,
    rep_nodes_idx: np.ndarray,
) -> pd.Series:
    """
    Build busmap for ALL buses in n.buses.

    Strategy:
    - Cluster is defined on base_buses (typically AC buses).
    - Each cluster is represented by the medoid bus name (rep_base).
    - Auxiliary buses (e.g., '<base> H2', '<base> battery') are mapped to
      '<rep_base> H2' / '<rep_base> battery' to preserve carrier separation.
    """
    base_buses = list(base_buses)
    labels_nodes = np.asarray(labels_nodes, dtype=int)
    rep_nodes_idx = np.asarray(rep_nodes_idx, dtype=int)

    rep_base_by_cluster: Dict[int, str] = {}
    for idx in rep_nodes_idx:
        c = int(labels_nodes[idx])
        rep_base_by_cluster[c] = base_buses[idx]

    base_to_rep: Dict[str, str] = {}
    for b, c in zip(base_buses, labels_nodes):
        base_to_rep[b] = rep_base_by_cluster[int(c)]

    mapping = {}
    for bus in n.buses.index.astype(str):
        base, suffix = strip_suffix(bus)
        if base in base_to_rep:
            mapping[bus] = base_to_rep[base] + suffix
        else:
            mapping[bus] = bus

    return pd.Series(mapping, name="busmap").reindex(n.buses.index.astype(str))


def apply_temporal_reduction(
    n,
    *,
    rep_days: np.ndarray,
    rep_weights: np.ndarray,
    hours_per_day: int = 24,
) -> None:
    """
    In-place temporal reduction:
    - Keep only snapshots corresponding to representative days.
    - Multiply snapshot_weightings by the representative day weight (cardinality).
    """
    rep_days = np.asarray(rep_days, dtype=int)
    rep_weights = np.asarray(rep_weights, dtype=float)

    snaps = n.snapshots
    T = len(snaps)
    if T % hours_per_day != 0:
        raise ValueError(f"Snapshots length {T} not divisible by hours_per_day={hours_per_day}.")
    D = T // hours_per_day

    if rep_days.min() < 0 or rep_days.max() >= D:
        raise ValueError("rep_days out of range.")

    rep_days_sorted = np.sort(rep_days)
    keep_idx = np.concatenate(
        [np.arange(d * hours_per_day, (d + 1) * hours_per_day) for d in rep_days_sorted]
    )
    keep_snaps = snaps[keep_idx]

    day_weight_map = {int(d): float(w) for d, w in zip(rep_days, rep_weights)}
    multipliers = []
    for d in rep_days_sorted:
        multipliers.extend([day_weight_map[int(d)]] * hours_per_day)
    mult = pd.Series(np.asarray(multipliers, dtype=float), index=keep_snaps)

    if getattr(n, "snapshot_weightings", None) is not None and not n.snapshot_weightings.empty:
        base_w = n.snapshot_weightings.reindex(keep_snaps).copy()
    else:
        base_w = pd.DataFrame(index=keep_snaps, data={"objective": 1.0})

    if hasattr(n, "set_snapshots"):
        n.set_snapshots(keep_snaps)
    else:
        n.snapshots = keep_snaps

    n.snapshot_weightings = base_w.reindex(n.snapshots).copy()
    for col in n.snapshot_weightings.columns:
        n.snapshot_weightings[col] = (
            n.snapshot_weightings[col].astype(float)
            * mult.reindex(n.snapshots).to_numpy()
        )