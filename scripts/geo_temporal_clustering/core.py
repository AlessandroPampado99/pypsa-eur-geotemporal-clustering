# -*- coding: utf-8 -*-
"""
Geo-temporal clustering core utilities for PyPSA-Eur integration.

This module is intentionally self-contained and uses only public PyPSA APIs.

All comments are in English by request.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import geopandas as gpd

from sklearn.cluster import AgglomerativeClustering
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
    rep_days: np.ndarray
    rep_weights: np.ndarray
    history: List[dict]
    day_pca_info: dict
    region_membership: Optional[pd.Series] = None


class AlternatingSpatioTemporalReducer:
    """
    Alternating reducer implementing:
    - Iter 0: node clustering uses ALL days with uniform weights.
    - Iter >=1: node clustering uses ONLY representative days (medoids) with weights.
    - Day clustering ALWAYS uses ALL days, but on spatially aggregated representation defined by current node clusters.
    - Optional PCA in the day clustering distance computation.
    - Optional spatial adjacency constraint on node merges.
    """

    def __init__(
        self,
        *,
        lambda_ts: float = 0.85,
        normalize: Literal["zscore", "minmax"] = "zscore",
        node_threshold: float = 0.25,
        day_threshold: float = 0.30,
        linkage: Literal["average", "complete"] = "complete",
        max_iter: int = 10,
        tol_no_change: int = 2,
        verbose: bool = True,
        norm_q: float = 0.95,
        use_pca_days: bool = False,
        pca_days_n_components: Union[int, float] = 0.95,
        pca_days_random_state: int = 0,
        standardize_day_matrix_cols: bool = False,
        enforce_spatial_adjacency: bool = False,
        regions_gdf: Optional[gpd.GeoDataFrame] = None,
        region_name_col: str = "name",
    ):
        self.lambda_ts = float(lambda_ts)
        self.normalize = normalize
        self.node_threshold = float(node_threshold)
        self.day_threshold = float(day_threshold)
        self.linkage = linkage
        self.max_iter = int(max_iter)
        self.tol_no_change = int(tol_no_change)
        self.verbose = bool(verbose)
        self.norm_q = float(norm_q)

        self.use_pca_days = bool(use_pca_days)
        self.pca_days_n_components = pca_days_n_components
        self.pca_days_random_state = int(pca_days_random_state)
        self.standardize_day_matrix_cols = bool(standardize_day_matrix_cols)

        self.enforce_spatial_adjacency = bool(enforce_spatial_adjacency)
        self.regions_gdf = regions_gdf
        self.region_name_col = str(region_name_col)

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
        N, D, _ = X.shape

        lat_deg = np.asarray(lat_deg, dtype=float)
        lon_deg = np.asarray(lon_deg, dtype=float)
        if lat_deg.shape != (N,) or lon_deg.shape != (N,):
            raise ValueError("lat_deg and lon_deg must have shape (N,).")

        if buses is None:
            raise ValueError("buses must be provided when fitting the spatial reducer.")
        if len(buses) != N:
            raise ValueError("buses must have length N.")
        buses = list(map(str, buses))

        # Feature-wise normalization
        if self.normalize == "zscore":
            Xn = zscore_global(X)
        elif self.normalize == "minmax":
            Xn = minmax_global(X)
        else:
            raise ValueError("normalize must be 'zscore' or 'minmax'.")

        D_geo = haversine_pairwise_km(lat_deg, lon_deg)
        D_geo_n = normalize_distance_matrix(D_geo, q=self.norm_q)

        node_connectivity = None
        region_membership = None

        if self.enforce_spatial_adjacency:
            if self.regions_gdf is None:
                raise ValueError(
                    "regions_gdf must be provided when enforce_spatial_adjacency=True."
                )

            node_connectivity, region_membership = build_bus_connectivity_from_regions(
                buses=buses,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                regions_gdf=self.regions_gdf,
                region_name_col=self.region_name_col,
            )

            if self.verbose:
                n_allowed_pairs = int(node_connectivity.nnz)
                print(
                    f"[Adjacency] Spatial connectivity enabled | "
                    f"nodes={N} | allowed_pairs={n_allowed_pairs}"
                )

        history: List[dict] = []
        stable_counter = 0

        labels_nodes_prev: Optional[np.ndarray] = None
        labels_days_prev: Optional[np.ndarray] = None

        rep_days = np.arange(D, dtype=int)
        rep_weights = np.ones(D, dtype=int)

        last_day_pca_info: dict = {}

        for it in range(self.max_iter):
            # 1) Node clustering
            if it == 0:
                X_for_nodes = Xn
                w_for_nodes = np.ones(D, dtype=float)
                used_days = D
            else:
                X_for_nodes = Xn[:, rep_days, :]
                w_for_nodes = rep_weights.astype(float)
                used_days = len(rep_days)

            D_ts_raw = build_node_ts_distance(X_for_nodes, w_for_nodes)
            D_ts_n = normalize_distance_matrix(D_ts_raw, q=self.norm_q)

            D_node = self.lambda_ts * D_ts_n + (1.0 - self.lambda_ts) * D_geo_n

            labels_nodes = _agglomerative_precomputed(
                D_node,
                distance_threshold=self.node_threshold,
                linkage=self.linkage,
                connectivity=node_connectivity,
            )
            K_nodes = int(len(np.unique(labels_nodes)))

            # 2) Day clustering (always on ALL days)
            X_agg, cluster_sizes = aggregate_nodes_by_cluster(
                Xn,
                labels_nodes,
                node_weights=node_weights,
            )

            D_day_raw, day_pca_info = build_day_distance(
                X_agg,
                cluster_sizes=cluster_sizes,
                use_pca=self.use_pca_days,
                pca_n_components=self.pca_days_n_components,
                pca_random_state=self.pca_days_random_state,
                standardize_day_matrix_cols=self.standardize_day_matrix_cols,
            )
            last_day_pca_info = day_pca_info

            D_day_n = normalize_distance_matrix(D_day_raw, q=self.norm_q)

            labels_days = _agglomerative_precomputed(
                D_day_n,
                distance_threshold=self.day_threshold,
                linkage=self.linkage,
            )
            K_days = int(len(np.unique(labels_days)))

            rep_days, rep_weights = representative_medoids(D_day_raw, labels_days)

            history.append(
                dict(
                    iter=it,
                    used_days_for_node_distance=int(used_days),
                    K_nodes=K_nodes,
                    K_days=K_days,
                    n_rep_days=int(len(rep_days)),
                    day_pca=day_pca_info,
                    adjacency_enabled=bool(self.enforce_spatial_adjacency),
                )
            )

            if self.verbose:
                if day_pca_info.get("use_pca", False):
                    r = day_pca_info["pca_n_components_selected"]
                    ev = day_pca_info["pca_explained_variance_ratio_sum"]
                    pca_str = f"PCA(r={r}, EV={ev:.3f})"
                else:
                    pca_str = "no PCA"

                print(
                    f"[Iter {it}] used_days_for_nodes={used_days} | "
                    f"K_nodes={K_nodes} | K_days={K_days} | "
                    f"rep_days={len(rep_days)} | {pca_str}"
                )

            same_nodes = (
                labels_nodes_prev is not None
                and np.array_equal(labels_nodes, labels_nodes_prev)
            )
            same_days = (
                labels_days_prev is not None
                and np.array_equal(labels_days, labels_days_prev)
            )

            stable_counter = stable_counter + 1 if (same_nodes and same_days) else 0

            labels_nodes_prev = labels_nodes.copy()
            labels_days_prev = labels_days.copy()

            if stable_counter >= self.tol_no_change:
                if self.verbose:
                    print(f"[Converged] Stable for {stable_counter} consecutive iterations.")
                break

        return ReductionResult(
            labels_nodes=labels_nodes,
            labels_days=labels_days,
            rep_days=rep_days.astype(int),
            rep_weights=rep_weights.astype(int),
            history=history,
            day_pca_info=last_day_pca_info,
            region_membership=region_membership,
        )


# =============================================================================
# PyPSA helpers (feature extraction + mapping)
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