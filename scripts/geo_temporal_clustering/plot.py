# -*- coding: utf-8 -*-
"""
Snakemake entrypoint: diagnostic plots for geo-temporal clustering.

This script ONLY reads mapping CSVs and creates plots. It does NOT modify networks.

Main behavior:
- Read the pre-clustering onshore regions geometry used by the base network.
- Color regions by assigned geo-temporal cluster.
- Overlay representative nodes as black-edged points.

All comments are in English by request.
"""

import logging
import sys
from pathlib import Path
from pathlib import Path as _Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repo root on sys.path (PyPSA-Eur scripts style)
ROOT = _Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def _smk_path(namedlist, key: str, idx: int = 0) -> str:
    """Get a path from snakemake input/output for both named and positional declarations."""
    try:
        return str(namedlist[key])
    except Exception:
        return str(namedlist[idx])


def _build_cluster_colormap(clusters: np.ndarray):
    """Build a stable categorical colormap and mapping."""
    k = len(clusters)
    cmap = plt.get_cmap("tab20" if k <= 20 else "tab20b")
    cluster_to_idx = {c: i % cmap.N for i, c in enumerate(clusters)}
    return cmap, cluster_to_idx


def _read_regions_unique(path: str) -> gpd.GeoDataFrame:
    """Read regions GeoJSON and ensure unique index by dissolving on 'name'."""
    regions = gpd.read_file(path)
    if "name" not in regions.columns:
        raise KeyError("Expected column 'name' in regions GeoJSON.")
    regions = regions.dissolve(by="name")
    if regions.index.has_duplicates:
        regions = regions[~regions.index.duplicated(keep="first")]
    return regions


def main() -> None:
    """Main plotting routine."""
    configure_logging(snakemake)

    nodes_fn = _smk_path(snakemake.input, "nodes_assignment", 0)
    rep_nodes_fn = _smk_path(snakemake.input, "representative_nodes", 1)
    regions_fn = _smk_path(snakemake.input, "regions", 2)

    out_png = Path(_smk_path(snakemake.output, "nodes_map", 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(nodes_fn)
    df_rep = pd.read_csv(rep_nodes_fn)
    regions = _read_regions_unique(regions_fn)

    logger.info("Using regions file: %s", regions_fn)
    logger.info("Regions columns after read: %s", list(regions.columns))
    logger.info("Regions count after dissolve: %s", len(regions))

    clusters = np.sort(df["node_cluster"].unique())
    cmap, cluster_to_idx = _build_cluster_colormap(clusters)

    # Map node cluster to regions using base bus name
    cluster_map = df.set_index("bus")["node_cluster"].to_dict()
    regions["node_cluster"] = regions.index.to_series().map(cluster_map)

    matched = regions["node_cluster"].notna().sum()
    logger.info("Matched regions to node clusters: %s / %s", matched, len(regions))

    # ------------------------------------------------------------------
    # Cartopy map
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(8.0, 8.0))
    ax = plt.axes(projection=ccrs.PlateCarree())

    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    pad_lon = max(0.5, 0.06 * (lon_max - lon_min))
    pad_lat = max(0.5, 0.06 * (lat_max - lat_min))
    ax.set_extent(
        [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.OCEAN, alpha=0.25)
    ax.add_feature(cfeature.LAND, alpha=0.15)
    ax.coastlines(resolution="10m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.6)

    shp = shpreader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_0_countries",
    )
    for rec in shpreader.Reader(shp).records():
        name = rec.attributes.get("ADMIN") or rec.attributes.get("NAME_LONG") or ""
        if name == "Italy":
            ax.add_geometries(
                [rec.geometry],
                crs=ccrs.PlateCarree(),
                alpha=0.10,
            )
            break

    # ------------------------------------------------------------------
    # Plot regions by cluster
    # ------------------------------------------------------------------
    matched_regions = regions[regions["node_cluster"].notna()].copy()
    unmatched_regions = regions[regions["node_cluster"].isna()].copy()

    for c in clusters:
        gdf_c = matched_regions.loc[matched_regions["node_cluster"] == c]
        if gdf_c.empty:
            continue

        col = cmap(cluster_to_idx[c])
        ax.add_geometries(
            gdf_c.geometry,
            crs=ccrs.PlateCarree(),
            facecolor=col,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.70,
            zorder=2,
        )

    if not unmatched_regions.empty:
        ax.add_geometries(
            unmatched_regions.geometry,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="lightgrey",
            linewidth=0.25,
            alpha=0.5,
            zorder=1,
        )

    # Representative nodes on top
    rep_color_idx = df_rep["rep_node_cluster"].map(cluster_to_idx).to_numpy()
    ax.scatter(
        df_rep["rep_lon"],
        df_rep["rep_lat"],
        s=30,
        c=cmap(rep_color_idx),
        alpha=1.0,
        edgecolors="k",
        linewidths=0.7,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    ax.set_title("Geo-temporal clustering: pre-clustering regions by cluster + representative buses")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_geo_temporal_cluster_network",
            clusters="adm",
            opts="Gt",
            configfiles=["config/config_clustering.yaml"],
        )

    main()