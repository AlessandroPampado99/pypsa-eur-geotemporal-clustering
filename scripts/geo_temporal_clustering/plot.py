# -*- coding: utf-8 -*-
"""
Snakemake entrypoint: diagnostic plots for geo-temporal clustering.

This script ONLY reads mapping CSVs and creates plots. It does NOT modify networks.

All comments are in English by request.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional plotting deps
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
except Exception as e:
    ccrs = None
    cfeature = None
    shpreader = None

try:
    from shapely.geometry import MultiPoint
except Exception:
    MultiPoint = None

# Ensure repo root on sys.path (PyPSA-Eur scripts style)
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._helpers import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    if "snakemake" not in globals():
        raise RuntimeError("This script is intended to be run by Snakemake.")

    configure_logging(snakemake)

    def _smk_path(namedlist, key: str, idx: int = 0) -> str:
        """Get a path from snakemake input/output for both named and positional declarations."""
        try:
            return str(namedlist[key])
        except Exception:
            return str(namedlist[idx])

    nodes_fn = _smk_path(snakemake.input, "nodes_assignment", 0)
    rep_nodes_fn = _smk_path(snakemake.input, "representative_nodes", 1)

    out_png = Path(_smk_path(snakemake.output, "nodes_map", 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(nodes_fn)
    df_rep = pd.read_csv(rep_nodes_fn)

    # Cluster ids
    clusters = np.sort(df["node_cluster"].unique())
    K = len(clusters)

    # Choose a categorical colormap (matplotlib default cycles)
    cmap = plt.get_cmap("tab20" if K <= 20 else "tab20b")

    # Build colors per point
    color_idx = df["node_cluster"].map({c: i % cmap.N for i, c in enumerate(clusters)}).to_numpy()
    colors = cmap(color_idx)

    if ccrs is None:
        # Fallback: plain scatter without basemap
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(df["lon"], df["lat"], s=60, c=colors, alpha=0.22, linewidths=0)
        ax.scatter(
            df_rep["rep_lon"], df_rep["rep_lat"],
            s=35, c=cmap(df_rep["rep_node_cluster"].map({c: i % cmap.N for i, c in enumerate(clusters)})),
            edgecolors="k", linewidths=0.8, alpha=1.0
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title("Geo-temporal clustering: base buses (halo) + representative buses")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    # Cartopy map (Italy-focused default)
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Map extent inferred from points (with padding)
    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    pad_lon = max(0.5, 0.06 * (lon_max - lon_min))
    pad_lat = max(0.5, 0.06 * (lat_max - lat_min))
    ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, alpha=0.25)
    ax.add_feature(cfeature.LAND, alpha=0.18)
    ax.coastlines(resolution="10m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.6)

    # Italy fill (best-effort; safe if not found)
    try:
        shp = shpreader.natural_earth(resolution="10m", category="cultural", name="admin_0_countries")
        for rec in shpreader.Reader(shp).records():
            name = rec.attributes.get("ADMIN") or rec.attributes.get("NAME_LONG") or ""
            if name == "Italy":
                ax.add_geometries([rec.geometry], crs=ccrs.PlateCarree(), alpha=0.25)
                break
    except Exception:
        pass

    # Plot convex hull per cluster (optional)
    if MultiPoint is not None:
        for c in clusters:
            pts = df.loc[df["node_cluster"] == c, ["lon", "lat"]].to_numpy()
            if pts.shape[0] < 3:
                continue
            hull = MultiPoint(pts).convex_hull
            if hull.geom_type == "Polygon":
                x, y = hull.exterior.xy
                col = cmap((np.where(clusters == c)[0][0]) % cmap.N)
                ax.plot(x, y, transform=ccrs.PlateCarree(), linewidth=1.0, alpha=0.35, color=col)
            elif hull.geom_type == "LineString":
                x, y = hull.xy
                col = cmap((np.where(clusters == c)[0][0]) % cmap.N)
                ax.plot(x, y, transform=ccrs.PlateCarree(), linewidth=1.0, alpha=0.35, color=col)

    # Halo layer: base nodes (large, transparent)
    ax.scatter(
        df["lon"], df["lat"],
        s=80, c=colors, alpha=0.20, linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    # Representative nodes (small, opaque, black edge)
    rep_color_idx = df_rep["rep_node_cluster"].map({c: i % cmap.N for i, c in enumerate(clusters)}).to_numpy()
    ax.scatter(
        df_rep["rep_lon"], df_rep["rep_lat"],
        s=28, c=cmap(rep_color_idx), alpha=1.0,
        edgecolors="k", linewidths=0.7,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    ax.set_title("Geo-temporal clustering: base buses (halo) + representative buses (medoids)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
