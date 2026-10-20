"""Plot the spatial and temporal mappings produced by geo-temporal clustering.
python \
    scripts/geo_temporal_clustering/plot_merged_regions.py \
    resources/IT_400/gtf-n20-d20/geotemporal_clustering/base_s_adm_elec_Gt \
    resources/IT_400/gtf-n20-d20/regions_onshore_base_s_adm.geojson
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def categorical_colors(n: int) -> list:
    """Return deterministic categorical colors without repeating them."""
    if n <= 20:
        return list(plt.get_cmap("tab20").colors[:n])
    return [plt.get_cmap("turbo")(x) for x in np.linspace(0.02, 0.98, n)]


def read_regions(path: str | Path) -> gpd.GeoDataFrame:
    """Read and deduplicate the original onshore regions."""
    regions = gpd.read_file(path)
    if "name" not in regions:
        raise KeyError(f"{path} has no 'name' column")
    return regions.dissolve(by="name")


def plot_regions(
    nodes_path: str | Path,
    representatives_path: str | Path,
    regions_path: str | Path,
    output_path: str | Path,
) -> None:
    """Color regions in the same merged cluster alike and label its representative."""
    nodes = pd.read_csv(nodes_path)
    representatives = pd.read_csv(representatives_path)
    regions = read_regions(regions_path)
    regions["node_cluster"] = regions.index.map(
        nodes.set_index("bus")["node_cluster"]
    )

    clusters = np.sort(nodes["node_cluster"].unique())
    colors = categorical_colors(len(clusters))
    color_for = dict(zip(clusters, colors))
    matched = regions.dropna(subset=["node_cluster"])
    unmatched = regions[regions["node_cluster"].isna()]

    fig, ax = plt.subplots(figsize=(10, 10))
    if not unmatched.empty:
        unmatched.plot(ax=ax, facecolor="0.92", edgecolor="white", linewidth=0.25)
    for cluster in clusters:
        matched[matched["node_cluster"] == cluster].plot(
            ax=ax, facecolor=color_for[cluster], edgecolor="white", linewidth=0.45
        )

    ax.scatter(
        representatives["rep_lon"], representatives["rep_lat"], s=42,
        c=[color_for[c] for c in representatives["rep_node_cluster"]],
        edgecolor="black", linewidth=1.0, zorder=3,
    )
    for row in representatives.itertuples(index=False):
        ax.annotate(
            f"C{int(row.rep_node_cluster)}\n{row.rep_bus}",
            (row.rep_lon, row.rep_lat), xytext=(4, 4), textcoords="offset points",
            fontsize=6, fontweight="bold", zorder=4,
        )

    handles = [Patch(facecolor=color_for[c], label=f"C{int(c)}") for c in clusters]
    handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                          markeredgecolor="black", label="representative region"))
    ax.legend(handles=handles, title="Merged cluster", loc="lower left", fontsize=7,
              title_fontsize=8, ncol=max(1, int(np.ceil(len(handles) / 11))))
    ax.set_title("Geo-temporal spatial merging\nRepresentative regions are outlined and labelled")
    ax.set_axis_off()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_days(
    days_path: str | Path,
    representatives_path: str | Path,
    output_path: str | Path,
) -> None:
    """Draw one colored cell per original day, in chronological reading order."""
    days = pd.read_csv(days_path).sort_values("day_index").reset_index(drop=True)
    representatives = pd.read_csv(representatives_path).sort_values("rep_day_cluster")
    clusters = np.sort(days["day_cluster"].unique())
    colors = categorical_colors(len(clusters))
    color_index = {cluster: i for i, cluster in enumerate(clusters)}

    n_days = len(days)
    ncols = min(37, n_days)
    nrows = int(np.ceil(n_days / ncols))
    grid = np.full((nrows, ncols), np.nan)
    day_numbers = np.full((nrows, ncols), -1, dtype=int)
    for position, row in enumerate(days.itertuples(index=False)):
        y, x = divmod(position, ncols)
        grid[y, x] = color_index[row.day_cluster]
        day_numbers[y, x] = int(row.day_index)

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(clusters) + 1) - 0.5, cmap.N)
    fig, ax = plt.subplots(figsize=(15, max(5, 0.55 * nrows + 2.5)))
    ax.pcolormesh(np.ma.masked_invalid(grid), cmap=cmap, norm=norm,
                  edgecolors="white", linewidth=0.45, shading="flat")

    representative_days = set(representatives["rep_day_index"].astype(int))
    for y, x in np.ndindex(day_numbers.shape):
        day = day_numbers[y, x]
        if day < 0:
            continue
        ax.text(x + 0.5, y + 0.55, str(day + 1), ha="center", va="center", fontsize=5)
        if day in representative_days:
            ax.scatter(x + 0.5, y + 0.2, marker="*", s=36, facecolor="white",
                       edgecolor="black", linewidth=0.5, zorder=3)

    ax.set_xlim(0, ncols)
    ax.set_ylim(nrows, 0)
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels(np.arange(1, ncols + 1), fontsize=6)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels(
        [f"days {r * ncols + 1}–{min((r + 1) * ncols, n_days)}" for r in range(nrows)],
        fontsize=7,
    )
    ax.set_xlabel("Position within chronological row")
    ax.set_title("Temporal merging in chronological order\n"
                 "Cell labels are original days (1-based); ★ is the representative")

    rep_for = representatives.set_index("rep_day_cluster")["rep_day_index"].to_dict()
    handles = [Patch(facecolor=colors[i],
                     label=f"C{int(c)} → day {int(rep_for[c]) + 1}")
               for i, c in enumerate(clusters)]
    ax.legend(handles=handles, title="Cluster → representative", loc="upper left",
              bbox_to_anchor=(1.01, 1), fontsize=7, title_fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Directory containing clustering CSV files")
    parser.add_argument("regions", type=Path, help="Pre-clustering onshore regions GeoJSON")
    parser.add_argument("--output-dir", type=Path, help="Defaults to RESULTS")
    args = parser.parse_args()
    output_dir = args.output_dir or args.results
    plot_regions(args.results / "nodes_assignment.csv",
                 args.results / "representative_nodes.csv", args.regions,
                 output_dir / "merged_regions.png")
    plot_days(args.results / "days_assignment.csv",
              args.results / "representative_days.csv",
              output_dir / "merged_days.png")


if __name__ == "__main__":
    main()
