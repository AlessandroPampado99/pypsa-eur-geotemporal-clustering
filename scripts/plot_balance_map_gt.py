# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Create static energy balance maps for geo-temporal clustered networks.

This GT-specific version:
- colors the original pre-clustering regions by geo-temporal node cluster,
- keeps pie charts and branch flows on the clustered representative buses,
- does not use marginal prices for the regional background.

All comments are in English by request.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from matplotlib.colors import ListedColormap
from packaging.version import Version, parse
from pypsa.plot import add_legend_lines, add_legend_patches, add_legend_semicircles
from pypsa.statistics import get_transmission_carriers

from scripts._helpers import (
    PYPSA_V1,
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.add_electricity import sanitize_carriers
from scripts.plot_power_network import load_projection

SEMICIRCLE_CORRECTION_FACTOR = 2 if parse(pypsa.__version__) <= Version("0.33.2") else 1


def _read_regions_unique(path: str) -> gpd.GeoDataFrame:
    """Read regions GeoJSON and ensure unique index by dissolving on 'name'."""
    regions = gpd.read_file(path)
    if "name" not in regions.columns:
        raise KeyError("Expected column 'name' in regions GeoJSON.")
    regions = regions.dissolve(by="name")
    if regions.index.has_duplicates:
        regions = regions[~regions.index.duplicated(keep="first")]
    return regions


def _read_busmap(path: str) -> pd.Series:
    """
    Read geo-temporal clustering busmap as a Series:
    original_bus -> clustered_bus
    """
    busmap = pd.read_csv(path, index_col=0).squeeze("columns")
    if not isinstance(busmap, pd.Series):
        raise TypeError("Expected busmap.csv to load as a pandas Series.")
    busmap.index = busmap.index.astype(str)
    busmap = busmap.astype(str)
    return busmap


def _prepare_plot_locations(n: pypsa.Network) -> pd.Series:
    """
    Prepare a robust location label for each bus.
    If location is missing/empty, use the bus name itself to avoid collapsing.
    """
    loc = n.buses["location"].replace("", pd.NA)
    loc = loc.fillna(pd.Series(n.buses.index, index=n.buses.index))
    return loc


def _remap_bus_coordinates_to_location_bus(n: pypsa.Network, plot_loc: pd.Series) -> None:
    """
    Remap bus coordinates to the coordinates of the bus referenced by plot_loc,
    but only if plot_loc matches an existing bus name. Otherwise keep original coords.

    This avoids NaNs and prevents buses from collapsing to a single 'EU' point.
    """
    x0 = n.buses["x"].copy()
    y0 = n.buses["y"].copy()

    x_by_bus = x0.copy()
    y_by_bus = y0.copy()

    x_new = plot_loc.map(x_by_bus)
    y_new = plot_loc.map(y_by_bus)

    n.buses["x"] = x_new.fillna(x0)
    n.buses["y"] = y_new.fillna(y0)


def _read_nodes_assignment(path: str) -> pd.DataFrame:
    """Read nodes assignment CSV created by geo-temporal clustering."""
    df = pd.read_csv(path)
    required = {"bus", "node_cluster"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"nodes_assignment.csv is missing required columns: {missing}")
    df["bus"] = df["bus"].astype(str)
    return df


def _build_cluster_cmap(cluster_values: pd.Series):
    """Build a categorical colormap for integer cluster ids."""
    unique_clusters = sorted(pd.Series(cluster_values).dropna().astype(int).unique())
    if len(unique_clusters) <= 20:
        base = plt.get_cmap("tab20")
    else:
        base = plt.get_cmap("tab20b")

    color_list = [base(i % base.N) for i in range(len(unique_clusters))]
    cmap = ListedColormap(color_list)
    cluster_to_code = {cluster: i for i, cluster in enumerate(unique_clusters)}
    return unique_clusters, cluster_to_code, cmap


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_balance_map_elec_only_gt",
            clusters="adm",
            opts="Gt",
            sector_opts="eleconly",
            planning_horizons="2050",
            carrier="AC",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    n = pypsa.Network(snakemake.input.network)
    sanitize_carriers(n, snakemake.config)
    pypsa.set_option("params.statistics.round", 8)
    pypsa.set_option("params.statistics.drop_zero", True)
    pypsa.set_option("params.statistics.nice_names", False)

    regions = _read_regions_unique(snakemake.input.regions)
    busmap = _read_busmap(snakemake.input.busmap)
    nodes_assignment = _read_nodes_assignment(snakemake.input.nodes_assignment)

    config = snakemake.params.plotting
    carrier = snakemake.wildcards.carrier.replace("_", " ")
    settings = snakemake.params.settings

    if settings is None:
        raise ValueError(
            f"No plotting.balance_map settings found for carrier '{carrier}'."
        )

    # Fill empty colors with light grey
    mask = n.carriers.color.isna() | n.carriers.color.eq("")
    n.carriers["color"] = n.carriers.color.mask(mask, "lightgrey")

    # Set EU bus coordinates from config if present
    eu_location = config["eu_node_location"]
    if "EU" in n.buses.index:
        n.buses.loc["EU", ["x", "y"]] = eu_location["x"], eu_location["y"]

    boundaries = config["map"]["boundaries"]
    unit_conversion = settings["unit_conversion"]
    branch_color = settings.get("branch_color") or "darkseagreen"

    if carrier not in n.buses.carrier.unique():
        raise ValueError(
            f"Carrier {carrier} is not in the network. Remove from configuration "
            "`plotting: balance_map: bus_carriers`."
        )

    # Keep standard plotting logic for representative buses
    plot_loc = _prepare_plot_locations(n)
    n.buses["location"] = plot_loc
    _remap_bus_coordinates_to_location_bus(n, plot_loc)

    # Bus sizes from energy balance of bus carrier
    eb = n.statistics.energy_balance(bus_carrier=carrier, groupby=["bus", "carrier"])

    # Remove energy balance of transmission carriers which relate to losses
    transmission_carriers = get_transmission_carriers(n, bus_carrier=carrier).rename(
        {"name": "carrier"}
    )
    components = transmission_carriers.unique("component")
    carriers = transmission_carriers.unique("carrier")

    carriers_in_eb = carriers[carriers.isin(eb.index.get_level_values("carrier"))]
    eb.loc[components] = eb.loc[components].drop(index=carriers_in_eb, level="carrier")
    eb = eb.dropna()

    bus_sizes = eb.groupby(level=["bus", "carrier"]).sum().div(unit_conversion)
    bus_sizes = bus_sizes.sort_values(ascending=False)

    # Carrier colors for pie charts
    n.carriers.update({"color": snakemake.params.plotting["tech_colors"]})
    carrier_colors = n.carriers.color.copy().replace("", "grey")
    colors = (
        bus_sizes.index.get_level_values("carrier")
        .unique()
        .to_series()
        .map(carrier_colors)
    )

    # Branch widths from transmission statistics
    flow = n.statistics.transmission(groupby=False, bus_carrier=carrier).div(
        unit_conversion
    )

    if not flow.empty:
        flow_reversed_mask = flow.index.get_level_values(1).str.contains("reversed")
        flow_reversed = flow[flow_reversed_mask].rename(
            lambda x: x.replace("-reversed", "")
        )
        flow = flow[~flow_reversed_mask].subtract(flow_reversed, fill_value=0)

    fallback = pd.Series(dtype=float)
    line_widths = flow.get("Line", fallback).abs()
    link_widths = flow.get("Link", fallback).abs()

    bus_size_factor = settings["bus_factor"]
    branch_width_factor = settings["branch_factor"]
    flow_size_factor = settings["flow_factor"]

    # ------------------------------------------------------------------
    # Expand node cluster ids from clustered buses back to original regions
    # ------------------------------------------------------------------
    cluster_by_clustered_bus = nodes_assignment.drop_duplicates("bus").set_index("bus")[
        "node_cluster"
    ]

    regions["clustered_bus"] = regions.index.to_series().map(busmap)
    regions["node_cluster"] = regions["clustered_bus"].map(cluster_by_clustered_bus)

    matched = regions["node_cluster"].notna().sum()
    total = len(regions)
    print(f"Matched regions with node_cluster: {matched}/{total}")

    unique_clusters, cluster_to_code, cluster_cmap = _build_cluster_cmap(
        regions["node_cluster"]
    )

    regions["cluster_code"] = regions["node_cluster"].map(cluster_to_code)

    crs = load_projection(snakemake.params.plotting)

    fig, ax = plt.subplots(
        figsize=(5, 6.5),
        subplot_kw={"projection": crs},
        layout="constrained",
    )

    # ------------------------------------------------------------------
    # Plot regional background first
    # ------------------------------------------------------------------
    matched_regions = regions[regions["cluster_code"].notna()].copy()
    unmatched_regions = regions[regions["cluster_code"].isna()].copy()

    if not unmatched_regions.empty:
        unmatched_regions.to_crs(crs.proj4_init).plot(
            ax=ax,
            facecolor="whitesmoke",
            edgecolor="lightgrey",
            linewidth=0.2,
            zorder=0,
        )

    if not matched_regions.empty:
        matched_regions.to_crs(crs.proj4_init).plot(
            ax=ax,
            column="cluster_code",
            cmap=cluster_cmap,
            edgecolor="white",
            linewidth=0.25,
            zorder=1,
        )

    # ------------------------------------------------------------------
    # Plot pies and flows on representative clustered buses
    # ------------------------------------------------------------------
    line_flow = flow.get("Line")
    link_flow = flow.get("Link")
    transformer_flow = flow.get("Transformer")

    n.plot(
        bus_sizes=bus_sizes * bus_size_factor,
        bus_colors=colors,
        bus_split_circles=True,
        line_widths=line_widths * branch_width_factor,
        link_widths=link_widths * branch_width_factor,
        line_flow=line_flow * flow_size_factor if line_flow is not None else None,
        link_flow=link_flow * flow_size_factor if link_flow is not None else None,
        link_color=branch_color,
        transformer_flow=transformer_flow * flow_size_factor
        if transformer_flow is not None
        else None,
        ax=ax,
        margin=0.2,
        geomap_colors={"border": "darkgrey", "coastline": "darkgrey"},
        geomap=True,
        boundaries=boundaries,
    )

    ax.set_title(f"{carrier} balance map on geo-temporal clusters")

    legend_kwargs = {
        "loc": "upper left",
        "frameon": False,
        "alignment": "left",
        "title_fontproperties": {"weight": "bold"},
    }

    pad = 0.18
    n.carriers.loc["", "color"] = "None"

    pos_carriers = bus_sizes[bus_sizes > 0].index.unique("carrier")
    neg_carriers = bus_sizes[bus_sizes < 0].index.unique("carrier")
    common_carriers = pos_carriers.intersection(neg_carriers)

    def get_total_abs(carr, sign):
        values = bus_sizes.loc[:, carr]
        return values[values * sign > 0].abs().sum()

    supp_carriers = sorted(
        set(pos_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) >= get_total_abs(c, -1)}
    )
    cons_carriers = sorted(
        set(neg_carriers) - set(common_carriers)
        | {c for c in common_carriers if get_total_abs(c, 1) < get_total_abs(c, -1)}
    )

    add_legend_patches(
        ax,
        n.carriers.color[supp_carriers],
        supp_carriers,
        legend_kw={
            "bbox_to_anchor": (0, -pad),
            "ncol": 1,
            "title": "Supply",
            **legend_kwargs,
        },
    )

    add_legend_patches(
        ax,
        n.carriers.color[cons_carriers],
        cons_carriers,
        legend_kw={
            "bbox_to_anchor": (0.5, -pad),
            "ncol": 1,
            "title": "Consumption",
            **legend_kwargs,
        },
    )

    legend_bus_sizes = settings["bus_sizes"]
    unit = settings["unit"]
    if legend_bus_sizes is not None:
        add_legend_semicircles(
            ax,
            [
                s * bus_size_factor * SEMICIRCLE_CORRECTION_FACTOR
                for s in legend_bus_sizes
            ],
            [f"{s} {unit}" for s in legend_bus_sizes],
            patch_kw={"color": "#666"},
            legend_kw={"bbox_to_anchor": (0, 1), **legend_kwargs},
        )

    legend_branch_sizes = settings["branch_sizes"]
    if legend_branch_sizes is not None:
        add_legend_lines(
            ax,
            [s * branch_width_factor for s in legend_branch_sizes],
            [f"{s} {unit}" for s in legend_branch_sizes],
            patch_kw={"color": "#666"},
            legend_kw={"bbox_to_anchor": (0.25, 1), **legend_kwargs},
        )

    fig.savefig(
        snakemake.output[0],
        dpi=400,
        bbox_inches="tight",
    )