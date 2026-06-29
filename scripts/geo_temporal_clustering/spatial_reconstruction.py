# -*- coding: utf-8 -*-
"""
Spatial reconstruction utilities for geo-temporal clustering.

This module centralizes the PyPSA spatial reconstruction logic used after
geo-temporal node clustering. It is intentionally separated from the Snakemake
entrypoint to keep the workflow script readable.

Main responsibilities:
- reconstruct a clustered PyPSA network from a full busmap;
- aggregate one-port components by clustered bus and carrier;
- aggregate Links by clustered bus ports and carrier;
- fix clustered bus coordinates for auxiliary suffix buses.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import numpy as np
import pandas as pd
import pypsa

from pypsa.clustering.spatial import get_clustering_from_busmap

logger = logging.getLogger(__name__)


def reconstruct_spatially_clustered_network(
    n: pypsa.Network,
    busmap: pd.Series,
    *,
    aggregate_one_ports: bool = True,
    aggregate_links: bool = True,
    custom_line_groupers: list[str] | None = None,
) -> tuple[pypsa.Network, pd.Series]:
    """
    Reconstruct a spatially clustered network from a full busmap.

    Parameters
    ----------
    n:
        Original PyPSA network before geo-temporal clustering.
    busmap:
        Mapping from every original bus to its clustered representative bus.
        This must include auxiliary buses such as "<bus> H2" and
        "<bus> battery".
    aggregate_one_ports:
        If True, aggregate Generator, Load, Store and StorageUnit components
        by clustered bus and carrier.
    aggregate_links:
        If True, aggregate remapped Links by all bus ports and carrier.
    custom_line_groupers:
        Extra static Line attributes used by PyPSA when grouping lines.

    Returns
    -------
    nc:
        Spatially clustered PyPSA network.
    linemap:
        Mapping from original lines to clustered lines.
    """
    busmap = _clean_busmap(busmap, n)
    _sanitize_line_attributes_before_clustering(n)

    logger.info(
        "Starting GT spatial reconstruction: buses=%d, unique mapped buses=%d.",
        len(busmap),
        busmap.nunique(),
    )

    _log_component_counts(n, prefix="Input network before GT spatial reconstruction")

    if "location" in n.buses.columns:
        n.buses.loc[:, "location"] = busmap.reindex(n.buses.index).astype(str)

    kwargs = _build_get_clustering_kwargs(
        n=n,
        aggregate_one_ports=aggregate_one_ports,
        custom_line_groupers=custom_line_groupers,
    )

    clustering = get_clustering_from_busmap(n, busmap, **kwargs)
    nc = clustering.n

    if aggregate_links:
        _aggregate_links_by_bus_ports_and_carrier(nc)

    _fix_clustered_bus_coordinates(n_original=n, n_clustered=nc)

    _log_component_counts(nc, prefix="Output network after GT spatial reconstruction")

    return nc, clustering.linemap


def _clean_busmap(busmap: pd.Series, n: pypsa.Network) -> pd.Series:
    """
    Validate and align the busmap to the network buses.
    """
    if not isinstance(busmap, pd.Series):
        busmap = pd.Series(busmap)

    busmap = busmap.astype(str)
    missing = n.buses.index.difference(busmap.index)

    if len(missing) > 0:
        raise ValueError(
            "The GT busmap does not cover all network buses. "
            f"Missing examples: {list(missing[:10])}"
        )

    busmap = busmap.reindex(n.buses.index).astype(str)

    if busmap.isna().any():
        missing = busmap.index[busmap.isna()]
        raise ValueError(
            "The GT busmap contains NaN mapped buses. "
            f"Examples: {list(missing[:10])}"
        )

    return busmap

def _sanitize_line_attributes_before_clustering(n: pypsa.Network) -> None:
    """
    Sanitize Line attributes before PyPSA line aggregation.
    """
    if n.lines.empty:
        return

    for col in ["dc", "under_construction", "s_nom_extendable"]:
        if col in n.lines.columns:
            n.lines[col] = n.lines[col].fillna(False).astype(bool)

            
def _build_get_clustering_kwargs(
    *,
    n: pypsa.Network,
    aggregate_one_ports: bool,
    custom_line_groupers: list[str] | None,
) -> dict[str, Any]:
    """
    Build kwargs for PyPSA's get_clustering_from_busmap.

    The function signature changed slightly across PyPSA versions, so unsupported
    keyword arguments are filtered out at runtime.
    """
    line_groupers = custom_line_groupers or []

    kwargs: dict[str, Any] = {
        "bus_strategies": _build_bus_strategies(n),
        "line_strategies": {},
        "custom_line_groupers": line_groupers,
    }

    if aggregate_one_ports:
        kwargs["aggregate_one_ports"] = {
            "Generator",
            "Load",
            "Store",
            "StorageUnit",
        }

        kwargs["generator_strategies"] = {
            "committable": "any",
            "ramp_limit_up": "max",
            "ramp_limit_down": "max",
            "ramp_limit_start_up": "max",
            "ramp_limit_shut_down": "max",
        }

    return _filter_supported_kwargs(get_clustering_from_busmap, kwargs)

def _build_bus_strategies(n: pypsa.Network) -> dict[str, Any]:
    """
    Build custom Bus aggregation strategies.

    PyPSA's default Bus aggregation uses strict consensus for several
    attributes. This is too strict when an extra no-load electric bus is assigned
    to the nearest clustered representative, because boolean-like metadata such
    as substation_lv may differ inside the cluster.

    For these flags, max/any is the intended logic:
    if at least one bus in the aggregate is a low-voltage substation, the
    clustered bus should retain that flag.
    """
    candidate_strategies: dict[str, Any] = {
        "substation_lv": "max",
        "substation_off": "max",
        "substation_dc": "max",
        "substation": "max",
    }

    return {
        col: strategy
        for col, strategy in candidate_strategies.items()
        if col in n.buses.columns
    }

def _filter_supported_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Drop kwargs not supported by the installed PyPSA version.
    """
    params = inspect.signature(func).parameters
    supported = set(params)

    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(set(kwargs) - set(filtered))

    if dropped:
        logger.warning(
            "Dropping unsupported get_clustering_from_busmap kwargs for this "
            "PyPSA version: %s",
            dropped,
        )

    return filtered


def _aggregate_links_by_bus_ports_and_carrier(n: pypsa.Network) -> None:
    """
    Aggregate Links by all bus ports and carrier after busmap clustering.

    Links are directed components, so bus0 -> bus1 is not equivalent to
    bus1 -> bus0. Therefore, the grouping keeps the ordered bus columns.

    Grouping logic:
    - same bus0, bus1, bus2, ...;
    - same carrier, if available.
    """
    if n.links.empty:
        logger.info("No Links found. Skipping Link aggregation.")
        return

    links = n.links.copy()
    old_link_count = len(links)

    bus_cols = _get_link_bus_columns(links)

    if "bus0" not in bus_cols or "bus1" not in bus_cols:
        logger.warning(
            "Cannot aggregate Links because bus0/bus1 are missing. "
            "Keeping Links unchanged."
        )
        return

    # Drop links that became internal after clustering.
    links = links.dropna(subset=["bus0", "bus1"]).copy()
    links = links.loc[links["bus0"].astype(str) != links["bus1"].astype(str)].copy()

    old_links_index = n.links.index.copy()

    if links.empty:
        logger.info("All Links became internal after clustering. Removing them.")
        n.remove("Link", old_links_index)
        return

    group_cols = bus_cols.copy()

    if "carrier" in links.columns:
        group_cols.append("carrier")

    grouped = links.groupby(group_cols, dropna=False, sort=False)

    if len(grouped) == len(links):
        logger.info(
            "No duplicate Links by ordered bus ports and carrier. "
            "Link count remains %d.",
            old_link_count,
        )
        return

    new_rows: list[pd.Series] = []
    group_members: dict[str, list[str]] = {}

    for i, (_, group) in enumerate(grouped):
        group = group.copy()
        new_name = _make_link_group_name(i, group, group_cols)

        row = _aggregate_link_static_group(group, bus_cols=bus_cols)
        row.name = new_name

        new_rows.append(row)
        group_members[new_name] = list(group.index.astype(str))

    new_links = pd.DataFrame(new_rows)
    new_links.index = new_links.index.astype(str)
    new_links.index.name = n.links.index.name

    new_dynamic = _aggregate_link_dynamic_data(
        n=n,
        old_links=links,
        group_members=group_members,
    )

    logger.info(
        "Aggregating Links by ordered bus ports and carrier: %d -> %d.",
        len(old_links_index),
        len(new_links),
    )

    n.remove("Link", old_links_index)
    n.add("Link", new_links.index, **new_links.to_dict(orient="series"))

    for attr, df in new_dynamic.items():
        if not df.empty:
            _import_dynamic_dataframe(n, component="Link", attr=attr, df=df)


def _get_link_bus_columns(links: pd.DataFrame) -> list[str]:
    """
    Return Link bus columns ordered as bus0, bus1, bus2, ...
    """
    bus_cols = [
        col
        for col in links.columns
        if col.startswith("bus") and col[3:].isdigit()
    ]

    return sorted(bus_cols, key=lambda col: int(col[3:]))


def _make_link_group_name(
    i: int,
    group: pd.DataFrame,
    group_cols: list[str],
) -> str:
    """
    Build a stable name for an aggregated Link group.
    """
    if len(group) == 1:
        return str(group.index[0])

    first = group.iloc[0]
    carrier = str(first["carrier"]) if "carrier" in group_cols else "link"

    safe_carrier = (
        carrier.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    return f"GT_Link_{i:05d}_{safe_carrier}"


def _aggregate_link_static_group(
    group: pd.DataFrame,
    *,
    bus_cols: list[str],
) -> pd.Series:
    """
    Aggregate static attributes of a group of equivalent Links.
    """
    row = group.iloc[0].copy()

    capacity_weights = _capacity_weights(group)

    for col in group.columns:
        values = group[col]

        if col in bus_cols or col == "carrier":
            row[col] = _first_valid(values)

        elif col in {"p_nom", "p_nom_min", "p_nom_max", "p_nom_mod"}:
            row[col] = pd.to_numeric(values, errors="coerce").fillna(0.0).sum()

        elif col in {"p_nom_extendable", "active"}:
            row[col] = values.fillna(False).astype(bool).any()

        elif col == "build_year":
            # Aggregating by carrier intentionally collapses vintages.
            row[col] = 0

        elif col in {
            "capital_cost",
            "marginal_cost",
            "length",
            "terrain_factor",
            "underwater_fraction",
            "p_min_pu",
            "p_max_pu",
            "efficiency",
            "efficiency2",
            "efficiency3",
            "efficiency4",
            "lifetime",
        }:
            row[col] = _weighted_average(values, capacity_weights)

        elif pd.api.types.is_numeric_dtype(values):
            row[col] = _weighted_average(values, capacity_weights)

        else:
            row[col] = _first_valid(values)

    return row


def _aggregate_link_dynamic_data(
    *,
    n: pypsa.Network,
    old_links: pd.DataFrame,
    group_members: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """
    Aggregate dynamic Link attributes consistently with static aggregation.
    """
    out: dict[str, pd.DataFrame] = {}

    for attr, data in n.links_t.items():
        if data.empty:
            continue

        data = data.reindex(columns=old_links.index)

        if data.empty:
            continue

        aggregated = pd.DataFrame(index=data.index)

        for new_name, members in group_members.items():
            members = [m for m in members if m in data.columns]

            if not members:
                continue

            part = data[members]

            if not part.notna().any().any():
                continue

            weights = _capacity_weights(old_links.loc[members])

            if attr in {"p_min_pu", "p_max_pu", "marginal_cost"} or attr.startswith("efficiency"):
                aggregated[new_name] = part.mul(weights, axis=1).sum(axis=1)

            elif attr in {"p_set"}:
                aggregated[new_name] = part.sum(axis=1, min_count=1)

            else:
                # Conservative fallback: most time-varying Link quantities are
                # power-like and should be additive after aggregation.
                aggregated[new_name] = part.sum(axis=1, min_count=1)

        if not aggregated.empty:
            out[attr] = aggregated

    return out


def _capacity_weights(df: pd.DataFrame) -> pd.Series:
    """
    Return normalized p_nom weights or uniform weights if p_nom is unavailable.
    """
    if "p_nom" in df.columns:
        weights = pd.to_numeric(df["p_nom"], errors="coerce").fillna(0.0)
    else:
        weights = pd.Series(1.0, index=df.index)

    total = weights.sum()

    if total > 0:
        return weights / total

    return pd.Series(1.0 / len(df), index=df.index)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float | Any:
    """
    Return weighted average with a robust first-valid fallback.
    """
    numeric = pd.to_numeric(values, errors="coerce")

    if numeric.notna().any():
        aligned_weights = weights.reindex(values.index).fillna(0.0)
        valid = numeric.notna()
        weight_sum = aligned_weights.loc[valid].sum()

        if weight_sum > 0:
            return float((numeric.loc[valid] * aligned_weights.loc[valid]).sum() / weight_sum)

        return float(numeric.loc[valid].mean())

    return _first_valid(values)


def _first_valid(values: pd.Series) -> Any:
    """
    Return the first non-null value, or NaN if all values are null.
    """
    valid = values.dropna()

    if valid.empty:
        return np.nan

    return valid.iloc[0]


def _import_dynamic_dataframe(
    n: pypsa.Network,
    *,
    component: str,
    attr: str,
    df: pd.DataFrame,
) -> None:
    """
    Import a dynamic dataframe into a PyPSA network.

    PyPSA exposes this helper as a private method, but it is also used by
    PyPSA workflows internally and is the most direct way to reattach dynamic
    data after manual component aggregation.
    """
    if hasattr(n, "_import_series_from_df"):
        n._import_series_from_df(df, component, attr)
        return

    # Fallback for future API changes.
    dynamic_container = getattr(n, f"{component.lower()}s_t")
    setattr(dynamic_container, attr, df)


def _fix_clustered_bus_coordinates(
    *,
    n_original: pypsa.Network,
    n_clustered: pypsa.Network,
) -> None:
    """
    Set coordinates of clustered buses to representative bus coordinates.

    Auxiliary buses such as "<bus> H2" and "<bus> battery" inherit coordinates
    from their electric base bus.
    """
    if "x" not in n_original.buses.columns or "y" not in n_original.buses.columns:
        return

    if "x" not in n_clustered.buses.columns or "y" not in n_clustered.buses.columns:
        return

    for bus in n_clustered.buses.index.astype(str):
        base_bus = _base_bus_from_auxiliary_bus(bus)

        if base_bus not in n_original.buses.index:
            continue

        n_clustered.buses.at[bus, "x"] = float(n_original.buses.at[base_bus, "x"])
        n_clustered.buses.at[bus, "y"] = float(n_original.buses.at[base_bus, "y"])


def _base_bus_from_auxiliary_bus(bus: str) -> str:
    """
    Return the electric base bus name for known auxiliary bus suffixes.
    """
    suffixes = (" H2", " battery")

    for suffix in suffixes:
        if bus.endswith(suffix):
            return bus[: -len(suffix)]

    return bus


def _log_component_counts(n: pypsa.Network, *, prefix: str) -> None:
    """
    Log compact component counts.
    """
    logger.info(
        "%s: buses=%d, generators=%d, loads=%d, stores=%d, "
        "storage_units=%d, links=%d, lines=%d.",
        prefix,
        len(n.buses),
        len(n.generators),
        len(n.loads),
        len(n.stores),
        len(n.storage_units),
        len(n.links),
        len(n.lines),
    )