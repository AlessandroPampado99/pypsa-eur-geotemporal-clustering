# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Metric computation for geo-temporal clustering comparison analysis.
"""

from typing import Any

import numpy as np
import pandas as pd


def get_snapshot_weights(n, column: str = "generators") -> pd.Series:
    """
    Return snapshot weights for energy aggregation.

    If the requested column is not available, the function falls back to the first
    available column in n.snapshot_weightings.
    """
    weights = n.snapshot_weightings

    if isinstance(weights, pd.DataFrame):
        if column in weights.columns:
            return weights[column].astype(float)

        fallback_column = weights.columns[0]
        print(
            f"[WARNING] Snapshot weighting column '{column}' not found. "
            f"Using '{fallback_column}' instead."
        )
        return weights[fallback_column].astype(float)

    return weights.astype(float)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a scalar-like value to float with a safe default."""
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass

    return float(value)


def compute_objective(n) -> float:
    """Compute total objective as objective + objective_constant."""
    objective = safe_float(getattr(n, "objective", 0.0), default=0.0)
    objective_constant = safe_float(getattr(n, "objective_constant", 0.0), default=0.0)
    return objective + objective_constant


def weighted_sum_timeseries(df: pd.DataFrame, weights: pd.Series) -> float:
    """Compute weighted sum over snapshots and assets."""
    if df is None or df.empty:
        return 0.0

    common_index = df.index.intersection(weights.index)
    if common_index.empty:
        raise ValueError("No common snapshots between time series and snapshot weights.")

    weighted = df.loc[common_index].mul(weights.loc[common_index], axis=0)
    return float(weighted.sum().sum())


def compute_total_demand(n, weights: pd.Series) -> float:
    """Compute total weighted demand from loads_t.p_set."""
    if not hasattr(n, "loads_t") or not hasattr(n.loads_t, "p_set"):
        return 0.0

    return weighted_sum_timeseries(n.loads_t.p_set, weights)


def compute_generation_by_carrier(
    n,
    weights: pd.Series,
    run_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute weighted generator production by carrier."""
    if n.generators.empty or n.generators_t.p.empty:
        return []

    generation = n.generators_t.p
    generators = n.generators

    common_generators = generation.columns.intersection(generators.index)
    generation = generation[common_generators]
    generators = generators.loc[common_generators]

    rows = []
    for carrier, asset_names in generators.groupby("carrier").groups.items():
        asset_names = list(asset_names)
        value = weighted_sum_timeseries(generation[asset_names], weights)

        rows.append(
            {
                "run": run_info["run"],
                "n_nodes": run_info["n_nodes"],
                "n_days": run_info["n_days"],
                "scan_type": run_info["scan_type"],
                "component": "Generator",
                "carrier": carrier,
                "value": value,
            }
        )

    return rows


def compute_generation_group(
    generation_rows: list[dict[str, Any]],
    carriers: list[str],
) -> float:
    """Aggregate generation rows for a configured carrier group."""
    carriers = set(carriers)
    return float(sum(row["value"] for row in generation_rows if row["carrier"] in carriers))


def get_component_df(n, component: str) -> pd.DataFrame:
    """Return the static dataframe for a PyPSA component."""
    attr_name = {
        "Generator": "generators",
        "Link": "links",
        "Store": "stores",
        "StorageUnit": "storage_units",
        "Line": "lines",
        "Transformer": "transformers",
    }.get(component)

    if attr_name is None:
        raise ValueError(f"Unsupported component: {component}")

    return getattr(n, attr_name)


def choose_capacity_series(
    df: pd.DataFrame,
    attribute: str,
    fallback_attribute: str | None = None,
) -> tuple[pd.Series, str]:
    """
    Return the preferred capacity series and the actual attribute used.

    The preferred attribute is used if it exists and has at least one non-null value.
    Otherwise, the fallback attribute is used.
    """
    if attribute in df.columns and df[attribute].notna().any():
        return df[attribute].fillna(0.0).astype(float), attribute

    if fallback_attribute and fallback_attribute in df.columns:
        return df[fallback_attribute].fillna(0.0).astype(float), fallback_attribute

    return pd.Series(0.0, index=df.index, dtype=float), attribute


def compute_capacity_by_component_carrier(
    n,
    run_info: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute capacity by component and carrier."""
    rows = []
    component_config = config.get("capacity", {}).get("components", {})

    for component, cfg in component_config.items():
        if not cfg.get("enabled", True):
            continue

        df = get_component_df(n, component)
        if df.empty:
            continue

        attribute = cfg["attribute"]
        fallback_attribute = cfg.get("fallback_attribute")

        capacity, used_attribute = choose_capacity_series(
            df=df,
            attribute=attribute,
            fallback_attribute=fallback_attribute,
        )

        if "carrier" in df.columns:
            carriers = df["carrier"].fillna("unknown").astype(str)
        else:
            carriers = pd.Series("all", index=df.index)

        temp = pd.DataFrame(
            {
                "carrier": carriers,
                "value": capacity,
            },
            index=df.index,
        )

        grouped = temp.groupby("carrier", dropna=False)["value"].sum()

        for carrier, value in grouped.items():
            rows.append(
                {
                    "run": run_info["run"],
                    "n_nodes": run_info["n_nodes"],
                    "n_days": run_info["n_days"],
                    "scan_type": run_info["scan_type"],
                    "component": component,
                    "carrier": carrier,
                    "attribute": used_attribute,
                    "value": float(value),
                }
            )

    return rows


def compute_capacity_group(
    capacity_rows: list[dict[str, Any]],
    component: str,
    carriers: list[str] | None = None,
) -> float:
    """Aggregate capacity rows for a component and optional carrier list."""
    total = 0.0
    carrier_set = set(carriers) if carriers is not None else None

    for row in capacity_rows:
        if row["component"] != component:
            continue

        if carrier_set is not None and row["carrier"] not in carrier_set:
            continue

        total += row["value"]

    return float(total)


def build_missing_carriers_report(
    run_info: dict[str, Any],
    available_generator_carriers: set[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Report configured carriers that are not present in the current network."""
    rows = []

    for group_name in ["renewable", "conventional"]:
        requested = set(config.get("carriers", {}).get(group_name, []))
        missing = sorted(requested - available_generator_carriers)

        for carrier in missing:
            rows.append(
                {
                    "run": run_info["run"],
                    "n_nodes": run_info["n_nodes"],
                    "n_days": run_info["n_days"],
                    "scan_type": run_info["scan_type"],
                    "carrier_group": group_name,
                    "missing_carrier": carrier,
                }
            )

    return rows


def collect_metrics_for_network(
    n,
    run_info: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Collect all configured metrics for one network."""
    weight_column = config.get("weights", {}).get("energy_weight_column", "generators")
    weights = get_snapshot_weights(n, column=weight_column)

    renewable_carriers = config["carriers"].get("renewable", [])
    conventional_carriers = config["carriers"].get("conventional", [])

    generation_rows = []
    capacity_rows = []
    missing_rows = []

    summary = {
        "run": run_info["run"],
        "n_nodes": run_info["n_nodes"],
        "n_days": run_info["n_days"],
        "scan_type": run_info["scan_type"],
        "network_path": run_info["network_path"],
    }

    metrics_cfg = config.get("metrics", {})

    if metrics_cfg.get("objective", True):
        summary["objective"] = compute_objective(n)

    if metrics_cfg.get("demand", True):
        summary["total_demand"] = compute_total_demand(n, weights)

    if metrics_cfg.get("generation", True):
        generation_rows = compute_generation_by_carrier(n, weights, run_info)

        summary["renewable_generation"] = compute_generation_group(
            generation_rows=generation_rows,
            carriers=renewable_carriers,
        )
        summary["conventional_generation"] = compute_generation_group(
            generation_rows=generation_rows,
            carriers=conventional_carriers,
        )

        available_generator_carriers = set(n.generators.carrier.dropna().astype(str).unique())
        missing_rows.extend(
            build_missing_carriers_report(
                run_info=run_info,
                available_generator_carriers=available_generator_carriers,
                config=config,
            )
        )

    if metrics_cfg.get("capacities", True):
        capacity_rows = compute_capacity_by_component_carrier(n, run_info, config)

        summary["renewable_capacity"] = compute_capacity_group(
            capacity_rows=capacity_rows,
            component="Generator",
            carriers=renewable_carriers,
        )
        summary["conventional_capacity"] = compute_capacity_group(
            capacity_rows=capacity_rows,
            component="Generator",
            carriers=conventional_carriers,
        )
        summary["store_energy_capacity"] = compute_capacity_group(
            capacity_rows=capacity_rows,
            component="Store",
            carriers=None,
        )
        summary["storage_unit_power_capacity"] = compute_capacity_group(
            capacity_rows=capacity_rows,
            component="StorageUnit",
            carriers=None,
        )
        summary["link_power_capacity"] = compute_capacity_group(
            capacity_rows=capacity_rows,
            component="Link",
            carriers=None,
        )
        summary["line_capacity"] = compute_capacity_group(
            capacity_rows=capacity_rows,
            component="Line",
            carriers=None,
        )

    return {
        "summary": summary,
        "generation_by_carrier": generation_rows,
        "capacity_by_component_carrier": capacity_rows,
        "missing_carriers": missing_rows,
    }