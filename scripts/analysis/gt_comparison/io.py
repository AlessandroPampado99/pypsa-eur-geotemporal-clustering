# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Input/output utilities for geo-temporal comparison analysis.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd
import pypsa


RUN_PATTERN = re.compile(r"n(?P<n_nodes>\d+)-d(?P<n_days>\d+)")


def parse_run_resolution(
    run_name: str,
    reference_n_nodes: int,
    reference_n_days: int,
) -> tuple[int, int]:
    """
    Parse n_nodes and n_days from a run name.

    The function searches for the pattern n<number>-d<number> anywhere in the run name.
    The reference run is treated separately and assigned the reference resolution.
    """
    match = RUN_PATTERN.search(run_name)

    if match is None:
        return reference_n_nodes, reference_n_days

    return int(match.group("n_nodes")), int(match.group("n_days"))


def classify_scan_type(
    run_name: str,
    n_nodes: int,
    n_days: int,
    reference_run: str,
    reference_n_nodes: int,
    reference_n_days: int,
) -> str:
    """Classify each run according to its location in the N-D resolution space."""
    if run_name == reference_run:
        return "complete"

    if n_days == reference_n_days and n_nodes != reference_n_nodes:
        return "nodes_scan"

    if n_nodes == reference_n_nodes and n_days != reference_n_days:
        return "days_scan"

    return "mixed_scan"


def build_network_path(config: dict[str, Any], run_name: str) -> Path:
    """Build the network path for a run using the configured template."""
    root_dir = str(Path(config["paths"]["root_dir"]).expanduser().resolve())
    runs_dir = config["paths"]["runs_dir"]
    filename = config["network"]["filename"]
    path_template = config["network"]["path_template"]

    path = path_template.format(
        root_dir=root_dir,
        runs_dir=runs_dir,
        run=run_name,
        filename=filename,
    )
    return Path(path).expanduser().resolve()


def discover_runs(config: dict[str, Any]) -> list[str]:
    """
    Discover run directories.

    This assumes that run folders are direct children of {root_dir}/{runs_dir}.
    """
    root_dir = Path(config["paths"]["root_dir"]).expanduser().resolve()
    runs_dir = root_dir / config["paths"]["runs_dir"]

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    run_names = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])
    return run_names


def build_run_table(config: dict[str, Any]) -> pd.DataFrame:
    """Build a table with run metadata and network paths."""
    reference_run = config["reference"]["run"]
    reference_n_nodes = int(config["reference"]["n_nodes"])
    reference_n_days = int(config["reference"]["n_days"])

    if config["runs"].get("auto_discover", False):
        run_names = discover_runs(config)
    else:
        run_names = list(config["runs"].get("include", []))

    exclude = set(config["runs"].get("exclude", []))
    run_names = [r for r in run_names if r not in exclude]

    if reference_run not in run_names:
        run_names = [reference_run] + run_names

    rows = []
    for run_name in run_names:
        n_nodes, n_days = parse_run_resolution(
            run_name=run_name,
            reference_n_nodes=reference_n_nodes,
            reference_n_days=reference_n_days,
        )
        scan_type = classify_scan_type(
            run_name=run_name,
            n_nodes=n_nodes,
            n_days=n_days,
            reference_run=reference_run,
            reference_n_nodes=reference_n_nodes,
            reference_n_days=reference_n_days,
        )
        network_path = build_network_path(config, run_name)

        rows.append(
            {
                "run": run_name,
                "n_nodes": n_nodes,
                "n_days": n_days,
                "scan_type": scan_type,
                "network_path": str(network_path),
            }
        )

    df = pd.DataFrame(rows)

    # Keep reference first, then sort by scan type and resolution.
    scan_order = {
        "complete": 0,
        "nodes_scan": 1,
        "days_scan": 2,
        "mixed_scan": 3,
    }
    df["_scan_order"] = df["scan_type"].map(scan_order).fillna(99)
    df = df.sort_values(["_scan_order", "n_days", "n_nodes", "run"]).drop(columns="_scan_order")

    return df.reset_index(drop=True)


def load_network(network_path: str | Path) -> pypsa.Network:
    """Load a PyPSA network from NetCDF."""
    return pypsa.Network(str(network_path))