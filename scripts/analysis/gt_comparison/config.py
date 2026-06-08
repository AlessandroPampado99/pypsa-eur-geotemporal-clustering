# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Configuration utilities for geo-temporal comparison analysis.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(config_path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Perform minimal validation of required config sections."""
    required_sections = ["paths", "network", "reference", "runs", "carriers", "metrics"]
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing required config section: '{section}'")

    required_paths = ["root_dir", "runs_dir", "output_dir"]
    for key in required_paths:
        if key not in config["paths"]:
            raise KeyError(f"Missing required paths key: '{key}'")

    if "path_template" not in config["network"]:
        raise KeyError("Missing required network key: 'path_template'")

    if "filename" not in config["network"]:
        raise KeyError("Missing required network key: 'filename'")

    if "run" not in config["reference"]:
        raise KeyError("Missing required reference key: 'run'")

    if "include" not in config["runs"]:
        raise KeyError("Missing required runs key: 'include'")