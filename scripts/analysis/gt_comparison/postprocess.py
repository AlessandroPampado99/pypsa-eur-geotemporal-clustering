# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Post-processing utilities for reference deltas.
"""

import numpy as np
import pandas as pd


def add_reference_deltas(
    df: pd.DataFrame,
    reference_run: str,
    id_columns: list[str],
    value_column: str | None = None,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add delta and relative_delta columns against a reference run.

    If value_column is None, deltas are computed for all numeric columns that are
    not listed as id columns.

    If value_column is provided, the dataframe is assumed to be long format.
    """
    if df.empty:
        return df

    out = df.copy()

    if value_column is None:
        numeric_columns = [
            col
            for col in out.columns
            if col not in id_columns and pd.api.types.is_numeric_dtype(out[col])
        ]

        ref_rows = out[out["run"] == reference_run]
        if ref_rows.empty:
            raise ValueError(f"Reference run '{reference_run}' not found.")

        ref = ref_rows.iloc[0]

        for col in numeric_columns:
            ref_value = ref[col]
            out[f"{col}_delta"] = out[col] - ref_value
            out[f"{col}_relative_delta"] = np.where(
                ref_value != 0.0,
                out[f"{col}_delta"] / ref_value,
                np.nan,
            )

        return out

    if group_columns is None:
        group_columns = []

    ref = out[out["run"] == reference_run][group_columns + [value_column]].copy()
    if ref.empty:
        raise ValueError(f"Reference run '{reference_run}' not found.")

    ref = ref.rename(columns={value_column: "reference_value"})

    merged = out.merge(ref, on=group_columns, how="left")
    merged["delta"] = merged[value_column] - merged["reference_value"]
    merged["relative_delta"] = np.where(
        merged["reference_value"] != 0.0,
        merged["delta"] / merged["reference_value"],
        np.nan,
    )

    return merged