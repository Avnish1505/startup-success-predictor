"""Vary one feature across a grid, holding all others fixed, and record the
pipeline's predicted probability at each point - a single-feature partial
dependence curve for one concrete prediction."""
from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline


def compute_partial_dependence(
    pipeline: Pipeline,
    feature_cols: list[str],
    base_row: pd.DataFrame,
    varying_feature: str,
    grid_values: list,
) -> pd.DataFrame:
    rows = []
    for value in grid_values:
        row = base_row.copy()
        row[varying_feature] = value
        rows.append(row)
    grid_df = pd.concat(rows, ignore_index=True)
    proba = pipeline.predict_proba(grid_df[feature_cols])[:, 1]
    return pd.DataFrame({varying_feature: grid_values, "predicted_probability": proba})
