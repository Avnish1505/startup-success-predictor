"""Fetch the raw Crunchbase startup CSV from Kaggle via kagglehub."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.data import schema

_CSV_FILENAME = "big_startup_secsees_dataset.csv"


def download_dataset() -> Path:
    """Download (or reuse the cached copy of) the Kaggle dataset.

    kagglehub caches downloads under ~/.cache/kagglehub and skips re-downloading
    when the version is already present, so this is safe to call repeatedly.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Run: pip install kagglehub"
        ) from exc

    try:
        dataset_dir = kagglehub.dataset_download(schema.KAGGLE_DATASET)
    except Exception as exc:
        has_env_creds = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        has_file_creds = (Path.home() / ".kaggle" / "kaggle.json").exists()
        if not (has_env_creds or has_file_creds):
            raise RuntimeError(
                "Kaggle credentials not found. Create ~/.kaggle/kaggle.json "
                "(from https://www.kaggle.com/settings -> API -> Create New Token) "
                "or set KAGGLE_USERNAME and KAGGLE_KEY environment variables, then retry."
            ) from exc
        raise RuntimeError(f"Failed to download Kaggle dataset '{schema.KAGGLE_DATASET}': {exc}") from exc

    return Path(dataset_dir) / _CSV_FILENAME


def load_raw_csv(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV as strings (no dtype inference) for explicit coercion downstream."""
    if csv_path is None:
        csv_path = download_dataset()
    df = pd.read_csv(csv_path, dtype=str)
    missing_cols = set(schema.RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Downloaded CSV is missing expected columns: {sorted(missing_cols)}")
    df["funding_rounds"] = pd.to_numeric(df["funding_rounds"], errors="raise")
    return df
