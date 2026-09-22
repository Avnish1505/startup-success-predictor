# Real Crunchbase Dataset Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 6-row lookup table in `train_model.py` with a real, leakage-aware data pipeline built on the Kaggle "big-startup-secsees-fail-dataset-from-crunchbase" dataset (66,368 rows, verified by direct download).

**Architecture:** `src/data/download.py` fetches the raw CSV via `kagglehub` (idempotent, clear error on missing credentials). `src/data/schema.py` holds static column/label/leakage constants plus pure validation helpers. `src/data/build.py` holds pure, network-free transformation functions (label filtering, funding coercion, category extraction, time-based split, feature engineering, AUC-gap probe) plus a `main()` that wires I/O + transforms + reporting + parquet output. `tests/test_data.py` exercises the pure functions with small synthetic frames — no network/Kaggle credential needed to run the suite.

**Tech Stack:** pandas, numpy, scikit-learn (ColumnTransformer/Pipeline/LogisticRegression/OneHotEncoder), kagglehub, pyarrow (parquet), pytest.

**Spec:** user-supplied spec (see conversation) — Kaggle dataset `yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase`, CDLA-Sharing-1.0.

## Verified real-data facts (do not re-derive, do not invent)

Downloaded via `kagglehub.dataset_download('yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase')` → single file `big_startup_secsees_dataset.csv`.

- Shape: **66,368 rows × 14 columns**, columns exactly: `permalink, name, homepage_url, category_list, funding_total_usd, status, country_code, state_code, region, city, funding_rounds, founded_at, first_funding_at, last_funding_at`.
- `status` value counts: `operating 53034, closed 6238, acquired 5549, ipo 1547`. No other status values exist.
- After dropping `operating`: **13,334 labeled rows** — positive (acquired+ipo) = 7096 (53.2%), negative (closed) = 6238 (46.8%).
- `funding_total_usd`: string column, **no commas found in this version** (spec anticipated commas — real data has none, so the coercion must still handle them defensively but must not assume they exist). `-` is the missing sentinel: 12,785/66,368 rows overall, 2,191/13,334 in the labeled subset. Zero other unparseable values found.
- `category_list`: pipe-delimited. 27,296 unique full combos (not ~58k as the spec guessed — report the real number). 506 unique primary (first-token) categories within the labeled subset — one-hot-able.
- `country_code`: 84 uniques in labeled subset, 1991 nulls. `region`: 595 uniques, 2152 nulls.
- `founded_at` has data-entry garbage: 114 rows dataset-wide with year <1900 or >2015 (e.g. `1015-01-30`, `2914-01-01`), 22 of those fall in the labeled subset. 3732 labeled rows have null `founded_at`. Total labeled rows needing the `first_funding_at` fallback: 3754/13334. **Zero rows have both `founded_at` and `first_funding_at` unusable** — the fallback fully covers the dataset.
- `first_funding_at` max = `2015-12-05`; use this dataset-wide max(first_funding_at, last_funding_at) as the dynamic upper bound for founded_at plausibility (not a hardcoded year).
- Using effective date = founded_at (if plausible) else first_funding_at, an 80th-percentile split gives boundary **2011-12-01**, train=10,672 rows (59.8% positive), test=2,662 rows (26.7% positive). The positive-rate drop in the test cohort is expected (newer companies haven't had time to resolve to acquired/ipo) — call this out in DATA_CARD.md, don't treat it as a bug.
- `pyarrow` 25.0.1 and `kagglehub` 1.0.2 and `pytest` 9.1.1 are installed in `.venv`. Kaggle credentials exist at `~/.kaggle/kaggle.json` (user `avnish1505`) and the download already succeeded once (cached at `~/.cache/kagglehub/...`).

## Global Constraints

- Label: `acquired`/`ipo` → 1, `closed` → 0, `operating` → dropped entirely (never labeled 0). Any status outside these 4 values must raise, not silently pass through.
- Leaky columns (full-set only, never in clean): `funding_total_usd`, `funding_rounds`, `last_funding_at`.
- Clean feature set: `founded_year` (from effective date), `time_to_first_funding_days` (first_funding_at − plausible founded_at), `country_code`, `region`, `primary_category`.
- Full feature set: clean + `funding_total_usd_log1p`, `funding_rounds`, `funding_span_days` (last_funding_at − first_funding_at).
- All transformers (imputers, scalers, one-hot encoders) must be fit on the train split only, inside an sklearn `Pipeline`/`ColumnTransformer`.
- `funding_total_usd` coercion must raise a `ValueError` naming the offending values if anything fails to parse after handling the known `-` sentinel and comma separators — never silently coerce unknown garbage to NaN.
- `category_list` is never one-hot encoded whole; only `primary_category` (first `|`-token) is encoded. The raw `category_list` column is preserved in the processed output for future multi-hot work.
- Pure transformation logic (labeling, coercion, splitting, feature engineering) lives in functions that take/return DataFrames and touch no filesystem/network, so tests run without Kaggle credentials.
- Delete `train_model.py` as part of this work (no other file references it — verified via grep).

---

## Task 1: Schema module

**Files:**
- Create: `src/__init__.py` (empty)
- Create: `src/data/__init__.py` (empty)
- Create: `src/data/schema.py`

**Interfaces:**
- Produces: `RAW_COLUMNS: list[str]`, `STATUS_ACQUIRED/STATUS_IPO/STATUS_CLOSED/STATUS_OPERATING: str`, `VALID_STATUSES: set[str]`, `LABEL_MAP: dict[str,int]`, `LEAKY_COLUMNS: list[str]`, `CLEAN_NUMERIC_FEATURES/CLEAN_CATEGORICAL_FEATURES/FULL_NUMERIC_FEATURES/FULL_CATEGORICAL_FEATURES: list[str]`, `MIN_FOUNDED_YEAR: int = 1900`, `DATA_VERSION: str = "v1"`, `KAGGLE_DATASET: str = "yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase"`.

- [ ] **Step 1: Write `src/data/schema.py`**

```python
"""Static schema constants for the Crunchbase startup outcome dataset."""

KAGGLE_DATASET = "yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase"
DATA_VERSION = "v1"

RAW_COLUMNS = [
    "permalink", "name", "homepage_url", "category_list", "funding_total_usd",
    "status", "country_code", "state_code", "region", "city",
    "funding_rounds", "founded_at", "first_funding_at", "last_funding_at",
]

STATUS_ACQUIRED = "acquired"
STATUS_IPO = "ipo"
STATUS_CLOSED = "closed"
STATUS_OPERATING = "operating"
VALID_STATUSES = {STATUS_ACQUIRED, STATUS_IPO, STATUS_CLOSED, STATUS_OPERATING}

# operating is intentionally absent: it is dropped, never labeled.
LABEL_MAP = {STATUS_ACQUIRED: 1, STATUS_IPO: 1, STATUS_CLOSED: 0}

# Measured at scrape time, after the outcome is known. Full-set only.
LEAKY_COLUMNS = ["funding_total_usd", "funding_rounds", "last_funding_at"]

CLEAN_NUMERIC_FEATURES = ["founded_year", "time_to_first_funding_days"]
CLEAN_CATEGORICAL_FEATURES = ["country_code", "region", "primary_category"]

FULL_NUMERIC_FEATURES = CLEAN_NUMERIC_FEATURES + [
    "funding_total_usd_log1p", "funding_rounds", "funding_span_days",
]
FULL_CATEGORICAL_FEATURES = list(CLEAN_CATEGORICAL_FEATURES)

MIN_FOUNDED_YEAR = 1900
```

- [ ] **Step 2: Sanity-check the module imports cleanly**

Run: `python3 -c "from src.data import schema; print(schema.LABEL_MAP, schema.LEAKY_COLUMNS)"`
Expected: prints `{'acquired': 1, 'ipo': 1, 'closed': 0} ['funding_total_usd', 'funding_rounds', 'last_funding_at']`

- [ ] **Step 3: Commit**

```bash
git add src/__init__.py src/data/__init__.py src/data/schema.py
git commit -m "feat(data): add schema constants for Crunchbase dataset"
```

---

## Task 2: Download module

**Files:**
- Create: `src/data/download.py`

**Interfaces:**
- Consumes: `schema.KAGGLE_DATASET`
- Produces: `download_dataset() -> pathlib.Path` (idempotent — kagglehub caches under `~/.cache/kagglehub`, safe to call repeatedly), `load_raw_csv(csv_path: pathlib.Path | None = None) -> pandas.DataFrame` (if `csv_path` is None, calls `download_dataset()` first).

- [ ] **Step 1: Write `src/data/download.py`**

```python
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
```

- [ ] **Step 2: Verify it downloads and loads the real file**

Run: `python3 -c "from src.data.download import load_raw_csv; df = load_raw_csv(); print(df.shape); print(list(df.columns))"`
Expected: `(66368, 14)` and the 14 column names from the spec.

- [ ] **Step 3: Commit**

```bash
git add src/data/download.py
git commit -m "feat(data): add idempotent Kaggle dataset download"
```

---

## Task 3: Build module — pure transforms

**Files:**
- Create: `src/data/build.py`

**Interfaces:**
- Consumes: `schema.*`, `download.load_raw_csv`
- Produces (all pure, DataFrame-in/DataFrame-or-Series-out, no I/O):
  - `coerce_funding_total_usd(series: pd.Series) -> pd.Series`
  - `label_and_filter(df: pd.DataFrame) -> pd.DataFrame` (adds `label` column, drops `operating` rows)
  - `extract_primary_category(category_list: pd.Series) -> pd.Series`
  - `compute_dates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]` returns `(founded_clean, effective_date)`
  - `time_based_split(effective_date: pd.Series, test_quantile: float = 0.8) -> tuple[pd.Series, pd.Series, pd.Timestamp]` returns `(train_mask, test_mask, boundary)`
  - `engineer_features(df: pd.DataFrame, founded_clean: pd.Series, effective_date: pd.Series) -> pd.DataFrame` returns a frame with all CLEAN + FULL feature columns plus `category_list` and `label`
  - `build_probe_pipeline(numeric_features: list[str], categorical_features: list[str]) -> sklearn.pipeline.Pipeline`
  - `auc_gap(train_df, test_df, y_train, y_test) -> tuple[float, float]` returns `(auc_clean, auc_full)`
  - `main() -> None`

- [ ] **Step 1: Write the failing tests first (see Task 4) before implementing** — per TDD, write `tests/test_data.py` (Task 4) targeting these function names/signatures, run it, confirm failures (`ModuleNotFoundError`/`ImportError` on `src.data.build`), THEN come back and implement this file. Do not implement `build.py` before Task 4's tests exist and fail.

- [ ] **Step 2: Write `src/data/build.py`**

```python
"""Build leakage-aware train/test feature tables from the raw Crunchbase CSV."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import schema
from src.data.download import load_raw_csv

PROCESSED_DIR = Path("data/processed")


def coerce_funding_total_usd(series: pd.Series) -> pd.Series:
    """Coerce the string funding_total_usd column to float, surfacing bad values."""
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.mask(cleaned == "-")
    cleaned = cleaned.str.replace(",", "", regex=False)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    unparseable = cleaned.notna() & numeric.isna()
    if unparseable.any():
        bad_values = sorted(cleaned[unparseable].unique().tolist())[:10]
        raise ValueError(
            f"funding_total_usd has {int(unparseable.sum())} unparseable values, e.g. {bad_values}"
        )
    return numeric


def label_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop `operating` rows and label acquired/ipo=1, closed=0."""
    unknown = set(df["status"].dropna().unique()) - schema.VALID_STATUSES
    if unknown:
        raise ValueError(f"Unexpected status values not in schema.VALID_STATUSES: {unknown}")
    out = df[df["status"] != schema.STATUS_OPERATING].copy()
    out["label"] = out["status"].map(schema.LABEL_MAP)
    if out["label"].isna().any():
        bad = out.loc[out["label"].isna(), "status"].unique().tolist()
        raise ValueError(f"Rows with status not in LABEL_MAP after filtering operating: {bad}")
    out["label"] = out["label"].astype(int)
    return out


def extract_primary_category(category_list: pd.Series) -> pd.Series:
    return category_list.str.split("|").str[0]


def compute_dates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (founded_clean, effective_date).

    founded_clean: founded_at parsed and NaT'd out where implausible (year < MIN_FOUNDED_YEAR
    or later than the dataset's own max observed funding date, which is a proxy for scrape time).
    effective_date: founded_clean, falling back to first_funding_at where founded_clean is NaT.
    Used for both the time-based split and the founded_year feature.
    """
    founded_raw = pd.to_datetime(df["founded_at"], errors="coerce")
    first_funding = pd.to_datetime(df["first_funding_at"], errors="coerce")
    last_funding = pd.to_datetime(df["last_funding_at"], errors="coerce")
    upper_bound = max(first_funding.max(), last_funding.max())

    plausible = founded_raw.notna() & (founded_raw.dt.year >= schema.MIN_FOUNDED_YEAR) & (founded_raw <= upper_bound)
    founded_clean = founded_raw.where(plausible)
    effective_date = founded_clean.where(plausible, first_funding)
    return founded_clean, effective_date


def time_based_split(effective_date: pd.Series, test_quantile: float = 0.8):
    if effective_date.isna().any():
        raise ValueError("effective_date has nulls; cannot split rows with no usable date")
    boundary = effective_date.quantile(test_quantile)
    train_mask = effective_date <= boundary
    test_mask = ~train_mask
    return train_mask, test_mask, boundary


def engineer_features(df: pd.DataFrame, founded_clean: pd.Series, effective_date: pd.Series) -> pd.DataFrame:
    first_funding = pd.to_datetime(df["first_funding_at"], errors="coerce")
    last_funding = pd.to_datetime(df["last_funding_at"], errors="coerce")

    out = pd.DataFrame(index=df.index)
    out["founded_year"] = effective_date.dt.year.astype(float)
    out["time_to_first_funding_days"] = (first_funding - founded_clean).dt.days.astype(float)
    out["country_code"] = df["country_code"]
    out["region"] = df["region"]
    out["primary_category"] = extract_primary_category(df["category_list"])
    out["category_list"] = df["category_list"]

    out["funding_total_usd_log1p"] = np.log1p(coerce_funding_total_usd(df["funding_total_usd"]))
    out["funding_rounds"] = df["funding_rounds"].astype(float)
    out["funding_span_days"] = (last_funding - first_funding).dt.days.astype(float)

    out["label"] = df["label"].values
    out["permalink"] = df["permalink"].values
    out["name"] = df["name"].values
    return out


def build_probe_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def _fit_score(train_df, test_df, y_train, y_test, numeric_features, categorical_features) -> float:
    pipe = build_probe_pipeline(numeric_features, categorical_features)
    pipe.fit(train_df[numeric_features + categorical_features], y_train)
    proba = pipe.predict_proba(test_df[numeric_features + categorical_features])[:, 1]
    return float(roc_auc_score(y_test, proba))


def auc_gap(train_df, test_df, y_train, y_test) -> tuple[float, float]:
    auc_clean = _fit_score(train_df, test_df, y_train, y_test, schema.CLEAN_NUMERIC_FEATURES, schema.CLEAN_CATEGORICAL_FEATURES)
    auc_full = _fit_score(train_df, test_df, y_train, y_test, schema.FULL_NUMERIC_FEATURES, schema.FULL_CATEGORICAL_FEATURES)
    return auc_clean, auc_full


def main() -> None:
    raw = load_raw_csv()
    labeled = label_and_filter(raw)
    founded_clean, effective_date = compute_dates(labeled)
    features = engineer_features(labeled, founded_clean, effective_date)

    train_mask, test_mask, boundary = time_based_split(effective_date)
    features["split"] = np.where(train_mask, "train", "test")

    train_df, test_df = features[train_mask], features[test_mask]
    y_train, y_test = train_df["label"], test_df["label"]
    auc_clean, auc_full = auc_gap(train_df, test_df, y_train, y_test)

    print(f"Total rows (raw): {len(raw)}")
    print(f"Labeled rows (operating dropped): {len(labeled)}")
    pos_rate = labeled["label"].mean()
    print(f"Class balance: positive={labeled['label'].sum()} ({pos_rate:.1%}), negative={(labeled['label']==0).sum()} ({1-pos_rate:.1%})")
    print(f"Split boundary date: {boundary.date()}")
    print(f"Train rows: {len(train_df)} (positive rate {y_train.mean():.1%})")
    print(f"Test rows: {len(test_df)} (positive rate {y_test.mean():.1%})")
    print(f"Clean feature count: {len(schema.CLEAN_NUMERIC_FEATURES) + len(schema.CLEAN_CATEGORICAL_FEATURES)}")
    print(f"Full feature count: {len(schema.FULL_NUMERIC_FEATURES) + len(schema.FULL_CATEGORICAL_FEATURES)}")
    print(f"AUC (clean): {auc_clean:.4f}")
    print(f"AUC (full):  {auc_full:.4f}")
    print(f"AUC gap (full - clean): {auc_full - auc_clean:.4f}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"startups_features_{schema.DATA_VERSION}.parquet"
    features.to_parquet(out_path, index=False)
    print(f"Wrote {len(features)} rows to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the real pipeline end-to-end**

Run: `python3 -m src.data.build`
Expected: prints match the "Verified real-data facts" section above (13334 labeled rows, boundary 2011-12-01, train 10672/test 2662, etc.) and writes `data/processed/startups_features_v1.parquet`.

- [ ] **Step 4: Commit**

```bash
git add src/data/build.py
git commit -m "feat(data): build leakage-aware clean/full feature pipeline with AUC-gap report"
```

---

## Task 4: Tests

**Files:**
- Create: `tests/__init__.py` (empty, if needed for discovery — check pytest config first)
- Create: `tests/test_data.py`

**Interfaces:**
- Consumes: `src.data.schema`, `src.data.build` (all pure functions from Task 3)

- [ ] **Step 1: Write `tests/test_data.py`**

```python
import pandas as pd
import pytest

from src.data import schema
from src.data.build import (
    coerce_funding_total_usd,
    compute_dates,
    engineer_features,
    extract_primary_category,
    label_and_filter,
    time_based_split,
)


def _raw_frame():
    return pd.DataFrame({
        "permalink": ["/a", "/b", "/c", "/d"],
        "name": ["A", "B", "C", "D"],
        "homepage_url": [None, None, None, None],
        "category_list": ["Apps|Games", "Biotech", None, "Media|News"],
        "funding_total_usd": ["1,000,000", "-", "500000", "250000"],
        "status": ["acquired", "operating", "closed", "ipo"],
        "country_code": ["USA", "USA", "GBR", "USA"],
        "state_code": ["CA", "CA", None, "NY"],
        "region": ["SF Bay", "SF Bay", "London", "NYC"],
        "city": ["SF", "SF", "London", "NYC"],
        "funding_rounds": [2, 1, 1, 3],
        "founded_at": ["2005-01-01", "2010-01-01", "2008-06-15", "2009-03-01"],
        "first_funding_at": ["2006-01-01", "2011-01-01", "2008-09-01", "2009-06-01"],
        "last_funding_at": ["2007-01-01", "2012-01-01", "2008-09-01", "2010-01-01"],
    })


def test_label_map_acquired_ipo_are_one_closed_is_zero():
    labeled = label_and_filter(_raw_frame())
    labels = dict(zip(labeled["name"], labeled["label"]))
    assert labels["A"] == 1  # acquired
    assert labels["D"] == 1  # ipo
    assert labels["C"] == 0  # closed


def test_operating_rows_are_dropped_not_labeled_zero():
    labeled = label_and_filter(_raw_frame())
    assert "operating" not in labeled["status"].values
    assert "B" not in labeled["name"].values  # B was the operating row


def test_unexpected_status_raises():
    df = _raw_frame()
    df.loc[0, "status"] = "zombie"
    with pytest.raises(ValueError):
        label_and_filter(df)


def test_funding_total_usd_coerces_commas_and_dash():
    result = coerce_funding_total_usd(pd.Series(["1,000,000", "-", "500000"]))
    assert result.tolist()[0] == 1_000_000.0
    assert pd.isna(result.tolist()[1])
    assert result.tolist()[2] == 500000.0


def test_funding_total_usd_raises_on_garbage():
    with pytest.raises(ValueError):
        coerce_funding_total_usd(pd.Series(["1,000,000", "not-a-number"]))


def test_primary_category_takes_first_pipe_token():
    result = extract_primary_category(pd.Series(["Apps|Games|Mobile", "Biotech", None]))
    assert result.tolist()[0] == "Apps"
    assert result.tolist()[1] == "Biotech"
    assert pd.isna(result.tolist()[2])


def test_no_clean_column_is_in_the_leaky_list():
    clean_cols = set(schema.CLEAN_NUMERIC_FEATURES) | set(schema.CLEAN_CATEGORICAL_FEATURES)
    leaky_cols = set(schema.LEAKY_COLUMNS)
    assert clean_cols.isdisjoint(leaky_cols)


def test_no_train_test_date_overlap():
    labeled = label_and_filter(_raw_frame())
    founded_clean, effective_date = compute_dates(labeled)
    train_mask, test_mask, boundary = time_based_split(effective_date, test_quantile=0.5)
    train_dates = effective_date[train_mask]
    test_dates = effective_date[test_mask]
    assert train_dates.max() <= boundary
    if len(test_dates):
        assert test_dates.min() > boundary
    assert not (set(train_dates.index) & set(test_dates.index))


def test_engineer_features_produces_clean_and_full_columns():
    labeled = label_and_filter(_raw_frame())
    founded_clean, effective_date = compute_dates(labeled)
    features = engineer_features(labeled, founded_clean, effective_date)
    for col in schema.CLEAN_NUMERIC_FEATURES + schema.CLEAN_CATEGORICAL_FEATURES:
        assert col in features.columns
    for col in schema.FULL_NUMERIC_FEATURES + schema.FULL_CATEGORICAL_FEATURES:
        assert col in features.columns
    assert "category_list" in features.columns
```

- [ ] **Step 2: Run tests, confirm they fail before `build.py` exists**

Run: `pytest tests/test_data.py -v`
Expected (before Task 3 Step 2): `ModuleNotFoundError: No module named 'src.data.build'` (or `ImportError`)

- [ ] **Step 3: After Task 3 is implemented, run tests again**

Run: `pytest tests/test_data.py -v`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_data.py
git commit -m "test(data): cover labeling, leakage split, and date-overlap invariants"
```

---

## Task 5: DATA_CARD.md, delete train_model.py, final verification

**Files:**
- Create: `DATA_CARD.md`
- Delete: `train_model.py`

**Interfaces:**
- Consumes: real numbers printed by `python -m src.data.build` (Task 3 Step 3 output) — DATA_CARD.md must report the actual run's numbers, not the estimates in this plan, in case pandas/kagglehub versions shift anything.

- [ ] **Step 1: Write `DATA_CARD.md`**

Content requirements (fill in with the actual `python -m src.data.build` output captured in Task 3 Step 3): source URL + Kaggle handle, CDLA-Sharing-1.0 license note, raw row/column counts, class balance (raw status breakdown + post-filter positive/negative counts and rates), split boundary date and train/test sizes/rates, clean vs full feature lists, the AUC gap finding, and an explicit survivorship-bias caveat paragraph: Crunchbase under-reports failed startups (many closures are never recorded as "closed" and instead vanish or stay "operating" indefinitely), so the observed 53%/47% positive rate among *resolved* companies overstates real-world startup success — this dataset's baseline should not be read as "53% of startups succeed."

- [ ] **Step 2: Delete `train_model.py`**

Run: `git rm train_model.py`
Verify no other file imports it: `grep -rn "train_model" --include="*.py" . | grep -v .venv` should already have returned nothing (verified during planning).

- [ ] **Step 3: Run full verification**

Run: `pytest -v && python3 -m src.data.build`
Expected: all tests pass; build script prints the full report and writes the parquet file.

- [ ] **Step 4: Commit**

```bash
git add DATA_CARD.md
git commit -m "docs(data): add DATA_CARD and delete synthetic train_model.py"
```

Note: `train_model.py` deletion is staged via `git rm` in Step 2; include it in this same commit per the spec ("Delete train_model.py in the same commit").

---

## Self-review notes

- Spec coverage: label rule (Task 3), leakage control clean/full + AUC gap (Task 3), time-based split with printed boundary + fit-on-train-only (Task 3), funding_total_usd coercion with surfaced errors (Task 3), category_list primary extraction without full one-hot (Task 3), all 4 deliverable files + tests + DATA_CARD (Tasks 1-5), train_model.py deletion in the same commit (Task 5), acceptance criteria print statements (Task 3 `main()`).
- No placeholders: all steps carry full code.
- Type consistency checked: `compute_dates` returns `(founded_clean, effective_date)` consistently used in `engineer_features` and `time_based_split` calls in both `build.py` and `test_data.py`.
