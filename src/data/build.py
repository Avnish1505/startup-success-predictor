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
