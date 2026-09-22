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
