import pandas as pd

from src.advisor.facts import assemble_facts_bundle, compute_cohort_stats, render_facts_bundle


def _synthetic_df(n_matching=50, n_total=200):
    rows = []
    for i in range(n_total):
        matching = i < n_matching
        rows.append({
            "country_code": "USA" if matching else "GBR",
            "primary_category": "Software" if matching else "Biotechnology",
            "funding_total_usd_log1p": 13.0 if matching else 20.0,  # ~440K vs ~485M
            "label": 1 if (matching and i % 2 == 0) else 0,
        })
    return pd.DataFrame(rows)


def test_compute_cohort_stats_exact_match_when_available():
    df = _synthetic_df(n_matching=50)
    result = compute_cohort_stats(df, "USA", "Software", funding_total_usd=440_000, min_n=30)
    assert result["level"] == "country+category+funding_band"
    assert result["n"] == 50
    assert result["success_rate"] == 0.5


def test_compute_cohort_stats_backs_off_to_country_and_category():
    # 20 USA+Software rows at this funding band (below min_n), but 20 more
    # USA+Software rows at a DIFFERENT funding band - country+category alone
    # should find 40 and report that level.
    rows = []
    for i in range(20):
        rows.append({"country_code": "USA", "primary_category": "Software", "funding_total_usd_log1p": 13.0, "label": i % 2})
    for i in range(20):
        rows.append({"country_code": "USA", "primary_category": "Software", "funding_total_usd_log1p": 20.0, "label": i % 2})
    df = pd.DataFrame(rows)
    result = compute_cohort_stats(df, "USA", "Software", funding_total_usd=440_000, min_n=30)
    assert result["level"] == "country+category"
    assert result["n"] == 40


def test_compute_cohort_stats_backs_off_to_country_alone():
    rows = []
    for i in range(35):
        rows.append({"country_code": "USA", "primary_category": "Biotechnology", "funding_total_usd_log1p": 20.0, "label": i % 2})
    df = pd.DataFrame(rows)
    result = compute_cohort_stats(df, "USA", "Software", funding_total_usd=440_000, min_n=30)
    assert result["level"] == "country"
    assert result["n"] == 35


def test_compute_cohort_stats_none_when_even_country_alone_too_small():
    df = _synthetic_df(n_matching=10, n_total=20)
    result = compute_cohort_stats(df, "USA", "Software", funding_total_usd=440_000, min_n=30)
    assert result is None


def test_assemble_facts_bundle_includes_raw_probability():
    bundle = assemble_facts_bundle(
        inputs={"country_code": "USA", "primary_category": "Software", "funding_total_usd": 1_000_000},
        probability=0.65, raw_probability=0.71, confidence_band=(0.55, 0.74),
        shap_contributions=[{"feature": "funding_total_usd_log1p", "value": 13.8, "shap_value": 0.5, "direction": "increases"}],
        percentile=72.3, cohort_stats=None,
    )
    assert bundle["raw_probability"] == 0.71


def test_render_facts_bundle_shows_raw_score_and_table():
    bundle = assemble_facts_bundle(
        inputs={"country_code": "USA", "primary_category": "Software", "funding_total_usd": 1_000_000},
        probability=0.65, raw_probability=0.71, confidence_band=(0.55, 0.74),
        shap_contributions=[{"feature": "funding_total_usd_log1p", "value": 13.8, "shap_value": 0.5, "direction": "increases"}],
        percentile=72.3, cohort_stats=None,
    )
    text = render_facts_bundle(bundle)
    assert "71.0%" in text  # raw score shown
    assert "| Feature | Value | SHAP | Direction |" in text  # a real table, not nested bullets
    assert "  - " not in text  # no nested-bullet markers left


def test_render_facts_bundle_foregrounds_named_feature():
    bundle = assemble_facts_bundle(
        inputs={"country_code": "USA", "primary_category": "Software", "funding_total_usd": 1_000_000},
        probability=0.65, raw_probability=0.71, confidence_band=(0.55, 0.74),
        shap_contributions=[
            {"feature": "primary_category", "value": "Software", "shap_value": 0.9, "direction": "increases"},
            {"feature": "funding_total_usd_log1p", "value": 13.8, "shap_value": -0.3, "direction": "decreases"},
        ],
        percentile=72.3, cohort_stats=None,
    )
    text = render_facts_bundle(bundle, foreground_feature="funding_total_usd_log1p")
    # the foregrounded feature's line appears before the generic table
    foreground_pos = text.index("funding_total_usd_log1p")
    table_pos = text.index("| Feature |")
    assert foreground_pos < table_pos
