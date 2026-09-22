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


def test_compute_cohort_stats_returns_none_below_min_n():
    df = _synthetic_df(n_matching=10)
    result = compute_cohort_stats(df, "USA", "Software", funding_total_usd=440_000, min_n=30)
    assert result is None


def test_compute_cohort_stats_returns_real_stats_above_min_n():
    df = _synthetic_df(n_matching=50)
    result = compute_cohort_stats(df, "USA", "Software", funding_total_usd=440_000, min_n=30)
    assert result is not None
    assert result["n"] == 50
    assert result["success_rate"] == 0.5
    assert result["country_code"] == "USA"


def test_assemble_facts_bundle_structure():
    bundle = assemble_facts_bundle(
        inputs={"country_code": "USA", "primary_category": "Software", "funding_total_usd": 1_000_000},
        probability=0.65,
        confidence_band=(0.55, 0.74),
        shap_contributions=[{"feature": "funding_total_usd_log1p", "value": 13.8, "shap_value": 0.5, "direction": "increases"}],
        percentile=72.3,
        cohort_stats={"country_code": "USA", "primary_category": "Software", "funding_band": "$100K-$1M", "n": 40, "success_rate": 0.55},
    )
    assert bundle["probability"] == 0.65
    assert bundle["confidence_band"] == (0.55, 0.74)
    assert bundle["percentile"] == 72.3
    assert bundle["cohort"]["n"] == 40
    assert len(bundle["top_contributions"]) == 1


def test_assemble_facts_bundle_handles_suppressed_cohort():
    bundle = assemble_facts_bundle(
        inputs={"country_code": "USA", "primary_category": "Software", "funding_total_usd": 1_000_000},
        probability=0.65, confidence_band=(0.55, 0.74), shap_contributions=[], percentile=72.3,
        cohort_stats=None,
    )
    assert bundle["cohort"] is None


def test_render_facts_bundle_is_deterministic_and_grounded():
    bundle = assemble_facts_bundle(
        inputs={"country_code": "USA", "primary_category": "Software", "funding_total_usd": 1_000_000},
        probability=0.65, confidence_band=(0.55, 0.74),
        shap_contributions=[{"feature": "funding_total_usd_log1p", "value": 13.8, "shap_value": 0.5, "direction": "increases"}],
        percentile=72.3, cohort_stats=None,
    )
    text_a = render_facts_bundle(bundle)
    text_b = render_facts_bundle(bundle)
    assert text_a == text_b
    assert "65.0%" in text_a
    assert "55.0" in text_a and "74.0" in text_a
    assert "72.3" in text_a
    assert "not enough" in text_a.lower() or "n <" in text_a.lower()  # honest about suppressed cohort
