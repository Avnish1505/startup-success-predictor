import numpy as np
import pytest

from src.models.confidence import (
    compute_confidence_bands,
    lookup_confidence_band,
    wilson_score_interval,
)


def test_wilson_interval_narrows_with_more_data():
    lo_small, hi_small = wilson_score_interval(5, 10, confidence=0.90)
    lo_large, hi_large = wilson_score_interval(500, 1000, confidence=0.90)
    assert (hi_small - lo_small) > (hi_large - lo_large)


def test_wilson_interval_contains_observed_rate():
    lo, hi = wilson_score_interval(30, 100, confidence=0.90)
    assert lo <= 0.30 <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_zero_successes():
    lo, hi = wilson_score_interval(0, 50, confidence=0.90)
    assert lo == 0.0
    assert hi > 0.0


def test_compute_confidence_bands_covers_full_range_and_sums_to_n():
    rng = np.random.default_rng(0)
    y_proba = rng.uniform(0, 1, size=500)
    y_true = (rng.uniform(0, 1, size=500) < y_proba).astype(int)
    bands = compute_confidence_bands(y_true, y_proba, n_bins=5)
    assert len(bands) == 5
    assert sum(b["n"] for b in bands) == 500
    assert bands[0]["bin_lower"] <= y_proba.min()
    assert bands[-1]["bin_upper"] >= y_proba.max()


def test_lookup_confidence_band_finds_containing_bin():
    bands = [
        {"bin_lower": 0.0, "bin_upper": 0.5, "n": 100, "phat": 0.2, "ci_lower": 0.13, "ci_upper": 0.29},
        {"bin_lower": 0.5, "bin_upper": 1.0, "n": 100, "phat": 0.8, "ci_lower": 0.71, "ci_upper": 0.87},
    ]
    lo, hi = lookup_confidence_band(0.9, bands)
    assert (lo, hi) == (0.71, 0.87)
    lo, hi = lookup_confidence_band(0.1, bands)
    assert (lo, hi) == (0.13, 0.29)


def test_lookup_confidence_band_raises_on_empty_bands():
    with pytest.raises(ValueError):
        lookup_confidence_band(0.5, [])


def test_lookup_confidence_band_shared_edge_resolves_to_upper_bin():
    # A probability landing exactly on the shared edge between two bins must
    # resolve to the upper (better) bin, not silently fall into the lower
    # one - a real bug found via manual UI testing (probability 0.676 landed
    # on a real bin edge and returned the wrong, much lower band).
    bands = [
        {"bin_lower": 0.0, "bin_upper": 0.5, "n": 100, "phat": 0.2, "ci_lower": 0.13, "ci_upper": 0.29},
        {"bin_lower": 0.5, "bin_upper": 1.0, "n": 100, "phat": 0.8, "ci_lower": 0.71, "ci_upper": 0.87},
    ]
    lo, hi = lookup_confidence_band(0.5, bands)
    assert (lo, hi) == (0.71, 0.87)


def test_lookup_confidence_band_top_bin_is_closed_at_its_upper_edge():
    bands = [
        {"bin_lower": 0.0, "bin_upper": 0.5, "n": 100, "phat": 0.2, "ci_lower": 0.13, "ci_upper": 0.29},
        {"bin_lower": 0.5, "bin_upper": 1.0, "n": 100, "phat": 0.8, "ci_lower": 0.71, "ci_upper": 0.87},
    ]
    lo, hi = lookup_confidence_band(1.0, bands)
    assert (lo, hi) == (0.71, 0.87)
