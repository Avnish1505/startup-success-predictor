import numpy as np

from src.models.percentile import build_reference_distribution, compute_percentile


def test_build_reference_distribution_sorts():
    scores = np.array([0.5, 0.1, 0.9, 0.3])
    ref = build_reference_distribution(scores)
    assert list(ref) == [0.1, 0.3, 0.5, 0.9]


def test_compute_percentile_at_extremes():
    ref = build_reference_distribution(np.linspace(0.0, 1.0, 101))
    assert compute_percentile(-1.0, ref) == 0.0
    assert compute_percentile(2.0, ref) == 100.0


def test_compute_percentile_at_median():
    ref = build_reference_distribution(np.linspace(0.0, 1.0, 101))
    pct = compute_percentile(0.5, ref)
    assert 48.0 <= pct <= 52.0


def test_compute_percentile_monotonic():
    ref = build_reference_distribution(np.random.default_rng(0).uniform(0, 1, 500))
    p_low = compute_percentile(0.2, ref)
    p_high = compute_percentile(0.8, ref)
    assert p_low < p_high
