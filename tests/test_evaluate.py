import numpy as np

from src.models.evaluate import calibration_curve_data, choose_threshold, compute_metrics


def test_compute_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_compute_metrics_threshold_changes_confusion_matrix():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.3, 0.6, 0.4, 0.9])
    low = compute_metrics(y_true, y_proba, threshold=0.1)
    high = compute_metrics(y_true, y_proba, threshold=0.95)
    assert low["confusion_matrix"]["fp"] == 2  # everything predicted positive
    assert high["confusion_matrix"]["tp"] == 0  # nothing predicted positive
    # threshold doesn't change threshold-independent metrics
    assert low["roc_auc"] == high["roc_auc"]


def test_choose_threshold_prefers_precision_at_beta_half():
    # a classifier that is only confident (and correct) about a minority of positives;
    # low threshold catches more true positives but adds false positives
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.2, 0.3, 0.55, 0.6, 0.9])
    threshold, score = choose_threshold(y_true, y_proba, beta=0.5)
    # thresholds below 0.55 pull in a false positive; F0.5 should prefer avoiding it
    assert threshold >= 0.55
    assert 0.0 < score <= 1.0


def test_calibration_curve_data_shape():
    rng = np.random.default_rng(42)
    y_proba = rng.uniform(0, 1, size=200)
    y_true = (rng.uniform(0, 1, size=200) < y_proba).astype(int)
    data = calibration_curve_data(y_true, y_proba, n_bins=5)
    assert len(data["mean_predicted"]) == len(data["fraction_positive"])
    assert len(data["mean_predicted"]) <= 5
    assert all(0.0 <= v <= 1.0 for v in data["mean_predicted"])
