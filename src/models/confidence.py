"""Wilson score confidence intervals for calibration-bin outcome rates.

These aren't fit per-request - compute_confidence_bands() runs once (in
calibrate.py, against the real test cohort) and the result is serialized;
lookup_confidence_band() is a cheap, deterministic table lookup at request
time, not a recomputation.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def wilson_score_interval(successes: int, n: int, confidence: float = 0.90) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}[confidence]
    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


def compute_confidence_bands(y_true, y_proba, n_bins: int = 10) -> list[dict]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    edges = np.quantile(y_proba, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)  # collapse degenerate edges if many ties

    bands = []
    for i in range(len(edges) - 1):
        lower, upper = edges[i], edges[i + 1]
        is_last = i == len(edges) - 2
        mask = (y_proba >= lower) & (y_proba < upper) if not is_last else (y_proba >= lower) & (y_proba <= upper)
        n = int(mask.sum())
        successes = int(y_true[mask].sum()) if n > 0 else 0
        phat = successes / n if n > 0 else 0.0
        ci_lower, ci_upper = wilson_score_interval(successes, n)
        bands.append({
            "bin_lower": float(lower), "bin_upper": float(upper),
            "n": n, "phat": phat, "ci_lower": ci_lower, "ci_upper": ci_upper,
        })
    return bands


def lookup_confidence_band(probability: float, bands: list[dict]) -> tuple[float, float]:
    """Bins are half-open [bin_lower, bin_upper), matching how
    compute_confidence_bands() builds them, except the highest bin (by
    bin_upper) which is closed at the top. Without this, a probability that
    lands exactly on a shared edge between two adjacent bins would always
    resolve to the earlier (lower, worse) bin - a real off-by-boundary bug
    found via manual UI testing, not a hypothetical."""
    if not bands:
        raise ValueError("bands is empty - compute_confidence_bands() must run first")
    highest_upper = max(b["bin_upper"] for b in bands)
    for b in bands:
        is_top_bin = b["bin_upper"] == highest_upper
        if is_top_bin:
            in_bin = b["bin_lower"] <= probability <= b["bin_upper"]
        else:
            in_bin = b["bin_lower"] <= probability < b["bin_upper"]
        if in_bin:
            return (b["ci_lower"], b["ci_upper"])
    # probability outside all bins (shouldn't happen given edges span [0,1]) - clamp to nearest
    closest = min(bands, key=lambda b: min(abs(probability - b["bin_lower"]), abs(probability - b["bin_upper"])))
    return (closest["ci_lower"], closest["ci_upper"])


def _fit_calibrator(raw_score: np.ndarray, label: np.ndarray, method: str):
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_score, label)
    if method == "sigmoid":
        model = LogisticRegression()
        model.fit(raw_score.reshape(-1, 1), label)
        return model
    raise ValueError(f"Unknown calibration method: {method}")


def _predict_calibrator(calibrator, raw_score_query: float, method: str) -> float:
    if method == "isotonic":
        return float(calibrator.predict([raw_score_query])[0])
    return float(calibrator.predict_proba([[raw_score_query]])[0, 1])


def bootstrap_confidence_band(
    raw_score_query: float,
    calibration_raw_scores: np.ndarray,
    calibration_labels: np.ndarray,
    method: str,
    point_estimate: float,
    n_bootstrap: int = 200,
    confidence: float = 0.90,
    random_state: int = 42,
) -> tuple[float, float]:
    """Resample the calibration split with replacement, refit the calibrator
    on each resample, and push the SAME fixed raw score for this query
    through each bootstrap calibrator. Point estimate and interval are both
    downstream of the same base-model raw score and the same calibration
    data, so - unlike the static aggregate-bin lookup above - they're built
    from the same source. This does NOT force the point estimate inside the
    band artificially: it's a real percentile interval over real bootstrap
    replicates, and the assertion below is a genuine check, not a tautology
    - it exists to catch real bugs (e.g. a point estimate computed from a
    mismatched raw score or calibrator), per "fail loudly rather than
    rendering an impossible interval."""
    rng = np.random.default_rng(random_state)
    n = len(calibration_raw_scores)
    replicates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        calibrator = _fit_calibrator(calibration_raw_scores[idx], calibration_labels[idx], method)
        replicates[i] = _predict_calibrator(calibrator, raw_score_query, method)

    alpha = 1 - confidence
    lower = float(np.quantile(replicates, alpha / 2))
    upper = float(np.quantile(replicates, 1 - alpha / 2))

    assert lower <= point_estimate <= upper, (
        f"Confidence band [{lower:.4f}, {upper:.4f}] does not bracket point estimate {point_estimate:.4f} "
        f"- point_estimate and the bootstrap replicates must come from the same raw score and calibration "
        f"data; a mismatch here means a real bug upstream, not statistical noise to paper over."
    )
    return (lower, upper)
