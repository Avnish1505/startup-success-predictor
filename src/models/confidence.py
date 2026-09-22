"""Wilson score confidence intervals for calibration-bin outcome rates.

These aren't fit per-request - compute_confidence_bands() runs once (in
calibrate.py, against the real test cohort) and the result is serialized;
lookup_confidence_band() is a cheap, deterministic table lookup at request
time, not a recomputation.
"""
from __future__ import annotations

import math

import numpy as np


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
    if not bands:
        raise ValueError("bands is empty - compute_confidence_bands() must run first")
    for b in bands:
        if b["bin_lower"] <= probability <= b["bin_upper"]:
            return (b["ci_lower"], b["ci_upper"])
    # probability outside all bins (shouldn't happen given edges span [0,1]) - clamp to nearest
    closest = min(bands, key=lambda b: min(abs(probability - b["bin_lower"]), abs(probability - b["bin_upper"])))
    return (closest["ci_lower"], closest["ci_upper"])
