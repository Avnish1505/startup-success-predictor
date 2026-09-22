"""Empirical CDF of predicted scores, for turning a probability into an
honest percentile against the training cohort's score distribution."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def build_reference_distribution(scores) -> np.ndarray:
    return np.sort(np.asarray(scores, dtype=float))


def compute_percentile(score: float, sorted_reference: np.ndarray) -> float:
    """Percentage of the reference distribution at or below `score`."""
    n = len(sorted_reference)
    rank = np.searchsorted(sorted_reference, score, side="right")
    return float(100.0 * rank / n)


def save_reference_distribution(sorted_reference: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, sorted_reference)


def load_reference_distribution(path: Path) -> np.ndarray:
    return np.load(Path(path))
