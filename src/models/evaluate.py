"""Pure metric, threshold-selection, and calibration-curve helpers.

No I/O and no model fitting here - these functions take arrays in and return
plain dicts out, so they're cheap to unit test and safe to reuse from
train.py, the CLI below, and the notebook.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(fbeta_score(y_true, y_pred, beta=1.0, zero_division=0)),
        "f0.5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "positive_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }


def choose_threshold(y_true, y_proba, beta: float = 0.5) -> tuple[float, float]:
    """Pick the threshold maximizing F-beta over out-of-fold predictions."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    candidates = np.linspace(0.01, 0.99, 99)
    best_threshold, best_score = 0.5, -1.0
    for t in candidates:
        y_pred = (y_proba >= t).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(t), float(score)
    return best_threshold, best_score


def calibration_curve_data(y_true, y_proba, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="quantile"
    )
    return {
        "mean_predicted": [float(v) for v in mean_predicted],
        "fraction_positive": [float(v) for v in fraction_positive],
        "n_bins_requested": n_bins,
    }


def evaluate_split(y_true, y_proba, threshold: float) -> dict:
    payload = compute_metrics(y_true, y_proba, threshold)
    payload["calibration"] = calibration_curve_data(y_true, y_proba)
    return payload


def _load_model_and_metadata(model_name: str):
    import json
    from pathlib import Path

    import joblib

    model_path = Path("models") / f"{model_name}.joblib"
    metadata_path = Path("models") / f"{model_name}_metadata.json"
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())
    return model, metadata


def main() -> None:
    import json
    import sys
    from pathlib import Path

    import pandas as pd

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.models.evaluate <model_name>")
    model_name = sys.argv[1]

    model, metadata = _load_model_and_metadata(model_name)
    df = pd.read_parquet("data/processed/startups_features_v1.parquet")
    test_df = df[df["split"] == "test"]
    feature_cols = metadata["numeric_features"] + metadata["categorical_features"]

    proba = model.predict_proba(test_df[feature_cols])[:, 1]
    payload = evaluate_split(test_df["label"], proba, metadata["threshold"])

    out_dir = Path("reports/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
