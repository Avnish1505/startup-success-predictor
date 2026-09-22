"""Calibrate the boosted model's raw predict_proba against a held-out split
that neither the base model nor the test cohort ever saw."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split

from src.data import schema
from src.models.evaluate import calibration_curve_data, compute_metrics
from src.models.percentile import build_reference_distribution, save_reference_distribution
from src.models.train import ModelSpec, build_pipeline, git_provenance

DATA_PATH = Path("data/processed/startups_features_v1.parquet")
PRODUCTION_DIR = Path("models/production")
FIGURES_DIR = Path("reports/figures")
MODEL_NAME = "hist_gradient_boosting_full_calibrated"


def choose_calibration_method(n_samples: int, threshold: int = 1000) -> str:
    return "isotonic" if n_samples >= threshold else "sigmoid"


def split_train_for_calibration(train_df: pd.DataFrame, calibration_frac: float = 0.2, random_state: int = 42):
    train_fit, calibration = train_test_split(
        train_df, test_size=calibration_frac, stratify=train_df["label"], random_state=random_state
    )
    return train_fit, calibration


def _plot_reliability(y_true, y_proba, title: str, brier: float, out_path: Path) -> None:
    cal = calibration_curve_data(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(cal["mean_predicted"], cal["fraction_positive"], marker="o", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(f"{title} (Brier = {brier:.4f})")
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    train_df, test_df = df[df["split"] == "train"], df[df["split"] == "test"]

    train_fit, calibration = split_train_for_calibration(train_df)
    method = choose_calibration_method(len(calibration))
    print(f"train_fit={len(train_fit)} calibration={len(calibration)} -> method={method}")

    spec = ModelSpec(
        "hist_gradient_boosting_full_base", HistGradientBoostingClassifier(random_state=42),
        schema.FULL_NUMERIC_FEATURES, schema.FULL_CATEGORICAL_FEATURES, "ordinal", "full", needs_scaling=False,
    )
    feature_cols = spec.numeric_features + spec.categorical_features

    base_pipeline = build_pipeline(spec)
    base_pipeline.fit(train_fit[feature_cols], train_fit["label"])

    calibrated_pipeline = CalibratedClassifierCV(FrozenEstimator(base_pipeline), method=method)
    calibrated_pipeline.fit(calibration[feature_cols], calibration["label"])

    X_test, y_test = test_df[feature_cols], test_df["label"]
    proba_before = base_pipeline.predict_proba(X_test)[:, 1]
    proba_after = calibrated_pipeline.predict_proba(X_test)[:, 1]

    metrics_before = compute_metrics(y_test, proba_before, threshold=0.5)
    metrics_after = compute_metrics(y_test, proba_after, threshold=0.5)
    print(f"Brier before calibration: {metrics_before['brier_score']:.4f}")
    print(f"Brier after calibration:  {metrics_after['brier_score']:.4f}")

    _plot_reliability(y_test, proba_before, "Reliability - before calibration",
                       metrics_before["brier_score"], FIGURES_DIR / "calibration_before.png")
    _plot_reliability(y_test, proba_after, "Reliability - after calibration",
                       metrics_after["brier_score"], FIGURES_DIR / "calibration_after.png")

    train_scores = calibrated_pipeline.predict_proba(train_df[feature_cols])[:, 1]
    reference = build_reference_distribution(train_scores)

    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_pipeline, PRODUCTION_DIR / f"{MODEL_NAME}.joblib")
    joblib.dump(base_pipeline, PRODUCTION_DIR / f"{MODEL_NAME}_base.joblib")
    save_reference_distribution(reference, PRODUCTION_DIR / "train_score_distribution.npy")

    metadata = {
        "model_name": MODEL_NAME,
        "calibration_method": method,
        "numeric_features": spec.numeric_features,
        "categorical_features": spec.categorical_features,
        "n_train_fit": int(len(train_fit)),
        "n_calibration": int(len(calibration)),
        "n_test": int(len(test_df)),
        "brier_before": metrics_before["brier_score"],
        "brier_after": metrics_after["brier_score"],
        "roc_auc_before": metrics_before["roc_auc"],
        "roc_auc_after": metrics_after["roc_auc"],
        **git_provenance(),
    }
    (PRODUCTION_DIR / f"{MODEL_NAME}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote production artifacts to {PRODUCTION_DIR}/")


if __name__ == "__main__":
    main()
