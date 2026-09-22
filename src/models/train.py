"""Train dummy/logistic-regression/HistGradientBoosting baselines on the
leakage-aware Crunchbase dataset, with an explicit clean-vs-full comparison.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data import schema
from src.models.evaluate import choose_threshold, evaluate_split

DATA_PATH = Path("data/processed/startups_features_v1.parquet")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
SPLIT_BOUNDARY_DATE = "2011-12-01"  # printed by `python -m src.data.build`


@dataclass
class ModelSpec:
    name: str
    estimator: object
    numeric_features: list
    categorical_features: list
    categorical_encoding: str  # "onehot" | "ordinal"
    feature_set: str  # "clean" | "full"
    needs_scaling: bool


def build_pipeline(spec: ModelSpec) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if spec.needs_scaling:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(numeric_steps)

    if spec.categorical_encoding == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore")
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", encoder),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, spec.numeric_features),
        ("cat", categorical_transformer, spec.categorical_features),
    ])
    return Pipeline([("preprocess", preprocessor), ("clf", spec.estimator)])


def build_specs() -> list[ModelSpec]:
    clean_num, clean_cat = schema.CLEAN_NUMERIC_FEATURES, schema.CLEAN_CATEGORICAL_FEATURES
    full_num, full_cat = schema.FULL_NUMERIC_FEATURES, schema.FULL_CATEGORICAL_FEATURES
    return [
        ModelSpec("dummy_most_frequent", DummyClassifier(strategy="most_frequent"),
                  full_num, full_cat, "onehot", "full", needs_scaling=False),
        ModelSpec("logistic_regression_full", LogisticRegression(max_iter=2000, random_state=42),
                  full_num, full_cat, "onehot", "full", needs_scaling=True),
        ModelSpec("logistic_regression_clean", LogisticRegression(max_iter=2000, random_state=42),
                  clean_num, clean_cat, "onehot", "clean", needs_scaling=True),
        ModelSpec("hist_gradient_boosting_full", HistGradientBoostingClassifier(random_state=42),
                  full_num, full_cat, "ordinal", "full", needs_scaling=False),
        ModelSpec("hist_gradient_boosting_clean", HistGradientBoostingClassifier(random_state=42),
                  clean_num, clean_cat, "ordinal", "clean", needs_scaling=False),
    ]


def git_provenance() -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
    return {"git_sha": sha, "git_dirty": dirty}


def run_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision", "neg_brier_score": "neg_brier_score"}
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(scores["test_roc_auc"])),
        "average_precision_mean": float(np.mean(scores["test_average_precision"])),
        "average_precision_std": float(np.std(scores["test_average_precision"])),
        "brier_score_mean": float(-np.mean(scores["test_neg_brier_score"])),
        "brier_score_std": float(np.std(scores["test_neg_brier_score"])),
        "n_splits": 5,
        "n_repeats": 10,
    }


def run_oof_threshold(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return choose_threshold(y, oof_proba, beta=0.5)


def train_one(spec: ModelSpec, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    feature_cols = spec.numeric_features + spec.categorical_features
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test, y_test = test_df[feature_cols], test_df["label"]

    cv_metrics = run_cv(build_pipeline(spec), X_train, y_train)
    threshold, oof_f_beta = run_oof_threshold(build_pipeline(spec), X_train, y_train)

    final_pipeline = build_pipeline(spec)
    final_pipeline.fit(X_train, y_train)
    test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_split(y_test, test_proba, threshold)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODELS_DIR / f"{spec.name}.joblib")

    metadata = {
        "model_name": spec.name,
        "feature_set": spec.feature_set,
        "numeric_features": spec.numeric_features,
        "categorical_features": spec.categorical_features,
        "categorical_encoding": spec.categorical_encoding,
        "split_boundary_date": SPLIT_BOUNDARY_DATE,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "threshold": threshold,
        "threshold_selection": "F0.5-maximizing on 5-fold stratified out-of-fold predictions",
        "oof_f0.5": oof_f_beta,
        "cv": cv_metrics,
        "test": test_metrics,
        **git_provenance(),
    }
    (MODELS_DIR / f"{spec.name}_metadata.json").write_text(json.dumps(metadata, indent=2))

    return {"spec": spec, "test_proba": test_proba, "metadata": metadata}


def plot_comparisons(results: list[dict], y_test: pd.Series) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        RocCurveDisplay.from_predictions(y_test, r["test_proba"], name=r["spec"].name, ax=ax)
    ax.set_title("ROC - test cohort (time-separated holdout)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        PrecisionRecallDisplay.from_predictions(y_test, r["test_proba"], name=r["spec"].name, ax=ax)
    ax.set_title("Precision-Recall - test cohort (time-separated holdout)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        cal = r["metadata"]["test"]["calibration"]
        ax.plot(cal["mean_predicted"], cal["fraction_positive"], marker="o", label=r["spec"].name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title("Calibration - test cohort (time-separated holdout)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    train_df, test_df = df[df["split"] == "train"], df[df["split"] == "test"]

    results = []
    for spec in build_specs():
        print(f"Training {spec.name} ({spec.feature_set} features)...")
        result = train_one(spec, train_df, test_df)
        results.append(result)
        m = result["metadata"]
        print(
            f"  CV ROC-AUC {m['cv']['roc_auc_mean']:.4f}+/-{m['cv']['roc_auc_std']:.4f} | "
            f"test ROC-AUC {m['test']['roc_auc']:.4f} | "
            f"test PR-AUC {m['test']['average_precision']:.4f} | "
            f"test Brier {m['test']['brier_score']:.4f} | "
            f"threshold {m['threshold']:.2f}"
        )

    full_gap_lr = next(r for r in results if r["spec"].name == "logistic_regression_full")["metadata"]["test"]["roc_auc"] - \
        next(r for r in results if r["spec"].name == "logistic_regression_clean")["metadata"]["test"]["roc_auc"]
    full_gap_hgb = next(r for r in results if r["spec"].name == "hist_gradient_boosting_full")["metadata"]["test"]["roc_auc"] - \
        next(r for r in results if r["spec"].name == "hist_gradient_boosting_clean")["metadata"]["test"]["roc_auc"]
    print(f"Clean-vs-full AUC gap (logistic regression): {full_gap_lr:.4f}")
    print(f"Clean-vs-full AUC gap (HistGradientBoosting): {full_gap_hgb:.4f}")

    plot_comparisons(results, test_df["label"])
    print(f"Wrote comparison figures to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
