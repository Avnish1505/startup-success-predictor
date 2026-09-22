# Model Training & Evaluation Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the training/evaluation layer on top of the real, leakage-aware Crunchbase dataset from `data/processed/startups_features_v1.parquet` (built in the prior data-pipeline plan): a dummy baseline, a logistic regression, a HistGradientBoostingClassifier, each evaluated honestly (no accuracy headline) with repeated CV + a single time-separated holdout, and an explicit clean-vs-full leakage comparison for both non-dummy model families.

**Architecture:** `src/models/train.py` builds sklearn `Pipeline`s (ColumnTransformer fit on train only), runs `RepeatedStratifiedKFold` CV for the metrics distribution and a separate single `StratifiedKFold` + `cross_val_predict` pass for out-of-fold threshold selection, fits the final pipeline on the full train cohort, evaluates once on the held-out time-separated test cohort, and saves model + metadata + figures + metrics JSON. `src/models/evaluate.py` holds the pure, reusable metric/threshold/calibration functions (unit-tested, no I/O) plus a CLI to re-score a saved model. `notebooks/01_baselines.ipynb` is a real, executed walkthrough built via `nbformat`/`nbclient` reusing the same functions. `MODEL_CARD.md` reports the real measured numbers.

**Tech Stack:** scikit-learn (DummyClassifier, LogisticRegression, HistGradientBoostingClassifier, ColumnTransformer, RepeatedStratifiedKFold/StratifiedKFold, cross_validate/cross_val_predict), matplotlib, joblib, nbformat/nbclient, pytest.

**Spec:** user-supplied spec (see conversation) — train dummy/LR/HGB, clean-vs-full comparison, ROC-AUC/PR-AUC/Brier/confusion-matrix-at-a-justified-threshold for every model, repeated CV + single holdout with both numbers reported and explained, no accuracy headline.

## Real data already on disk (verified, do not re-derive)

`data/processed/startups_features_v1.parquet`: 13,334 rows, columns include `founded_year, time_to_first_funding_days, country_code, region, primary_category, category_list, funding_total_usd_log1p, funding_rounds, funding_span_days, label, permalink, name, split`. `split == 'train'`: 10,672 rows, positive rate 59.83%. `split == 'test'`: 2,662 rows, positive rate 26.71%. `label`: 1=acquired/ipo, 0=closed. Feature lists live in `src/data/schema.py`: `CLEAN_NUMERIC_FEATURES = ['founded_year','time_to_first_funding_days']`, `CLEAN_CATEGORICAL_FEATURES = ['country_code','region','primary_category']`, `FULL_NUMERIC_FEATURES = CLEAN_NUMERIC_FEATURES + ['funding_total_usd_log1p','funding_rounds','funding_span_days']`, `FULL_CATEGORICAL_FEATURES = CLEAN_CATEGORICAL_FEATURES` (unchanged).

Installed this session: matplotlib, nbformat, nbclient, ipykernel, nbconvert (verified importable). sklearn 1.9.1. Current git SHA at plan time: `100385d`.

## Global Constraints

- Every preprocessing step (imputer, scaler, encoder) lives inside an sklearn `Pipeline`/`ColumnTransformer` fit on the train split only — never fit on test.
- Report ROC-AUC, average precision (PR-AUC), Brier score, and a confusion matrix at a threshold chosen via out-of-fold F0.5 maximization (justification: in an early screening use case, a false "will succeed" misleads a founder/investor more than a missed "might succeed," so precision is weighted higher than recall — F0.5, not F1). Never report accuracy as a headline metric.
- Report both repeated-CV metrics (mean ± std over `RepeatedStratifiedKFold(n_splits=5, n_repeats=10)` on the train cohort) and a single held-out evaluation on the time-separated test cohort, and explicitly explain why they differ (test cohort is a later, harder era with a 26.71% positive rate vs train's 59.83% — CV measures same-era generalization, the holdout measures forward-in-time generalization, which is the realistic deployment scenario).
- Train both `LogisticRegression` and `HistGradientBoostingClassifier` on both the clean and full feature sets (4 real model fits + 1 dummy baseline = 5 total), so the clean-vs-full AUC gap is measured for both a linear model and a tree ensemble, not asserted from one probe.
- `LogisticRegression` categorical features go through one-hot encoding (linear-model-appropriate); `HistGradientBoostingClassifier` categorical features go through ordinal encoding with `handle_unknown='use_encoded_value', unknown_value=-1` (tree-appropriate, avoids one-hot blowup and sidesteps sklearn-version-dependent native categorical APIs). Numeric features get median imputation everywhere; `LogisticRegression` additionally gets `StandardScaler` (trees don't need it).
- `funding_total_usd_log1p` (already log1p-transformed in the processed parquet) satisfies the "log-transformed funding" requirement for `LogisticRegression` — do not re-derive from raw `funding_total_usd`.
- Do not artificially rebalance classes (no `class_weight='balanced'`, no resampling) — the spec wants honest metrics on the natural distribution, not inflated ones.
- Saved model metadata must include: feature names (numeric + categorical, by name), split boundary date (read from `data/processed` build, i.e. `2011-12-01`), sklearn version, git SHA (+ dirty flag), trained-at timestamp, feature set (`clean`/`full`), chosen threshold, CV metrics, test metrics.
- `models/` (joblib + per-model metadata JSON) is a regenerable build artifact → gitignored, like `data/processed/`. `reports/figures/*.png`, `reports/metrics/*.json`, `notebooks/01_baselines.ipynb`, and `MODEL_CARD.md` are deliverables → committed.
- Pure, deterministic helper functions (metric computation, threshold selection, calibration curve data) live in `src/models/evaluate.py` and are unit tested in `tests/test_evaluate.py` with small synthetic inputs — no model fitting or I/O in the tests.

---

## Task 1: `src/models/evaluate.py` — pure metrics/threshold/calibration functions (TDD)

**Files:**
- Create: `src/models/__init__.py` (empty)
- Create: `tests/test_evaluate.py`
- Create: `src/models/evaluate.py`

**Interfaces:**
- Produces: `compute_metrics(y_true, y_proba, threshold) -> dict`, `choose_threshold(y_true, y_proba, beta=0.5) -> tuple[float, float]` (threshold, achieved f-beta), `calibration_curve_data(y_true, y_proba, n_bins=10) -> dict`, `evaluate_split(y_true, y_proba, threshold) -> dict` (wraps `compute_metrics` + `calibration_curve_data` into one payload ready for JSON).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.evaluate'`

- [ ] **Step 3: Write `src/models/evaluate.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_evaluate.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/__init__.py src/models/evaluate.py tests/test_evaluate.py
git commit -m "feat(models): add pure metrics/threshold/calibration helpers"
```

---

## Task 2: `evaluate.py` CLI for re-scoring a saved model

**Files:**
- Modify: `src/models/evaluate.py` (append CLI, no changes to Task 1 functions)

**Interfaces:**
- Consumes: a joblib model file + `data/processed/startups_features_v1.parquet` + the model's saved metadata JSON (for its feature list, threshold, and feature-set name)
- Produces: `reports/metrics/<model_name>_metrics.json` (re-computed `evaluate_split` payload) when run as `python -m src.models.evaluate <model_name>`

- [ ] **Step 1: Append the CLI to `src/models/evaluate.py`**

```python
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
```

- [ ] **Step 2: Verify import still works (real run happens after Task 3 produces a saved model)**

Run: `python3 -c "from src.models.evaluate import main; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/models/evaluate.py
git commit -m "feat(models): add CLI to re-score a saved model to metrics JSON"
```

---

## Task 3: `src/models/train.py` — train all 5 models, save artifacts, generate figures

**Files:**
- Create: `src/models/train.py`

**Interfaces:**
- Consumes: `src.data.schema` (feature lists), `src.models.evaluate` (`choose_threshold`, `evaluate_split`)
- Produces: `models/<name>.joblib`, `models/<name>_metadata.json` for each of `dummy_most_frequent`, `logistic_regression_full`, `logistic_regression_clean`, `hist_gradient_boosting_full`, `hist_gradient_boosting_clean`; `reports/figures/roc_comparison.png`, `reports/figures/pr_comparison.png`, `reports/figures/calibration_comparison.png`; console report; returns nothing (script entry point via `main()`).

- [ ] **Step 1: Write `src/models/train.py`**

```python
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
```

- [ ] **Step 2: Run the training script end-to-end**

Run: `python3 -m src.models.train`
Expected: prints CV + test metrics for all 5 models, prints both clean-vs-full AUC gaps, writes 10 files to `models/` and 3 PNGs to `reports/figures/`.

- [ ] **Step 3: Sanity-check the dummy baseline is truly trivial**

Run: `python3 -c "import json; print(json.load(open('models/dummy_most_frequent_metadata.json'))['test'])"`
Expected: `roc_auc` at or near 0.5, `average_precision` near the test positive rate (0.2671), confusion matrix has 0 in either the tp+fp column or tn+fn column (constant prediction).

- [ ] **Step 4: Generate the metrics JSON exports via the evaluate.py CLI for every model**

Run:
```bash
for m in dummy_most_frequent logistic_regression_full logistic_regression_clean hist_gradient_boosting_full hist_gradient_boosting_clean; do
  python3 -m src.models.evaluate "$m"
done
```
Expected: 5 files written under `reports/metrics/`.

- [ ] **Step 5: Commit**

```bash
git add src/models/train.py
git commit -m "feat(models): train dummy/logreg/HGB baselines with clean-vs-full comparison"
```

---

## Task 4: `notebooks/01_baselines.ipynb`

**Files:**
- Create: `notebooks/01_baselines.ipynb`
- Create: `scripts/build_baselines_notebook.py` (throwaway generator script — not a deliverable itself, just how the notebook gets built; fine to keep for reproducibility since it makes the notebook re-buildable without hand-editing JSON)

**Interfaces:**
- Consumes: `src.models.train.build_specs/train_one` is NOT reused directly (that would re-run 5 full CV cycles inside a notebook, slow); instead the notebook loads the already-saved `models/*.joblib` + `models/*_metadata.json` artifacts from Task 3 and the processed parquet, and reuses `src.models.evaluate.evaluate_split` for any recomputation it needs.

- [ ] **Step 1: Write `scripts/build_baselines_notebook.py`**

```python
"""Generate notebooks/01_baselines.ipynb, then execute it in place so it
carries real outputs (not just unexecuted code)."""
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Baseline models on the Crunchbase startup outcome dataset\n\n"
    "Loads the models trained by `python -m src.models.train` and reports "
    "their saved CV and held-out test metrics side by side. Run "
    "`python -m src.models.train` first if `models/` is empty."
))

cells.append(nbf.v4.new_code_cell(
    "import json\n"
    "from pathlib import Path\n\n"
    "import pandas as pd\n\n"
    "MODEL_NAMES = [\n"
    "    'dummy_most_frequent',\n"
    "    'logistic_regression_full',\n"
    "    'logistic_regression_clean',\n"
    "    'hist_gradient_boosting_full',\n"
    "    'hist_gradient_boosting_clean',\n"
    "]\n\n"
    "metadata = {name: json.loads(Path(f'../models/{name}_metadata.json').read_text()) for name in MODEL_NAMES}\n"
    "len(metadata)"
))

cells.append(nbf.v4.new_markdown_cell("## Metrics table (test = time-separated holdout, CV = repeated stratified CV on train)"))

cells.append(nbf.v4.new_code_cell(
    "rows = []\n"
    "for name, m in metadata.items():\n"
    "    rows.append({\n"
    "        'model': name,\n"
    "        'feature_set': m['feature_set'],\n"
    "        'cv_roc_auc_mean': m['cv']['roc_auc_mean'],\n"
    "        'cv_roc_auc_std': m['cv']['roc_auc_std'],\n"
    "        'test_roc_auc': m['test']['roc_auc'],\n"
    "        'test_average_precision': m['test']['average_precision'],\n"
    "        'test_brier': m['test']['brier_score'],\n"
    "        'threshold': m['threshold'],\n"
    "        'test_f0.5': m['test']['f0.5'],\n"
    "    })\n"
    "table = pd.DataFrame(rows).set_index('model')\n"
    "table"
))

cells.append(nbf.v4.new_markdown_cell(
    "## CV vs. test: why they differ\n\n"
    "CV metrics are computed on repeated stratified folds *within* the train cohort "
    "(same era, positive rate 59.83%). The test metric is a single evaluation on the "
    "time-separated holdout (a later era, positive rate 26.71%). The gap between the "
    "two numbers reflects genuine distribution shift across time, not just sampling noise - "
    "this is the realistic deployment scenario (scoring newer companies), so the test number "
    "is the one that matters for expectations, not the CV number."
))

cells.append(nbf.v4.new_code_cell(
    "table[['cv_roc_auc_mean', 'test_roc_auc']].assign(\n"
    "    gap=lambda d: d['test_roc_auc'] - d['cv_roc_auc_mean']\n"
    ")"
))

cells.append(nbf.v4.new_markdown_cell("## Clean vs. full: the leakage finding"))

cells.append(nbf.v4.new_code_cell(
    "for family in ['logistic_regression', 'hist_gradient_boosting']:\n"
    "    full_auc = metadata[f'{family}_full']['test']['roc_auc']\n"
    "    clean_auc = metadata[f'{family}_clean']['test']['roc_auc']\n"
    "    print(f'{family}: full={full_auc:.4f} clean={clean_auc:.4f} gap={full_auc - clean_auc:.4f}')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Figures\n\n"
    "See `../reports/figures/roc_comparison.png`, `pr_comparison.png`, "
    "`calibration_comparison.png` for the full ROC/PR/calibration curves across all 5 models."
))

nb["cells"] = cells

Path_out = "notebooks/01_baselines.ipynb"
with open(Path_out, "w") as f:
    nbf.write(nb, f)

client = NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": "notebooks"}})
client.execute()
with open(Path_out, "w") as f:
    nbf.write(nb, f)
print(f"Wrote and executed {Path_out}")
```

- [ ] **Step 2: Run it**

Run: `mkdir -p notebooks && python3 scripts/build_baselines_notebook.py`
Expected: `Wrote and executed notebooks/01_baselines.ipynb`, no exceptions (an exception here means a code cell raised - inspect and fix before proceeding).

- [ ] **Step 3: Spot-check the notebook actually has outputs, not just source**

Run: `python3 -c "import nbformat; nb = nbformat.read('notebooks/01_baselines.ipynb', as_version=4); print(sum(1 for c in nb.cells if c.cell_type=='code' and c.get('outputs')))"`
Expected: a number > 0 (at least the metrics-table and gap cells produced output).

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_baselines.ipynb scripts/build_baselines_notebook.py
git commit -m "docs(models): add executed baselines notebook"
```

---

## Task 5: `MODEL_CARD.md` with real numbers

**Files:**
- Create: `MODEL_CARD.md`

**Interfaces:**
- Consumes: the real printed output of Task 3 Step 2 and the `models/*_metadata.json` files — MODEL_CARD.md must be filled in with those actual numbers, not placeholders, and updated if a rerun changes them (CV uses `random_state=42` throughout so reruns should be stable, but always read the metadata files rather than trusting memory).

- [ ] **Step 1: Read every metadata JSON to pull exact numbers**

Run: `python3 -c "
import json
from pathlib import Path
for p in sorted(Path('models').glob('*_metadata.json')):
    m = json.loads(p.read_text())
    print(m['model_name'], m['feature_set'], m['cv']['roc_auc_mean'], m['cv']['roc_auc_std'], m['test']['roc_auc'], m['test']['average_precision'], m['test']['brier_score'], m['threshold'], m['test']['confusion_matrix'])
"`

- [ ] **Step 2: Write `MODEL_CARD.md`**

Content requirements, all values taken from Step 1's real output (do not invent):
- One paragraph stating headline metric is ROC-AUC/PR-AUC, not accuracy, and why (class imbalance shifts across train/test: 59.83% -> 26.71% positive).
- A metrics table: rows = 5 models, columns = feature_set, CV ROC-AUC (mean±std), test ROC-AUC, test PR-AUC, test Brier, threshold, test confusion matrix, test F0.5.
- A "CV vs. holdout" section explaining the gap in terms of the time-based split (same reasoning as the notebook's markdown cell).
- A "Clean vs. full: the leakage finding" section reporting both the logistic-regression gap and the HistGradientBoosting gap with the real numbers, explicitly stating whichever is true: if clean is much worse, say so plainly as the finding, don't soften it.
- Context against the spec's cited external benchmarks (ROC-AUC ~0.86 on 34k Crunchbase companies; F0.5=0.097 as "good" at a 0.78% positive rate) — note our dataset differs in size (13,334 labeled rows) and especially in test-cohort positive rate (26.71%, not 0.78%), so these numbers aren't directly comparable; state where our numbers land without claiming to replicate the cited study.
- A "What this model cannot do" section: cannot predict outcomes for companies that haven't resolved (operating rows were excluded from training entirely, so the model has never seen a "still going" label); cannot be trusted on data from well after the training window without re-validation (the CV-vs-holdout gap shows real drift even a few years out); the full-feature model cannot be used for live screening of an active company at all (its strongest features are only observable after the outcome, i.e. total funding raised and rounds count as of scrape time) - only the clean-feature model is honest for that use case, and it is measurably weaker; calibration is only as good as shown in `reports/figures/calibration_comparison.png`, don't treat raw probabilities as precise without checking that plot; the training data undercounts failures (see `DATA_CARD.md`'s survivorship-bias caveat), so even the clean model's positive-rate calibration likely runs optimistic in the real world.

- [ ] **Step 3: Commit**

```bash
git add MODEL_CARD.md
git commit -m "docs(models): add MODEL_CARD with measured metrics and leakage finding"
```

---

## Self-review notes

- Spec coverage: dummy/LR/HGB training order (Task 3), log-transformed funding + one-hot for LR (Task 3 `build_pipeline`/`build_specs`), clean-vs-full for "the same model" done for both LR and HGB (Task 3), all 4 metrics + justified threshold (Task 1 + Task 3), repeated CV + single holdout with both reported and explained (Task 3 `run_cv` + Task 5 write-up), no accuracy headline (Task 5 explicit framing, `compute_metrics` never surfaces raw accuracy), all 5 deliverable files (train.py Task 3, evaluate.py Tasks 1-2, notebook Task 4, MODEL_CARD.md Task 5, reports/figures/ Task 3), metrics-as-JSON (Task 2 CLI + Task 3 Step 4), acceptance's clean-vs-full gap in the metrics table (Task 5).
- No placeholders: all steps carry full code.
- Type consistency: `ModelSpec` fields used identically across `build_pipeline`, `build_specs`, `train_one`; `evaluate_split`'s return shape (`compute_metrics` dict + `calibration` key) is what Task 3's `plot_comparisons` and Task 5's MODEL_CARD both read from.
