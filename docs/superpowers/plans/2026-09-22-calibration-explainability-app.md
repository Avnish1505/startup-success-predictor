# Calibration, Explainability, Percentile & App Rewire Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Step-2 `hist_gradient_boosting_full` model's raw `predict_proba` into a trustworthy, explainable, user-facing number: calibrate it so "73%" means ~73/100, replace the fake if-statement "explanations" in `app.py` with real SHAP contributions, replace the probability-relabeled-as-percentile line with a real empirical percentile, and fully rewire `app.py`'s Predictor tab onto this real pipeline (confirmed with the user: the old 4-feature toy model in `predictor.py`/`startup_model.pkl` was never connected to the Step 1/2 pipeline — this plan replaces that wiring in `app.py` specifically; `api.py`/`predictor.py`/`analytics.py` are out of scope and left as-is).

**Architecture:** `src/models/calibrate.py` re-splits the train cohort (never touching test) into a `train_fit` portion and a held-out `calibration` portion, retrains a fresh base `HistGradientBoostingClassifier` pipeline on `train_fit` only (the Step 2 `hist_gradient_boosting_full.joblib` was fit on *all* of train, so it can't be reused for leak-free calibration), fits `CalibratedClassifierCV` with `sklearn.frozen.FrozenEstimator` on the calibration split, and evaluates both the raw base model and the calibrated model once on the untouched test cohort. `src/models/explain.py` wraps a cached `shap.TreeExplainer` over the base (pre-calibration) tree model — SHAP needs the raw tree, not the isotonic/sigmoid wrapper — and maps signed contributions back to human feature names (clean 1:1 mapping since categoricals are ordinal-encoded, not one-hot). `src/models/percentile.py` holds pure empirical-CDF functions plus a serialized reference array of calibrated train-cohort scores. `app.py`'s Predictor tab is rewired to collect the 8 real features, call the calibrated pipeline, and render the real percentile + real SHAP reasons.

**Tech Stack:** scikit-learn (`CalibratedClassifierCV`, `sklearn.frozen.FrozenEstimator`, `train_test_split`), `shap` (`TreeExplainer`), matplotlib, numpy, joblib, pytest, streamlit.

**Spec:** user-supplied spec (see conversation) — CalibratedClassifierCV (isotonic if large enough, sigmoid otherwise) fit on a held-out calibration split never touching test, reliability diagrams before/after with Brier scores; `shap.TreeExplainer` cached at load time exposing top-N signed contributions; a real empirical-CDF percentile serialized next to the model, replacing the mislabeled `app.py` line; deletion of the fake if-statement "explanations."

## Verified real state (do not re-derive)

- `shap` 0.52.0 and `sklearn.frozen.FrozenEstimator` are installed/available (checked this session). `shap.TreeExplainer` works directly on a bare `HistGradientBoostingClassifier` (confirmed: `shap_values` on 50 rows through the Step 2 `hist_gradient_boosting_full` pipeline's preprocessor returned shape `(50, 8)`, matching `schema.FULL_NUMERIC_FEATURES + schema.FULL_CATEGORICAL_FEATURES` in that exact order — the `ColumnTransformer` puts `num` before `cat`).
- Because Step 2's `hist_gradient_boosting_full` used `OrdinalEncoder` (not one-hot) for categoricals, each of the 8 transformed columns maps 1:1 to one original feature name — no aggregation needed to make SHAP values human-readable.
- `app.py`'s Predictor tab currently: takes `funding, team_size, experience, market`; calls `predictor.predict_startup` (loads `startup_model.pkl`, the old 6-row-lookup-table model from the now-deleted `train_model.py`); has the two named fake-explanation `if` lines; has the `f"...performs better than {int(prob)}% of similar startups"` line using the raw probability as if it were a percentile. `api.py` independently loads `startup_model.pkl` directly and is **not** touched by this plan (out of scope, confirmed with user — flag it as still-legacy in the final report, don't silently leave the user thinking it was updated).
- Train cohort = 10,672 rows, test cohort = 2,662 rows (from `data/processed/startups_features_v1.parquet`, `split` column). An 80/20 stratified split of train for calibration gives roughly 8,538 `train_fit` / 2,134 `calibration` rows — comfortably over a reasonable isotonic-regression sample-size floor.

## Global Constraints

- The calibration split must never overlap with the test cohort, and must never have been seen by the base model during its own fit — so the base tree model for this plan is **retrained from scratch on `train_fit` only**, not reused from Step 2's `hist_gradient_boosting_full.joblib` (which saw all of train).
- `choose_calibration_method(n_samples, threshold=1000)` is a real, tested decision function (isotonic if `n_samples >= threshold`, else sigmoid) — not a hardcoded choice — even though our actual calibration split (~2,134 rows) will land on isotonic.
- Reliability diagrams and Brier scores are computed on the untouched **test** cohort, before (raw base model) and after (calibrated model), as two separate figures: `reports/figures/calibration_before.png` and `reports/figures/calibration_after.png`.
- `shap.TreeExplainer` is built once (module-level cache, e.g. `functools.lru_cache` or an explicit singleton) and reused across calls — never rebuilt per request/prediction.
- The empirical percentile is computed against **calibrated** scores on the training cohort (the same quantity displayed to users as "%"), not raw base-model scores.
- Production artifacts the app needs at runtime (calibrated pipeline, percentile reference array, category dropdown option lists) must be **committed to git**, unlike the Step 2 comparison models — the live Streamlit deployment can't regenerate Kaggle-downloaded, credential-gated data at deploy time. These live under `models/production/`, carved out of the existing `/models/` gitignore rule with a negation pattern.
- `app.py` deletions are exactly the two named blocks (fake `if funding > 150000` / `if experience > 2` reasons, and the `f"...performs better than {int(prob)}%..."` line) plus whatever input/model-call code must change to support the real 8-feature model — not a rewrite of the Analytics or AI Advisor tabs, which are untouched.
- `predictor.py`, `startup_model.pkl`, `api.py`, `analytics.py`, `advisor_ai.py` are untouched by this plan.
- Every new pure function (percentile math, calibration-method choice, SHAP contribution formatting) gets a unit test with synthetic inputs, no network/Kaggle dependency, matching the precedent in `tests/test_data.py` and `tests/test_evaluate.py`.

---

## Task 1: `src/models/percentile.py` — empirical CDF (TDD)

**Files:**
- Create: `tests/test_percentile.py`
- Create: `src/models/percentile.py`

**Interfaces:**
- Produces: `build_reference_distribution(scores) -> np.ndarray` (sorted copy), `compute_percentile(score, sorted_reference) -> float` (0-100), `save_reference_distribution(sorted_reference, path)`, `load_reference_distribution(path) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_percentile.py
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
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `python3 -m pytest tests/test_percentile.py -v`
Expected: `ModuleNotFoundError: No module named 'src.models.percentile'`

- [ ] **Step 3: Write `src/models/percentile.py`**

```python
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
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `python3 -m pytest tests/test_percentile.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/percentile.py tests/test_percentile.py
git commit -m "feat(models): add empirical percentile module"
```

---

## Task 2: `src/models/calibrate.py` — leak-free calibration + reliability diagrams

**Files:**
- Create: `tests/test_calibrate.py`
- Create: `src/models/calibrate.py`

**Interfaces:**
- Consumes: `src.data.schema` (`FULL_NUMERIC_FEATURES`, `FULL_CATEGORICAL_FEATURES`), `src.models.train.build_pipeline`/`ModelSpec` (reuse, don't duplicate the pipeline-building logic), `src.models.evaluate` (`compute_metrics`, `calibration_curve_data`)
- Produces: `choose_calibration_method(n_samples, threshold=1000) -> str`, `split_train_for_calibration(train_df, calibration_frac=0.2, random_state=42) -> tuple[pd.DataFrame, pd.DataFrame]`, `main()` writing `models/production/hist_gradient_boosting_full_calibrated.joblib`, `models/production/hist_gradient_boosting_full_calibrated_metadata.json`, `models/production/train_score_distribution.npy`, `reports/figures/calibration_before.png`, `reports/figures/calibration_after.png`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calibrate.py
import numpy as np
import pandas as pd

from src.models.calibrate import choose_calibration_method, split_train_for_calibration


def test_choose_calibration_method_isotonic_when_large():
    assert choose_calibration_method(5000, threshold=1000) == "isotonic"


def test_choose_calibration_method_sigmoid_when_small():
    assert choose_calibration_method(200, threshold=1000) == "sigmoid"


def test_choose_calibration_method_boundary_is_isotonic():
    assert choose_calibration_method(1000, threshold=1000) == "isotonic"


def test_split_train_for_calibration_disjoint_and_stratified():
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({
        "label": rng.integers(0, 2, size=n),
        "x": rng.normal(size=n),
    })
    train_fit, calibration = split_train_for_calibration(df, calibration_frac=0.2, random_state=42)
    assert len(train_fit) + len(calibration) == n
    assert set(train_fit.index).isdisjoint(set(calibration.index))
    assert abs(len(calibration) / n - 0.2) < 0.02
    # stratification keeps positive rate close between the two splits
    assert abs(train_fit["label"].mean() - calibration["label"].mean()) < 0.05
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `python3 -m pytest tests/test_calibrate.py -v`
Expected: `ModuleNotFoundError: No module named 'src.models.calibrate'`

- [ ] **Step 3: Write `src/models/calibrate.py`**

```python
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
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `python3 -m pytest tests/test_calibrate.py -v`
Expected: all PASS

- [ ] **Step 5: Run the real calibration pipeline**

Run: `python3 -m src.models.calibrate`
Expected: prints split sizes, chosen method (`isotonic`, given ~2,134 calibration rows), both Brier scores (after should be <= before or very close - if not, note it honestly, don't force it), writes 5 files under `models/production/` and 2 PNGs under `reports/figures/`.

- [ ] **Step 6: Commit**

```bash
git add src/models/calibrate.py tests/test_calibrate.py
git commit -m "feat(models): add leak-free CalibratedClassifierCV with before/after reliability diagrams"
```

---

## Task 3: `src/models/explain.py` — cached SHAP explainer

**Files:**
- Create: `tests/test_explain.py`
- Create: `src/models/explain.py`

**Interfaces:**
- Consumes: `models/production/hist_gradient_boosting_full_calibrated_base.joblib` (the pre-calibration tree pipeline - SHAP explains the tree, not the isotonic/sigmoid wrapper)
- Produces: `get_explainer(pipeline) -> shap.TreeExplainer` (cached - same pipeline object in, same explainer out, no rebuild), `explain_prediction(pipeline, feature_names, input_df) -> list[dict]` (each dict: `feature`, `value`, `shap_value`, `direction`), sorted by `abs(shap_value)` descending.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_explain.py
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.models.explain import explain_prediction, get_explainer


def _toy_pipeline():
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.uniform(0, 10, n)  # strongly, monotonically predictive
    x2 = rng.uniform(0, 10, n)  # noise
    y = (x1 + rng.normal(0, 0.5, n) > 5).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "cat": rng.choice(["a", "b"], n), "label": y})

    preprocessor = ColumnTransformer([
        ("num", "passthrough", ["x1", "x2"]),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["cat"]),
    ])
    pipeline = Pipeline([("preprocess", preprocessor), ("clf", HistGradientBoostingClassifier(random_state=0))])
    pipeline.fit(df[["x1", "x2", "cat"]], df["label"])
    return pipeline


def test_get_explainer_is_cached():
    pipeline = _toy_pipeline()
    e1 = get_explainer(pipeline)
    e2 = get_explainer(pipeline)
    assert e1 is e2


def test_explain_prediction_returns_sorted_signed_contributions():
    pipeline = _toy_pipeline()
    row = pd.DataFrame([{"x1": 8.0, "x2": 5.0, "cat": "a"}])
    result = explain_prediction(pipeline, ["x1", "x2", "cat"], row, top_n=3)
    assert len(result) == 3
    mags = [abs(r["shap_value"]) for r in result]
    assert mags == sorted(mags, reverse=True)
    assert {"feature", "value", "shap_value", "direction"} <= result[0].keys()


def test_explain_prediction_direction_matches_expected_relationship():
    # x1 drives the label positively - a high x1 should push the SHAP bar toward "increases"
    pipeline = _toy_pipeline()
    high = explain_prediction(pipeline, ["x1", "x2", "cat"], pd.DataFrame([{"x1": 9.5, "x2": 5.0, "cat": "a"}]), top_n=3)
    low = explain_prediction(pipeline, ["x1", "x2", "cat"], pd.DataFrame([{"x1": 0.5, "x2": 5.0, "cat": "a"}]), top_n=3)
    high_x1 = next(r for r in high if r["feature"] == "x1")
    low_x1 = next(r for r in low if r["feature"] == "x1")
    assert high_x1["shap_value"] > low_x1["shap_value"]
    assert high_x1["direction"] == "increases"
    assert low_x1["direction"] == "decreases"
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `python3 -m pytest tests/test_explain.py -v`
Expected: `ModuleNotFoundError: No module named 'src.models.explain'`

- [ ] **Step 3: Write `src/models/explain.py`**

```python
"""Cached SHAP TreeExplainer over the boosted model, exposing top-N signed
contributions for a single prediction in human-readable feature names."""
from __future__ import annotations

from functools import lru_cache

import shap
from sklearn.pipeline import Pipeline


@lru_cache(maxsize=4)
def _cached_explainer(pipeline_id: int, pipeline: Pipeline) -> shap.TreeExplainer:
    return shap.TreeExplainer(pipeline.named_steps["clf"])


def get_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    """Build (once) or reuse a TreeExplainer for this exact pipeline object."""
    return _cached_explainer(id(pipeline), pipeline)


def explain_prediction(pipeline: Pipeline, feature_names: list[str], input_df, top_n: int = 5) -> list[dict]:
    explainer = get_explainer(pipeline)
    transformed = pipeline.named_steps["preprocess"].transform(input_df)
    shap_values = explainer.shap_values(transformed)
    if shap_values.ndim == 3:  # some SHAP/sklearn version combos add a class axis
        shap_values = shap_values[:, :, 1]

    row_values = shap_values[0]
    raw_row = input_df.iloc[0]

    contributions = []
    for name, sv in zip(feature_names, row_values):
        contributions.append({
            "feature": name,
            "value": raw_row[name],
            "shap_value": float(sv),
            "direction": "increases" if sv > 0 else "decreases",
        })
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    return contributions[:top_n]
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `python3 -m pytest tests/test_explain.py -v`
Expected: all PASS. If `_cached_explainer` errors on hashing the `Pipeline` (unhashable), drop `pipeline` from the `lru_cache` key by caching on `id(pipeline)` alone via a plain dict instead of `lru_cache` - adjust implementation, re-run, confirm PASS before moving on (don't leave this ambiguous, resolve it in this step).

- [ ] **Step 5: Verify against the real production model + one directional sanity check**

Run:
```bash
python3 -c "
import joblib, pandas as pd
from src.models.explain import explain_prediction
from src.data import schema

pipeline = joblib.load('models/production/hist_gradient_boosting_full_calibrated_base.joblib')
features = schema.FULL_NUMERIC_FEATURES + schema.FULL_CATEGORICAL_FEATURES
base = {'founded_year': 2013.0, 'time_to_first_funding_days': 180.0, 'funding_total_usd_log1p': 12.0,
        'funding_rounds': 2.0, 'funding_span_days': 365.0, 'country_code': 'USA', 'region': 'SF Bay',
        'primary_category': 'Software'}
low = dict(base, funding_total_usd_log1p=8.0)
high = dict(base, funding_total_usd_log1p=16.0)
low_r = explain_prediction(pipeline, features, pd.DataFrame([low]), top_n=8)
high_r = explain_prediction(pipeline, features, pd.DataFrame([high]), top_n=8)
low_v = next(r for r in low_r if r['feature']=='funding_total_usd_log1p')['shap_value']
high_v = next(r for r in high_r if r['feature']=='funding_total_usd_log1p')['shap_value']
print('low funding shap:', low_v, 'high funding shap:', high_v)
assert high_v > low_v, 'more funding should push the SHAP bar toward success'
print('OK: direction matches domain expectation')
"
```
Expected: `OK: direction matches domain expectation` (more raised funding should push the contribution up, matching the positive correlation already established in `DATA_CARD.md`/`MODEL_CARD.md`). If it fails, investigate before proceeding - don't paper over a real directional inversion.

- [ ] **Step 6: Commit**

```bash
git add src/models/explain.py tests/test_explain.py
git commit -m "feat(models): add cached SHAP explainer for signed feature contributions"
```

---

## Task 4: Rewire `app.py`'s Predictor tab onto the real, calibrated model

**Files:**
- Modify: `app.py`
- Create: `models/production/category_options.json` (generated by a short one-off script, then committed - not regenerated on every run, so no new module needed for this)

**Interfaces:**
- Consumes: `models/production/hist_gradient_boosting_full_calibrated.joblib` (calibrated, for the displayed probability), `models/production/hist_gradient_boosting_full_calibrated_base.joblib` (base, for SHAP), `models/production/train_score_distribution.npy` (for percentile), `src.models.percentile.compute_percentile`, `src.models.explain.explain_prediction`

- [ ] **Step 1: Generate `models/production/category_options.json`**

Run:
```bash
python3 -c "
import json
import pandas as pd

df = pd.read_parquet('data/processed/startups_features_v1.parquet')
train_df = df[df['split'] == 'train']
options = {
    'country_code': sorted(train_df['country_code'].dropna().unique().tolist()),
    'region': sorted(train_df['region'].dropna().unique().tolist()),
    'primary_category': sorted(train_df['primary_category'].dropna().unique().tolist()),
}
with open('models/production/category_options.json', 'w') as f:
    json.dump(options, f, indent=2)
print({k: len(v) for k, v in options.items()})
"
```
Expected: prints counts close to 84/595/506 (see the data-pipeline plan's verified facts); writes the JSON file.

- [ ] **Step 2: Replace the Predictor tab's inputs and prediction call in `app.py`**

Replace the `import` line `from predictor import predict_startup` with:

```python
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import schema
from src.models.explain import explain_prediction
from src.models.percentile import compute_percentile, load_reference_distribution

PRODUCTION_DIR = Path("models/production")
FEATURE_COLS = schema.FULL_NUMERIC_FEATURES + schema.FULL_CATEGORICAL_FEATURES


@st.cache_resource
def load_production_artifacts():
    calibrated = joblib.load(PRODUCTION_DIR / "hist_gradient_boosting_full_calibrated.joblib")
    base = joblib.load(PRODUCTION_DIR / "hist_gradient_boosting_full_calibrated_base.joblib")
    reference = load_reference_distribution(PRODUCTION_DIR / "train_score_distribution.npy")
    options = json.loads((PRODUCTION_DIR / "category_options.json").read_text())
    return calibrated, base, reference, options
```

Replace the Predictor tab's input widgets (the `col1, col2 = st.columns(2)` block through the `market = st.selectbox(...)` line) with:

```python
    calibrated_pipeline, base_pipeline, train_reference, category_options = load_production_artifacts()

    col1, col2 = st.columns(2)
    with col1:
        founded_date = st.date_input("Founded date", value=date(2013, 1, 1))
        first_funding_date = st.date_input("First funding date", value=date(2013, 6, 1))
        last_funding_date = st.date_input("Most recent funding date", value=date(2014, 1, 1))
        funding_total_usd = st.number_input("Total funding raised ($)", min_value=0, value=1_000_000, step=10_000)
        funding_rounds = st.number_input("Number of funding rounds", min_value=1, value=2, step=1)
    with col2:
        country_code = st.selectbox("Country", category_options["country_code"] + ["UNKNOWN"],
                                     index=category_options["country_code"].index("USA") if "USA" in category_options["country_code"] else 0)
        region = st.selectbox("Region", category_options["region"] + ["UNKNOWN"])
        primary_category = st.selectbox("Primary category", category_options["primary_category"] + ["UNKNOWN"])

    input_row = pd.DataFrame([{
        "founded_year": float(founded_date.year),
        "time_to_first_funding_days": float((first_funding_date - founded_date).days),
        "funding_total_usd_log1p": float(np.log1p(funding_total_usd)),
        "funding_rounds": float(funding_rounds),
        "funding_span_days": float((last_funding_date - first_funding_date).days),
        "country_code": country_code,
        "region": region,
        "primary_category": primary_category,
    }])
```

Replace the `if st.button("Predict"):` block's body with:

```python
    if st.button("Predict"):
        try:
            prob = float(calibrated_pipeline.predict_proba(input_row[FEATURE_COLS])[0, 1]) * 100
            st.session_state.prediction_prob = prob
            st.session_state.prediction_input = input_row
        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")
```

- [ ] **Step 3: Replace the "Why this result?" fake reasons with real SHAP contributions**

Replace:
```python
            reasons = []
            if funding > 150000: reasons.append("💰 Strong funding boosts success chances")
            if experience > 2: reasons.append("👨‍💼 Experienced founders improve execution")
            if team_size > 5: reasons.append("👥 Larger team supports scaling")
            if market == 2: reasons.append("🌍 Large market increases opportunity")

            for r in reasons: st.write("- " + r)
```
with:
```python
            contributions = explain_prediction(
                base_pipeline, FEATURE_COLS, st.session_state.prediction_input, top_n=5
            )
            for c in contributions:
                arrow = "⬆️" if c["direction"] == "increases" else "⬇️"
                st.write(f"- {arrow} `{c['feature']}` = {c['value']} {c['direction']} predicted success (SHAP {c['shap_value']:+.3f})")
```

- [ ] **Step 4: Replace the mislabeled percentile line**

Replace:
```python
        st.info(f"Your startup performs better than {int(prob)}% of similar startups")
```
with:
```python
        percentile = compute_percentile(prob / 100.0, train_reference)
        st.info(f"Your predicted score is higher than {percentile:.1f}% of startups in the training cohort (calibrated empirical percentile, not the raw probability)")
```

- [ ] **Step 5: Fix the report/download block's stale variable references**

The existing `report_data = f"...Funding: ${funding}\nTeam: {team_size}\nExperience: {experience} yrs\nMarket: {market}"` line references variables that no longer exist after Step 2's replacement. Replace with:
```python
        report_data = (
            f"Startup Success Probability (calibrated): {prob:.2f}%\n"
            f"Percentile vs. training cohort: {percentile:.1f}%\n"
            f"Founded: {founded_date} | First funding: {first_funding_date} | Last funding: {last_funding_date}\n"
            f"Total funding: ${funding_total_usd} | Rounds: {funding_rounds}\n"
            f"Country: {country_code} | Region: {region} | Category: {primary_category}"
        )
```

- [ ] **Step 6: Start the app and verify it renders and predicts without error**

Run (background, then check logs / hit the local URL):
```bash
streamlit run app.py --server.headless true --server.port 8501 &
sleep 3
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501
```
Expected: `200`. Then use the `run` skill or `claude-in-chrome` (if available in this environment) to actually click "Predict" in the browser with default inputs and confirm: a probability renders, the percentile line no longer says "performs better than {prob}%" verbatim (should show a distinct percentile number), SHAP-based reasons render (not the old fake if-statement text), and no exception banner appears. Stop the background streamlit process when done (`kill %1` or find/kill by port).

- [ ] **Step 7: Commit**

```bash
git add -f models/production/category_options.json app.py
git commit -m "feat(app): rewire Predictor tab onto the calibrated model with real SHAP reasons and percentile"
```

Note: `models/production/category_options.json` needs `-f` (or the `.gitignore` negation from Task 2 already covers `models/production/` - if the negation pattern is in place from Task 2's commit, drop `-f`; verify with `git status` before committing which one is needed).

---

## Task 5: `.gitignore` negation for `models/production/`, README calibration section, final verification

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Consumes: real Brier scores printed by Task 2 Step 5 (`python -m src.models.calibrate` output) - README must quote the actual measured values, not placeholders.

- [ ] **Step 1: Update `.gitignore`**

Change:
```
# Generated model artifacts (regenerate with: python -m src.models.train)
/models/
```
to:
```
# Generated model artifacts (regenerate with: python -m src.models.train)
/models/*
!/models/production/
```

- [ ] **Step 2: Verify the negation works**

Run: `git status --porcelain=v1 | grep production`
Expected: shows `models/production/` files as untracked/addable (not silently ignored).

- [ ] **Step 3: Update `README.md`**

Add a new `## 🎯 Calibration & Explainability` section (after "🧠 Model Details & Performance", before "📸 Demo") with:
- One paragraph explaining raw `predict_proba` isn't calibrated and what `CalibratedClassifierCV` does about it, referencing the real chosen method (`isotonic`, given ~2,134 calibration rows) from Task 2 Step 5's output.
- Both figures embedded: `![Before calibration](reports/figures/calibration_before.png)` and `![After calibration](reports/figures/calibration_after.png)`, each captioned with its real Brier score from Task 2 Step 5's output (e.g. "Brier before: X.XXXX, after: Y.YYYY").
- A short note that SHAP (`shap.TreeExplainer`) now drives the "Why this result?" section and the percentile line is a real empirical percentile against the training cohort, not the raw probability.

Also fix the now-stale parts of the existing README: the "🧠 Model Details & Performance" section (still describes the old 6-row RandomForest) should be updated to describe `hist_gradient_boosting_full_calibrated` (features, calibration method, and a pointer to `MODEL_CARD.md` for full metrics); "⚙️ Installation & Setup" step 4 (`python train_model.py`) should be replaced with the real pipeline commands (`python -m src.data.build`, `python -m src.models.train`, `python -m src.models.calibrate`); "📁 Project Structure" should list the new `src/data/`, `src/models/` modules.

- [ ] **Step 4: Run full verification**

Run: `python3 -m pytest -v`
Expected: all tests (Step 1-3's new tests plus every prior test) pass.

- [ ] **Step 5: Commit**

```bash
git add .gitignore README.md
git commit -m "docs: add calibration section to README, scope models/production/ into git"
```

---

## Self-review notes

- Spec coverage: CalibratedClassifierCV isotonic-vs-sigmoid + held-out split never touching test (Task 2), reliability diagrams before/after + Brier for both (Task 2), cached `shap.TreeExplainer` + top-N signed contributions (Task 3), real empirical percentile serialized next to the model + deletion of the mislabeled line (Task 1 + Task 4 Step 4), deletion of the fake `if` explanations (Task 4 Step 3), all deliverable files + tests for all three + both calibration figures (Tasks 1-2-3), both curves + Brier scores in README (Task 5), acceptance's directional SHAP check (Task 3 Step 5).
- No placeholders: all steps carry full code.
- Type consistency: `ModelSpec`/`build_pipeline`/`git_provenance` are imported from `src.models.train` rather than redefined, so Task 2 can't drift from Task 3 (Step 2 plan)'s definitions; `FEATURE_COLS` in `app.py` matches the exact `numeric_features + categorical_features` order `explain_prediction` and the saved pipelines expect.
- Risk flagged explicitly rather than silently handled: Step 3's `lru_cache` on an sklearn `Pipeline` object may fail to hash - the plan calls this out and requires resolving it before moving on, not working around it silently.
