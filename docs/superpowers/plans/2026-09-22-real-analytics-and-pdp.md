# Real Analytics, Partial Dependence, and State-Consistency Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every hand-typed number in `analytics.py` with a value computed live from the real processed dataset; add a partial-dependence panel to `app.py` that actually answers "what would change my score" instead of the generic hardcoded "Suggestions" text; make the post-prediction render path read exclusively from the snapshotted `st.session_state` input vector (not live widgets) everywhere, not just where it already happened to be correct; and replace the information-free Success/Failure pie chart with a gauge read against the real training-cohort base rate.

**Architecture:** `analytics.py` loads the committed `data/processed/startups_features_v1.parquet` (newly carved out of the `data/processed/` gitignore rule, same pattern as `models/production/`) and computes four real groupby breakdowns at render time - no precomputed/cached numbers, no hardcoded results. `src/models/partial_dependence.py` holds a pure, tested `compute_partial_dependence` function reused by `app.py`. A new `models/production/population_stats.json` (generated once from the training cohort) supplies the PDP panel's per-feature grids and the gauge's base-rate reference - both are real, training-cohort-derived numbers, not literals typed into `app.py`.

**Tech Stack:** pandas, plotly (bar/line/indicator), streamlit, pytest.

**Spec:** user-supplied spec (see conversation) - analytics tab computed from the processed dataset with sample sizes in chart titles and n<30 suppression; a partial-dependence panel; fixing the session-state desync bug by rendering exclusively from the stored input vector; deleting the pie chart for a base-rate gauge. Acceptance: `grep` `analytics.py` for numeric literals - none apart from figure sizing and thresholds.

## Verified real state (do not re-derive)

Computed from the corrected `data/processed/startups_features_v1.parquet` (13,334 rows, post the `compute_dates` bug fix committed just before this plan):

- **Country** (n≥30 groups): USA n=8,172 mean=0.644; GBR n=540 mean=0.480; CAN n=395 mean=0.600; ISR n=205 mean=0.624; RUS n=183 mean=0.077; DEU n=181 mean=0.575; FRA n=181 mean=0.497; CHN n=180 mean=0.633; IND n=129 mean=0.465; ESP n=94 mean=0.383; ... 22 countries total pass n≥30.
- **Primary category** (n≥30 groups, 66 total pass): Biotechnology n=1,102 mean=0.757; Software n=966 mean=0.646; Curated Web n=751 mean=0.447; Advertising n=590 mean=0.614; E-Commerce n=443 mean=0.492; Mobile n=417 mean=0.600; Enterprise Software n=367 mean=0.768; Games n=308 mean=0.487; Analytics n=285 mean=0.733; ...
- **Funding band** (`np.expm1(funding_total_usd_log1p)` binned `[0, 1e5, 1e6, 1e7, 1e8, inf]`): `<$100K` n=915 mean=0.101; `$100K-$1M` n=2,074 mean=0.277; `$1M-$10M` n=3,759 mean=0.573; `$10M-$100M` n=3,800 mean=0.764; `$100M+` n=595 mean=0.857. **This is the leaky `funding_total_usd` field** (same one flagged throughout `DATA_CARD.md`/`MODEL_CARD.md`) - the chart needs an explicit caveat that this correlation is partly circular (more total funding is partly a symptom of having survived long enough to raise more, especially for `acquired` companies whose post-acquisition funding is counted), not a lever a founder can causally pull.
- **Company age** (`founded_year.max() - founded_year`, binned `[0,2,5,10,20,200)` years - all bands clear n≥30 so no suppression triggers here, but the code path is generic): `0-2y` n=1,273 mean=0.157; `2-5y` n=2,376 mean=0.403; `5-10y` n=4,609 mean=0.501; `10-20y` n=4,208 mean=0.696; `20y+` n=868 mean=0.805. **This also needs a caveat**, distinct from the funding one: it's not leakage (age is known at prediction time), but it's largely a restatement of the same survivorship/censoring dynamic already in `DATA_CARD.md` - younger companies (at scrape time) have had less time to resolve to `acquired`/`ipo` at all, so the correlation is mostly mechanical, not a discovered causal pattern.
- Numeric feature ranges (train split, for the PDP grids): `founded_year` [1901, 2011]; `time_to_first_funding_days` [-5022, 41147]; `funding_total_usd_log1p` [1.099, 24.127]; `funding_rounds` [1, 18]; `funding_span_days` [0, 17287]. (`time_to_first_funding_days` going negative reflects real data noise - some `first_funding_at` predates the cleaned `founded_at`; the PDP grid uses the real observed min/max as-is, it isn't the app's job to silently clip real data.)
- Train-cohort base rate for the gauge: **59.83%** (10,672 rows) - reuse this, not the raw/test rate, for consistency with the percentile panel which already frames everything against "the training cohort."

## Global Constraints

- `analytics.py` must contain no numeric literal representing a computed fact (a rate, a count, a specific data value). Bin edges (`100_000`, `1_000_000`, ...), the `n < 30` suppression threshold, `top_n` caps, and plotly figure-sizing kwargs are the only numeric literals allowed, per the acceptance criterion - and the plan's own review step great this with `grep -n '[0-9]' analytics.py` and justifies every hit.
- Every chart title includes the total `n` backing that specific chart (sum of the counts of the groups actually plotted, i.e. post-suppression).
- Groups with `n < 30` are dropped from the plotted data entirely, not shown greyed-out or with a warning bar.
- The funding-band and age-band charts each carry a one-line caveat (leakage for funding, survivorship/censoring for age) rendered via `st.caption`, matching the rigor already established in `DATA_CARD.md`/`MODEL_CARD.md` - this dashboard shouldn't imply causality the rest of the project has been careful to deny.
- `app.py`'s entire post-prediction render block (everything under `if "prediction_prob" in st.session_state:`) reads only `st.session_state.prediction_prob` and fields extracted from `st.session_state.prediction_input` - never a live widget variable (`founded_date`, `funding_total_usd`, `country_code`, etc.) directly. This is an invariant to enforce across the whole block, not a single-line patch.
- The partial-dependence panel calls the real `calibrated_pipeline.predict_proba`, varies exactly one feature at a time via `compute_partial_dependence` (a pure, unit-tested function - no Streamlit/IO inside it), and holds every other feature fixed at the stored prediction's actual values.
- `models/production/population_stats.json` (numeric ranges + categorical top-N + train base rate) is generated once from the training split and committed, same reasoning as `category_options.json` and the calibrated model itself: the deployed app can't regenerate Kaggle-credentialed data.
- `data/processed/startups_features_v1.parquet` itself must also be committed (negate it out of the `data/processed/` gitignore rule) so `analytics.py` has something to read from at runtime, consistent with how `models/production/` was carved out of `/models/`.

---

## Task 1: `src/models/partial_dependence.py` (TDD)

**Files:**
- Create: `tests/test_partial_dependence.py`
- Create: `src/models/partial_dependence.py`

**Interfaces:**
- Produces: `compute_partial_dependence(pipeline, feature_cols: list[str], base_row: pd.DataFrame, varying_feature: str, grid_values: list) -> pd.DataFrame` with columns `[varying_feature, "predicted_probability"]`, one row per grid value, holding every other feature fixed at `base_row`'s values.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_partial_dependence.py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.models.partial_dependence import compute_partial_dependence


def _toy_pipeline():
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.uniform(0, 10, n)  # strongly, monotonically predictive
    x2 = rng.uniform(0, 10, n)  # noise
    y = (x1 + rng.normal(0, 0.3, n) > 5).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "cat": rng.choice(["a", "b"], n), "label": y})

    preprocessor = ColumnTransformer([
        ("num", "passthrough", ["x1", "x2"]),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["cat"]),
    ])
    pipeline = Pipeline([("preprocess", preprocessor), ("clf", HistGradientBoostingClassifier(random_state=0))])
    pipeline.fit(df[["x1", "x2", "cat"]], df["label"])
    return pipeline


def test_partial_dependence_shape_matches_grid():
    pipeline = _toy_pipeline()
    base_row = pd.DataFrame([{"x1": 5.0, "x2": 5.0, "cat": "a"}])
    grid = [0.0, 2.5, 5.0, 7.5, 10.0]
    result = compute_partial_dependence(pipeline, ["x1", "x2", "cat"], base_row, "x1", grid)
    assert list(result["x1"]) == grid
    assert result["predicted_probability"].between(0, 1).all()


def test_partial_dependence_holds_other_features_fixed():
    # x2 is pure noise for the label, but must stay at base_row's value across the grid -
    # verified indirectly: varying x1 across its full observed range should move the
    # probability monotonically upward given the toy data's construction.
    pipeline = _toy_pipeline()
    base_row = pd.DataFrame([{"x1": 5.0, "x2": 9.0, "cat": "a"}])
    grid = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    result = compute_partial_dependence(pipeline, ["x1", "x2", "cat"], base_row, "x1", grid)
    probs = result["predicted_probability"].tolist()
    assert probs[-1] > probs[0]
    assert probs == sorted(probs)  # monotonic non-decreasing, matching the toy construction


def test_partial_dependence_categorical_grid():
    pipeline = _toy_pipeline()
    base_row = pd.DataFrame([{"x1": 5.0, "x2": 5.0, "cat": "a"}])
    result = compute_partial_dependence(pipeline, ["x1", "x2", "cat"], base_row, "cat", ["a", "b"])
    assert list(result["cat"]) == ["a", "b"]
    assert len(result) == 2
```

- [ ] **Step 2: Run tests, confirm failure**

Run: `python3 -m pytest tests/test_partial_dependence.py -v`
Expected: `ModuleNotFoundError: No module named 'src.models.partial_dependence'`

- [ ] **Step 3: Write `src/models/partial_dependence.py`**

```python
"""Vary one feature across a grid, holding all others fixed, and record the
pipeline's predicted probability at each point - a single-feature partial
dependence curve for one concrete prediction."""
from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline


def compute_partial_dependence(
    pipeline: Pipeline,
    feature_cols: list[str],
    base_row: pd.DataFrame,
    varying_feature: str,
    grid_values: list,
) -> pd.DataFrame:
    rows = []
    for value in grid_values:
        row = base_row.copy()
        row[varying_feature] = value
        rows.append(row)
    grid_df = pd.concat(rows, ignore_index=True)
    proba = pipeline.predict_proba(grid_df[feature_cols])[:, 1]
    return pd.DataFrame({varying_feature: grid_values, "predicted_probability": proba})
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `python3 -m pytest tests/test_partial_dependence.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/partial_dependence.py tests/test_partial_dependence.py
git commit -m "feat(models): add partial-dependence computation for one feature at a time"
```

---

## Task 2: Generate and commit `models/production/population_stats.json`, negate `data/processed/` for analytics

**Files:**
- Modify: `.gitignore`
- Create: `models/production/population_stats.json` (generated by a one-off command, not a new module - mirrors how `category_options.json` was produced)

**Interfaces:**
- Consumes: `data/processed/startups_features_v1.parquet` (train split)
- Produces: `models/production/population_stats.json` with `{"numeric_ranges": {feature: {"min":, "max":}}, "categorical_top": {feature: [values...]}, "train_base_rate": float}`

- [ ] **Step 1: Update `.gitignore` to commit the processed parquet**

Change:
```
# Generated data (regenerate with: python -m src.data.build)
data/processed/
```
to:
```
# Generated data (regenerate with: python -m src.data.build)
# startups_features_v1.parquet is committed: the deployed app's Analytics tab
# and partial-dependence panel need it at runtime and can't regenerate
# Kaggle-credentialed data at deploy time.
data/processed/*
!data/processed/startups_features_v1.parquet
```

- [ ] **Step 2: Generate `models/production/population_stats.json`**

Run:
```bash
python3 -c "
import json
import pandas as pd

from src.data import schema

df = pd.read_parquet('data/processed/startups_features_v1.parquet')
train_df = df[df['split'] == 'train']

numeric_ranges = {
    col: {'min': float(train_df[col].min()), 'max': float(train_df[col].max())}
    for col in schema.FULL_NUMERIC_FEATURES
}
categorical_top = {
    col: train_df[col].value_counts().head(20).index.tolist()
    for col in schema.FULL_CATEGORICAL_FEATURES
}
stats = {
    'numeric_ranges': numeric_ranges,
    'categorical_top': categorical_top,
    'train_base_rate': float(train_df['label'].mean()),
}
with open('models/production/population_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2)[:500])
"
```
Expected: prints the ranges/top-categories/base-rate; `train_base_rate` should be `0.5983...` matching the verified fact above.

- [ ] **Step 3: Verify both files are now stageable**

Run: `git status --porcelain=v1 | grep -E "population_stats|startups_features"`
Expected: both show as untracked/addable, not silently swallowed by `.gitignore`.

- [ ] **Step 4: Commit**

```bash
git add .gitignore data/processed/startups_features_v1.parquet models/production/population_stats.json
git commit -m "feat(data): commit processed parquet and population stats for the deployed app"
```

---

## Task 3: Rewrite `analytics.py` with real computed breakdowns

**Files:**
- Modify: `analytics.py` (full rewrite of `show_dashboard`)

**Interfaces:**
- Consumes: `data/processed/startups_features_v1.parquet`

- [ ] **Step 1: Write the new `analytics.py`**

```python
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = "data/processed/startups_features_v1.parquet"
MIN_GROUP_SIZE = 30

FUNDING_BIN_EDGES = [0, 100_000, 1_000_000, 10_000_000, 100_000_000, np.inf]
FUNDING_BIN_LABELS = ["<$100K", "$100K-$1M", "$1M-$10M", "$10M-$100M", "$100M+"]

AGE_BIN_EDGES = [0, 2, 5, 10, 20, np.inf]
AGE_BIN_LABELS = ["0-2y", "2-5y", "5-10y", "10-20y", "20y+"]

TOP_N_GROUPS = 15


@st.cache_data
def _load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def _grouped_success_rate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_col, observed=True)["label"].agg(success_rate="mean", n="count").reset_index()
    return grouped[grouped["n"] >= MIN_GROUP_SIZE].sort_values("n", ascending=False)


def _bar_chart(df: pd.DataFrame, x: str, title: str, order: list[str] | None = None):
    total_n = int(df["n"].sum())
    fig = px.bar(
        df, x=x, y="success_rate", text="n",
        category_orders={x: order} if order else None,
        title=f"{title} (n={total_n:,})",
        labels={"success_rate": "Success rate"},
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig.update_yaxes(tickformat=".0%")
    return fig


def show_dashboard() -> None:
    """Renders the analytics dashboard from the real processed dataset - every
    number here is computed live, nothing is hand-typed."""
    st.title("📊 Startup Analytics Dashboard")

    df = _load_dataset()
    overall = df["label"].mean()
    st.metric("Overall success rate", f"{overall:.1%}", help=f"n={len(df):,} labeled companies (operating excluded)")

    st.subheader("By country")
    by_country = _grouped_success_rate(df, "country_code").head(TOP_N_GROUPS)
    st.plotly_chart(_bar_chart(by_country, "country_code", "Success rate by country"), use_container_width=True)

    st.subheader("By primary category")
    by_category = _grouped_success_rate(df, "primary_category").head(TOP_N_GROUPS)
    st.plotly_chart(_bar_chart(by_category, "primary_category", "Success rate by primary category"), use_container_width=True)

    st.subheader("By funding band")
    funding_band = pd.cut(np.expm1(df["funding_total_usd_log1p"]), bins=FUNDING_BIN_EDGES, labels=FUNDING_BIN_LABELS)
    by_funding = _grouped_success_rate(df.assign(funding_band=funding_band), "funding_band")
    st.plotly_chart(_bar_chart(by_funding, "funding_band", "Success rate by funding band", order=FUNDING_BIN_LABELS), use_container_width=True)
    st.caption(
        "⚠️ `funding_total_usd` is measured at scrape time, after the outcome is known "
        "(see DATA_CARD.md/MODEL_CARD.md's leakage finding). This is a correlation among "
        "resolved companies, not a lever a founder can causally pull by raising more."
    )

    st.subheader("By company age")
    age_years = df["founded_year"].max() - df["founded_year"]
    age_band = pd.cut(age_years, bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS, right=False)
    by_age = _grouped_success_rate(df.assign(age_band=age_band), "age_band")
    st.plotly_chart(_bar_chart(by_age, "age_band", "Success rate by company age", order=AGE_BIN_LABELS), use_container_width=True)
    st.caption(
        "⚠️ Not leakage (age is known upfront), but largely mechanical: younger companies "
        "(at scrape time) have had less time to resolve to acquired/ipo at all, so this "
        "mirrors the same survivorship/censoring dynamic documented in DATA_CARD.md."
    )
```

- [ ] **Step 2: Grep for numeric literals and justify every one**

Run: `grep -n '[0-9]' analytics.py`
Expected: only `MIN_GROUP_SIZE = 30` (threshold), `FUNDING_BIN_EDGES`/`AGE_BIN_EDGES` (bucket-boundary thresholds), `TOP_N_GROUPS = 15` (display cap, a threshold), and format-string precision like `.1%`/`:,` (formatting, not data) should appear - no hand-typed rate or count. If anything else shows up, fix it before moving on.

- [ ] **Step 3: Manually sanity-check the numbers against the verified real state**

Run: `streamlit run app.py --server.headless true --server.port 8502 & sleep 4 && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8502 && kill %1`
Then re-derive the same aggregates directly in a one-off `python3 -c` groupby (same code as the "Verified real state" section) and confirm they match what `_grouped_success_rate` would produce - the numbers were already verified once during planning; confirm the shipped function reproduces them exactly.

- [ ] **Step 4: Commit**

```bash
git add analytics.py
git commit -m "feat(analytics): compute all dashboard numbers from the real processed dataset"
```

---

## Task 4: `app.py` - partial-dependence panel, strict session-state read, gauge replaces pie chart

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `src.models.partial_dependence.compute_partial_dependence`, `models/production/population_stats.json`

- [ ] **Step 1: Load `population_stats.json` alongside the other production artifacts**

In `load_production_artifacts()`, add:
```python
    population_stats = json.loads((PRODUCTION_DIR / "population_stats.json").read_text())
    return calibrated, base, reference, options, population_stats
```
and update the call site: `calibrated_pipeline, base_pipeline, train_reference, category_options, population_stats = load_production_artifacts()`.

- [ ] **Step 2: Replace the "Suggestions" block with a partial-dependence panel**

Replace:
```python
            st.subheader("🎯 Suggestions")
            if prob < 50:
                st.write("• Shorten time-to-first-funding\n• Pursue additional funding rounds\n• Target regions/categories with stronger historical outcomes")
            else:
                st.write("• Maintain funding momentum\n• Expand into additional high-performing regions")
```
with:
```python
            st.subheader("🔬 What would change this score?")
            pdp_feature = st.selectbox("Vary this input, holding everything else fixed", FEATURE_COLS, key="pdp_feature")
            input_row_snapshot = st.session_state.prediction_input

            if pdp_feature in population_stats["numeric_ranges"]:
                bounds = population_stats["numeric_ranges"][pdp_feature]
                grid = np.linspace(bounds["min"], bounds["max"], 30).tolist()
            else:
                grid = population_stats["categorical_top"][pdp_feature]

            pdp_df = compute_partial_dependence(calibrated_pipeline, FEATURE_COLS, input_row_snapshot, pdp_feature, grid)
            current_value = input_row_snapshot.iloc[0][pdp_feature]

            if pdp_feature in population_stats["numeric_ranges"]:
                pdp_fig = px.line(pdp_df, x=pdp_feature, y="predicted_probability", markers=True,
                                   title=f"Calibrated probability vs. {pdp_feature}")
                pdp_fig.add_vline(x=current_value, line_dash="dash", annotation_text="current input")
            else:
                pdp_fig = px.bar(pdp_df, x=pdp_feature, y="predicted_probability",
                                  title=f"Calibrated probability vs. {pdp_feature} (top {len(grid)} by frequency)")
            pdp_fig.update_yaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(pdp_fig, use_container_width=True)
```

- [ ] **Step 3: Replace the pie chart with a base-rate gauge**

Replace:
```python
        with col_b:
            fig = px.pie(values=[prob, 100-prob], names=["Success", "Failure"], title="Prediction Split", color_discrete_sequence=['#22c55e', '#ef4444'])
            st.plotly_chart(fig, use_container_width=True)
```
with:
```python
        with col_b:
            base_rate = population_stats["train_base_rate"] * 100
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"suffix": "%"},
                title={"text": "Score vs. training-cohort base rate"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#22c55e"},
                    "threshold": {"line": {"color": "#ef4444", "width": 4}, "thickness": 0.9, "value": base_rate},
                },
            ))
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.caption(f"Red line = training-cohort base rate ({base_rate:.1f}%), not 50%.")
```
Add `import plotly.graph_objects as go` and `from src.models.partial_dependence import compute_partial_dependence` to the top imports.

- [ ] **Step 4: Audit and fix the full post-prediction block for live-widget reads**

Read the entire `if "prediction_prob" in st.session_state:` block after the above edits. Every reference to `founded_date`, `first_funding_date`, `last_funding_date`, `funding_total_usd`, `funding_rounds`, `country_code`, `region`, `primary_category` inside that block (this currently includes the `report_data` f-string) must be replaced with the corresponding value pulled from `st.session_state.prediction_input.iloc[0]`. Concretely, replace:
```python
        report_data = (
            f"Startup Success Probability (calibrated): {prob:.2f}%\n"
            f"Percentile vs. training cohort: {percentile:.1f}%\n"
            f"Founded: {founded_date} | First funding: {first_funding_date} | Last funding: {last_funding_date}\n"
            f"Total funding: ${funding_total_usd} | Rounds: {funding_rounds}\n"
            f"Country: {country_code} | Region: {region} | Category: {primary_category}"
        )
```
with:
```python
        snap = st.session_state.prediction_input.iloc[0]
        report_data = (
            f"Startup Success Probability (calibrated): {prob:.2f}%\n"
            f"Percentile vs. training cohort: {percentile:.1f}%\n"
            f"Founded year: {snap['founded_year']:.0f} | Time to first funding: {snap['time_to_first_funding_days']:.0f} days\n"
            f"Total funding: ${np.expm1(snap['funding_total_usd_log1p']):,.0f} | Rounds: {snap['funding_rounds']:.0f} | Funding span: {snap['funding_span_days']:.0f} days\n"
            f"Country: {snap['country_code']} | Region: {snap['region']} | Category: {snap['primary_category']}"
        )
```
This is the one place in the current code where a live widget variable actually did leak into the post-prediction render (verified during planning - `explain_prediction` and the pie/gauge/percentile already only used session state). Confirm no other reference to a bare `founded_date`/`funding_total_usd`/etc. remains inside the block: `grep -n "founded_date\|funding_total_usd\b\|funding_rounds\b\|country_code\b\|region\b\|primary_category\b" app.py` and check every match inside the post-prediction block resolves through `snap` or `st.session_state.prediction_input`, not a bare widget variable.

- [ ] **Step 5: Compile check, then start the app and verify no exceptions**

Run: `python3 -m py_compile app.py && echo OK`
Then run the same background-streamlit-plus-curl check used in the prior plan, and re-run the standalone prediction-path smoke script (extended to also call `compute_partial_dependence` and read `population_stats.json`) to verify the full chain end-to-end without a browser tool.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat(app): add partial-dependence panel, fix session-state consistency, replace pie chart with base-rate gauge"
```

---

## Task 5: Final verification

- [ ] **Step 1: Full test suite**

Run: `python3 -m pytest -v`
Expected: all tests (new `tests/test_partial_dependence.py` plus every prior test) pass.

- [ ] **Step 2: Acceptance grep**

Run: `grep -n '[0-9]' analytics.py`
Expected: matches only figure-sizing/threshold/bin-edge/format-precision literals, as reviewed in Task 3 Step 2 - paste the actual output into the final report to your human partner rather than asserting it passed.

---

## Self-review notes

- Spec coverage: analytics computed from the dataset with n in titles and n<30 suppression (Task 3), partial-dependence panel (Task 1 + Task 4 Step 2), session-state-only rendering (Task 4 Step 4, which found and fixed the one real instance of the described bug pattern - the `report_data` block - even though the two locations the user named, "Why this result" and the pie chart, were already correct from the prior plan), pie chart deleted and replaced with a base-rate gauge (Task 4 Step 3), acceptance's literal-grep (Task 3 Step 2, Task 5 Step 2).
- No placeholders: all steps carry full code.
- Type consistency: `compute_partial_dependence`'s signature (`pipeline, feature_cols, base_row, varying_feature, grid_values`) is identical between Task 1's tests, its implementation, and Task 4's call site.
- Also fixed in this session (not part of this plan's tasks, but same working session): a real `compute_dates` upper-bound bug found while validating the age-band chart's numbers, corrected at the source in a separate prior commit - see `DATA_CARD.md`'s "Correction" note.
