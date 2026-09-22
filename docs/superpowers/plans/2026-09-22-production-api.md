# Production-Shaped API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `api.py`'s query-string-bound, unvalidated, relative-path-broken `/predict` with a real FastAPI service on the calibrated production pipeline: Pydantic request/response models, `/predict`, `/predict/batch`, `/health`, `/model-info`, model/explainer/ECDF loaded once via a lifespan handler, structured 422s, `tests/test_api.py`, CI, pinned requirements, and a Dockerfile.

**Architecture:** `api.py` mirrors `app.py`'s Step 3/4 production wiring (`models/production/hist_gradient_boosting_full_calibrated*`, `src/models/explain.py`, `src/models/percentile.py`) instead of the old 4-feature toy model - the spec's own `PredictionResponse` shape (`calibrated` flag, `percentile`, `top SHAP contributors`) only makes sense against that pipeline, and this closes the gap flagged in the README's Future Scope after Step 3's app.py rewire. All I/O paths resolve via `Path(__file__).resolve().parent`, matching `predictor.py`'s existing correct pattern. Model/base-model/ECDF/metadata load once in a FastAPI `lifespan` context manager into `app.state`-equivalent module dict, not at import time or per request - and the SHAP explainer is deliberately primed once during that same startup, not lazily on first request.

**Tech Stack:** FastAPI, Pydantic v2, uvicorn, httpx (TestClient), pytest, ruff, GitHub Actions, Docker.

**Spec:** user-supplied spec (see conversation) - Pydantic models with `Field` constraints, bounded-batch endpoint, lifespan-loaded model/calibrator/explainer/ECDF, structured 422 errors, `tests/test_api.py`, `.github/workflows/ci.yml`, pinned `requirements.txt` + `requirements-dev.txt`, non-root-user Dockerfile with healthcheck. Acceptance: pytest passes, CI green, `uvicorn api:app` started from `/` still serves predictions.

## Verified real state (do not re-derive)

- Current `api.py` binds bare scalars (`funding: float, experience: int, team: int, market: int`) as query params on a `POST` (FastAPI's default for unannotated scalar params with no `Body()`/Pydantic model), loads `startup_model.pkl` via a bare relative path that breaks under any other CWD, and serves the old 4-feature toy model - unrelated to the real pipeline built in Steps 1-4.
- `predictor.py`'s correct path pattern to copy: `BASE_DIR = Path(__file__).resolve().parent`, then `BASE_DIR / "startup_model.pkl"`.
- Installed and importable this session: `httpx` 0.28.1, `pydantic` 2.13.5, `fastapi` 0.141.1, `uvicorn` 0.53.0, `ruff` 0.16.1 (as a standalone binary; not yet in `requirements-dev.txt`). Docker CLI 29.7.2 is installed but its daemon (`colima`) is not running in this environment - the Dockerfile will be written carefully and reviewed by hand, but an actual `docker build` cannot be validated here; say so rather than claiming it was tested.
- Exact installed versions for pinning (`pip show`): `streamlit==1.64.0`, `scikit-learn==1.9.1`, `joblib==1.6.0`, `numpy==2.5.3`, `pandas==3.0.6`, `google-genai==2.24.0`, `huggingface_hub==1.32.0`, `fastapi==0.141.1`, `uvicorn==0.53.0`, `requests==2.34.2`, `plotly==7.1.0`, `pyarrow==25.0.1`, `shap==0.52.0`, `pydantic==2.13.5`. Dev-only: `pytest==9.1.1`, `httpx==0.28.1`, `kagglehub==1.0.2`, `matplotlib==3.11.2`, `nbformat==5.11.1`, `nbclient==0.11.0`, `ipykernel==7.3.0`, `nbconvert==7.17.1`, `ruff==0.16.1`.
- `ruff check src/ tests/` under a deliberately-scoped ruleset (`E4,E7,E9,F,I` - a standard, non-opinionated baseline; the current default invocation also flags `N999` on `src/__init__.py` because the repo directory itself is named `startup-success-predictor`, which is a directory-naming false positive, plus a few opinionated `RUF0xx` preview-ish rules) already passes clean after this session's import-sort fixes (committed separately). `pyproject.toml` will pin this exact rule selection so CI behavior is explicit and deterministic, not "whatever ruff defaults to this version."
- Real regression value for the known-input test (computed directly against the committed `models/production/hist_gradient_boosting_full_calibrated.joblib`): `founded_year=2013, time_to_first_funding_days=151, funding_total_usd=1_000_000, funding_rounds=2, funding_span_days=214, country_code="USA", region="SF Bay Area", primary_category="Software"` -> **probability = 0.42857142857142855**. `"SF Bay Area"` and `"Software"` are confirmed-real values from `models/production/category_options.json`.
- `models/production/hist_gradient_boosting_full_calibrated_metadata.json` (committed, so available at API runtime) has exactly: `model_name`, `calibration_method`, `numeric_features`, `categorical_features`, `n_train_fit`, `n_calibration`, `n_test`, `brier_before`, `brier_after`, `roc_auc_before`, `roc_auc_after`, `git_sha`, `git_dirty` - this is the real source for `/model-info`, not a re-parse of `MODEL_CARD.md` prose. The Step 2 comparison-model metadata under plain `models/` is gitignored and won't exist in CI/deployment, so `/model-info` must only ever touch `models/production/`.

## Global Constraints

- Every file the API touches at runtime lives under `models/production/` (already committed) - never the gitignored `models/` comparison outputs or `data/processed/` (the API doesn't need the dataset at all, only `app.py`'s Analytics tab does).
- All paths in `api.py` resolve from `Path(__file__).resolve().parent`, so `uvicorn api:app` works identically regardless of the process's current working directory.
- Model, base model, ECDF reference, and metadata load exactly once, in a `lifespan` async context manager - not at import time (breaks `TestClient` construction speed and import-time side effects) and not per-request (defeats the point of "cache the explainer at load time").
- `StartupFeatures` matches the real 8-feature schema (`src/data/schema.py`'s `FULL_NUMERIC_FEATURES`/`FULL_CATEGORICAL_FEATURES`), not the spec's illustrative `funding_usd`/`team_size` example names (those don't exist in this pipeline - `team_size` was the old toy model's feature). `Field(ge=...)` constraints follow the same pattern the spec illustrated (`funding_total_usd: float = Field(ge=0)`, `funding_rounds: int = Field(ge=1)`).
- `/predict/batch` is bounded via a wrapper Pydantic model (`Field(min_length=1, max_length=100)` on a `list[StartupFeatures]` field) rather than relying on version-sensitive `Body()` constraint support.
- Pydantic's default `RequestValidationError` handling (automatic 422 naming the failed field) is left untouched; a custom top-level `Exception` handler is added only to stop *unexpected* errors from leaking a raw stack trace to the client (still logged server-side via the default logger).
- `requirements.txt` becomes the lean **production runtime** set (what `app.py` and `api.py` actually import to serve predictions) with exact `==` pins; `requirements-dev.txt` holds test/lint/training/notebook-only tools (`pytest`, `httpx`, `ruff`, `kagglehub`, `matplotlib`, `nbformat`, `nbclient`, `ipykernel`, `nbconvert`), also pinned. README's install section gets a one-line update for this split.
- CI (`.github/workflows/ci.yml`) installs both requirement files, runs `ruff check` scoped to `src/ tests/ api.py` (not the whole repo - untouched legacy files like `advisor_ai.py`/`gemini_test.py` are out of this task's scope and shouldn't gate CI on pre-existing lint debt), then `pytest`.
- Dockerfile only packages what `api.py` needs (`api.py`, `src/`, `models/production/`) - not the Streamlit app, not the training pipeline - runs as a non-root user, and has a `HEALTHCHECK` hitting `/health` via Python's stdlib (no extra `curl` package needed in the slim image).

---

## Task 1: `pyproject.toml` ruff config + `requirements-dev.txt` split

**Files:**
- Create: `pyproject.toml`
- Create: `requirements-dev.txt`
- Modify: `requirements.txt`
- Modify: `README.md` (one-line install instruction update)

**Interfaces:**
- Produces: a `[tool.ruff]`/`[tool.ruff.lint]` config CI will use verbatim; `requirements.txt` (production) and `requirements-dev.txt` (dev/test/tooling), both pinned.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[tool.ruff]
target-version = "py312"
line-length = 120

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I"]
```

- [ ] **Step 2: Rewrite `requirements.txt` to the pinned production set**

```
streamlit==1.64.0
scikit-learn==1.9.1
joblib==1.6.0
numpy==2.5.3
pandas==3.0.6
google-genai==2.24.0
huggingface_hub==1.32.0
fastapi==0.141.1
uvicorn==0.53.0
requests==2.34.2
plotly==7.1.0
pyarrow==25.0.1
shap==0.52.0
pydantic==2.13.5
```

- [ ] **Step 3: Write `requirements-dev.txt`**

```
-r requirements.txt
pytest==9.1.1
httpx==0.28.1
ruff==0.16.1
kagglehub==1.0.2
matplotlib==3.11.2
nbformat==5.11.1
nbclient==0.11.0
ipykernel==7.3.0
nbconvert==7.17.1
```

- [ ] **Step 4: Update `README.md`'s dependency install step**

Find the `### 2. Install Dependencies` section and replace its code block:
```bash
pip install -r requirements.txt
```
with:
```bash
pip install -r requirements.txt          # production runtime only
pip install -r requirements-dev.txt      # + testing/linting/training/notebook tools
```

- [ ] **Step 5: Verify the pinned sets actually install clean in the current environment**

Run: `pip3 install -q -r requirements.txt -r requirements-dev.txt 2>&1 | tail -20`
Expected: no errors (everything is already installed at these exact versions this session, so this should be a no-op confirming the pins are internally consistent).

- [ ] **Step 6: Re-run ruff under the new explicit config**

Run: `ruff check src/ tests/`
Expected: `All checks passed!` (already true before this step; confirms `pyproject.toml`'s config doesn't change that).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements.txt requirements-dev.txt README.md
git commit -m "build: pin requirements.txt to production runtime, add requirements-dev.txt, add ruff config"
```

---

## Task 2: Rewrite `api.py`

**Files:**
- Modify: `api.py` (full rewrite)

**Interfaces:**
- Consumes: `src.data.schema`, `src.models.explain.explain_prediction`, `src.models.percentile.compute_percentile`/`load_reference_distribution`, `models/production/hist_gradient_boosting_full_calibrated*`

- [ ] **Step 1: Write `api.py`**

```python
"""Production FastAPI service for the calibrated startup-outcome model.

All paths resolve from this file's location (not the process CWD), and the
model/explainer/ECDF load exactly once, at startup, via the lifespan handler
below - never per request.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.data import schema
from src.models.explain import explain_prediction
from src.models.percentile import compute_percentile, load_reference_distribution

BASE_DIR = Path(__file__).resolve().parent
PRODUCTION_DIR = BASE_DIR / "models" / "production"
FEATURE_COLS = schema.FULL_NUMERIC_FEATURES + schema.FULL_CATEGORICAL_FEATURES
MAX_BATCH_SIZE = 100

ml_state: dict = {}


class StartupFeatures(BaseModel):
    founded_year: float = Field(ge=1900, le=2100, description="Year the company was founded")
    time_to_first_funding_days: float = Field(description="Days between founding and first funding round")
    funding_total_usd: float = Field(ge=0, description="Total funding raised, in USD")
    funding_rounds: int = Field(ge=1, description="Number of funding rounds")
    funding_span_days: float = Field(ge=0, description="Days between first and most recent funding round")
    country_code: str = Field(min_length=1, max_length=8)
    region: str = Field(min_length=1)
    primary_category: str = Field(min_length=1)


class BatchPredictionRequest(BaseModel):
    items: list[StartupFeatures] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class ShapContribution(BaseModel):
    feature: str
    value: float | str
    shap_value: float
    direction: Literal["increases", "decreases"]


class PredictionResponse(BaseModel):
    probability: float
    calibrated: bool
    percentile: float
    model_version: str
    top_contributors: list[ShapContribution]


class ModelInfoResponse(BaseModel):
    model_version: str
    calibration_method: str
    n_train_fit: int
    n_calibration: int
    n_test: int
    roc_auc_before_calibration: float
    roc_auc_after_calibration: float
    brier_before_calibration: float
    brier_after_calibration: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    calibrated = joblib.load(PRODUCTION_DIR / "hist_gradient_boosting_full_calibrated.joblib")
    base = joblib.load(PRODUCTION_DIR / "hist_gradient_boosting_full_calibrated_base.joblib")
    reference = load_reference_distribution(PRODUCTION_DIR / "train_score_distribution.npy")
    metadata = json.loads((PRODUCTION_DIR / "hist_gradient_boosting_full_calibrated_metadata.json").read_text())

    # Build (and cache) the SHAP explainer once here, at startup - not lazily
    # on the first prediction request.
    warmup_row = pd.DataFrame([{
        col: 0.0 if col in schema.FULL_NUMERIC_FEATURES else "UNKNOWN" for col in FEATURE_COLS
    }])
    explain_prediction(base, FEATURE_COLS, warmup_row, top_n=1)

    ml_state["calibrated"] = calibrated
    ml_state["base"] = base
    ml_state["reference"] = reference
    ml_state["metadata"] = metadata
    ml_state["model_version"] = f"{metadata['model_name']}@{metadata['git_sha'][:8]}"
    yield
    ml_state.clear()


app = FastAPI(title="Startup Success Predictor API", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _features_to_row(features: StartupFeatures) -> pd.DataFrame:
    return pd.DataFrame([{
        "founded_year": features.founded_year,
        "time_to_first_funding_days": features.time_to_first_funding_days,
        "funding_total_usd_log1p": float(np.log1p(features.funding_total_usd)),
        "funding_rounds": float(features.funding_rounds),
        "funding_span_days": features.funding_span_days,
        "country_code": features.country_code,
        "region": features.region,
        "primary_category": features.primary_category,
    }])


def _predict_one(features: StartupFeatures) -> PredictionResponse:
    row = _features_to_row(features)
    probability = float(ml_state["calibrated"].predict_proba(row[FEATURE_COLS])[0, 1])
    percentile = compute_percentile(probability, ml_state["reference"])
    contributions = explain_prediction(ml_state["base"], FEATURE_COLS, row, top_n=5)
    return PredictionResponse(
        probability=probability,
        calibrated=True,
        percentile=percentile,
        model_version=ml_state["model_version"],
        top_contributors=[ShapContribution(**c) for c in contributions],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded="calibrated" in ml_state)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    m = ml_state["metadata"]
    return ModelInfoResponse(
        model_version=ml_state["model_version"],
        calibration_method=m["calibration_method"],
        n_train_fit=m["n_train_fit"],
        n_calibration=m["n_calibration"],
        n_test=m["n_test"],
        roc_auc_before_calibration=m["roc_auc_before"],
        roc_auc_after_calibration=m["roc_auc_after"],
        brier_before_calibration=m["brier_before"],
        brier_after_calibration=m["brier_after"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StartupFeatures) -> PredictionResponse:
    return _predict_one(features)


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchPredictionRequest) -> list[PredictionResponse]:
    return [_predict_one(f) for f in request.items]
```

- [ ] **Step 2: Byte-compile check**

Run: `python3 -m py_compile api.py && echo OK`
Expected: `OK`

- [ ] **Step 3: Start the server from a directory that is NOT the repo root and hit every endpoint**

Run:
```bash
cd / && (cd /Users/avnishsingh1505/startup-success-predictor && python3 -m uvicorn api:app --host 127.0.0.1 --port 8010 --app-dir /Users/avnishsingh1505/startup-success-predictor &)
```
Actually run it the way the acceptance criterion phrases it - launched such that the working directory at invocation time is `/`, not the repo root:
```bash
cd / && nohup python3 -m uvicorn --host 127.0.0.1 --port 8010 api:app --app-dir /Users/avnishsingh1505/startup-success-predictor > /tmp/api_log.txt 2>&1 &
sleep 3
curl -s http://127.0.0.1:8010/health
curl -s http://127.0.0.1:8010/model-info
curl -s -X POST http://127.0.0.1:8010/predict -H "Content-Type: application/json" -d '{"founded_year": 2013, "time_to_first_funding_days": 151, "funding_total_usd": 1000000, "funding_rounds": 2, "funding_span_days": 214, "country_code": "USA", "region": "SF Bay Area", "primary_category": "Software"}'
curl -s -X POST http://127.0.0.1:8010/predict -H "Content-Type: application/json" -d '{"funding_rounds": 0}'
kill %1
```
Expected: `/health` returns `{"status":"ok","model_loaded":true}`; `/model-info` returns real calibration metadata; the valid `/predict` call returns `probability` ≈ `0.42857...`; the invalid call (missing required fields, `funding_rounds` below its `ge=1` floor) returns HTTP 422 naming the failed field(s), not a stack trace. If uvicorn's `--app-dir` doesn't behave as expected for this check, fall back to invoking `python3 -c "import subprocess; subprocess.run([...], cwd='/')"` to launch uvicorn with CWD genuinely at `/` while still finding `api.py` via `sys.path` - the real acceptance bar is "the process's CWD is not the repo root," however that's arranged.

- [ ] **Step 4: Commit**

```bash
git add api.py
git commit -m "feat(api): rewrite api.py as a production-shaped FastAPI service on the calibrated pipeline"
```

---

## Task 3: `tests/test_api.py`

**Files:**
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `api.app` via `fastapi.testclient.TestClient` (httpx-backed)

- [ ] **Step 1: Write `tests/test_api.py`**

```python
import pytest
from fastapi.testclient import TestClient

from api import app

KNOWN_INPUT = {
    "founded_year": 2013,
    "time_to_first_funding_days": 151,
    "funding_total_usd": 1_000_000,
    "funding_rounds": 2,
    "funding_span_days": 214,
    "country_code": "USA",
    "region": "SF Bay Area",
    "primary_category": "Software",
}
KNOWN_PROBABILITY = 0.42857142857142855


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["calibration_method"] == "isotonic"
    assert body["n_train_fit"] > 0
    assert 0.0 <= body["roc_auc_after_calibration"] <= 1.0


def test_predict_known_input_regression(client):
    response = client.post("/predict", json=KNOWN_INPUT)
    assert response.status_code == 200
    body = response.json()
    assert body["probability"] == pytest.approx(KNOWN_PROBABILITY, abs=0.005)
    assert body["calibrated"] is True
    assert 0.0 <= body["percentile"] <= 100.0
    assert body["model_version"].startswith("hist_gradient_boosting_full_calibrated@")
    assert len(body["top_contributors"]) == 5
    for contributor in body["top_contributors"]:
        assert contributor["direction"] in ("increases", "decreases")


def test_predict_rejects_invalid_funding_rounds(client):
    bad_input = dict(KNOWN_INPUT, funding_rounds=0)  # violates ge=1
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("funding_rounds" in str(err["loc"]) for err in detail)


def test_predict_rejects_missing_field(client):
    incomplete = dict(KNOWN_INPUT)
    del incomplete["country_code"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("country_code" in str(err["loc"]) for err in detail)


def test_predict_batch(client):
    response = client.post("/predict/batch", json={"items": [KNOWN_INPUT, KNOWN_INPUT]})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert body[0]["probability"] == pytest.approx(KNOWN_PROBABILITY, abs=0.005)


def test_predict_batch_rejects_oversized_list(client):
    oversized = {"items": [KNOWN_INPUT] * 101}
    response = client.post("/predict/batch", json=oversized)
    assert response.status_code == 422


def test_predict_batch_rejects_empty_list(client):
    response = client.post("/predict/batch", json={"items": []})
    assert response.status_code == 422
```

- [ ] **Step 2: Run the tests**

Run: `python3 -m pytest tests/test_api.py -v`
Expected: all PASS. If `test_predict_known_input_regression` fails on the exact probability, re-verify by computing it directly against `models/production/hist_gradient_boosting_full_calibrated.joblib` the same way it was derived during planning - don't loosen the tolerance to paper over a real behavior change.

- [ ] **Step 3: Commit**

```bash
git add tests/test_api.py
git commit -m "test(api): cover health, model-info, schema rejection, known-input regression, and batch"
```

---

## Task 4: `.github/workflows/ci.yml`

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Lint (ruff)
        run: ruff check src/ tests/ api.py

      - name: Test
        run: pytest -v
```

- [ ] **Step 2: Verify the exact commands it runs succeed locally**

Run: `ruff check src/ tests/ api.py && pytest -v`
Expected: ruff clean, all tests pass - this is the same sequence CI will run, so a local pass here is a strong signal CI will be green (final confirmation still requires an actual push, noted in the final report rather than asserted as fact).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow (ruff + pytest) on push and PR"
```

---

## Task 5: `Dockerfile`

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Write `.dockerignore`**

```
.venv/
__pycache__/
*.pyc
.git/
.pytest_cache/
.ruff_cache/
notebooks/
docs/
reports/
data/
.sf/
*.ipynb
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY src/ src/
COPY models/production/ models/production/

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Hand-review (docker daemon unavailable in this environment)**

Run: `docker info` to reconfirm the daemon is unreachable here.
Expected: connection error, as verified during planning. Note explicitly in the final report to the human partner that the Dockerfile was written and reviewed but `docker build`/`docker run` could not be executed in this environment - don't claim it was tested when it wasn't. If a working Docker daemon becomes available, the concrete verification is: `docker build -t startup-api . && docker run -d -p 8000:8000 --name startup-api-test startup-api && sleep 5 && docker inspect --format='{{.State.Health.Status}}' startup-api-test` should eventually report `healthy`, then `docker rm -f startup-api-test`.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "build: add Dockerfile (non-root user, healthcheck) for the API service"
```

---

## Task 6: Final verification and README note

**Files:**
- Modify: `README.md` (Future Scope bullet update, api.py description update)

- [ ] **Step 1: Update README's Project Structure and Future Scope**

In `## 📁 Project Structure`, change:
```
- `predictor.py` / `startup_model.pkl` / `api.py`: legacy 4-feature toy model, still used by the standalone FastAPI backend only (`api.py`) - **not** the Streamlit app as of this version.
```
to:
```
- `api.py`: production FastAPI backend on the calibrated pipeline (Pydantic validation, lifespan-loaded model/explainer/ECDF, `/predict`, `/predict/batch`, `/health`, `/model-info`). See `tests/test_api.py`.
- `predictor.py` / `startup_model.pkl`: legacy 4-feature toy model, no longer used by either `app.py` or `api.py` as of this version - kept only as an artifact of the project's earlier state.
```
In `## 🚀 Future Scope`, remove the now-done bullet `Rewire api.py onto the same calibrated pipeline app.py now uses (currently still serves the legacy 4-feature toy model).`

- [ ] **Step 2: Full local verification**

Run: `pytest -v && ruff check src/ tests/ api.py`
Expected: all tests pass (this now includes `tests/test_api.py`), ruff clean.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: mark api.py rewrite done in README"
```

---

## Self-review notes

- Spec coverage: Pydantic `StartupFeatures`/`PredictionResponse` with `Field` constraints (Task 2), `/predict`, `/predict/batch` (bounded), `/health`, `/model-info` (Task 2), lifespan-loaded model/calibrator/explainer/ECDF (Task 2 `lifespan`), structured 422s (Task 2, verified in Task 2 Step 3 and Task 3), `tests/test_api.py` with schema rejection + known-input regression + batch + health (Task 3), `.github/workflows/ci.yml` on push/PR (Task 4), pinned `requirements.txt` + `requirements-dev.txt` (Task 1), Dockerfile with non-root user + healthcheck (Task 5), `Path(__file__)` pattern copied from `predictor.py` (Task 2), acceptance's "uvicorn started from `/` still serves predictions" (Task 2 Step 3).
- No placeholders: all steps carry full code.
- Type consistency: `FEATURE_COLS` order in `api.py` matches `schema.FULL_NUMERIC_FEATURES + schema.FULL_CATEGORICAL_FEATURES` used everywhere else (`app.py`, `src/models/train.py`, `calibrate.py`); `_features_to_row`'s output columns match exactly what `explain_prediction`/`predict_proba` expect.
- Honesty flag carried through explicitly rather than silently dropped: Docker build/run cannot be verified in this environment (daemon not running) - Task 5 Step 3 says so and gives the exact commands to verify later instead of asserting success.
