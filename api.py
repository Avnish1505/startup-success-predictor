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
