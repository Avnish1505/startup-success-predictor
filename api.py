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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.models.explain import explain_prediction
from src.models.percentile import compute_percentile, load_reference_distribution

BASE_DIR = Path(__file__).resolve().parent
PRODUCTION_DIR = BASE_DIR / "models" / "production"
MODEL_NAMES = {"clean": "hist_gradient_boosting_clean_calibrated", "full": "hist_gradient_boosting_full_calibrated"}
DEFAULT_FEATURE_SET = "clean"
MAX_BATCH_SIZE = 100

ml_state: dict = {}


class StartupFeatures(BaseModel):
    founded_year: float = Field(ge=1900, le=2100, description="Year the company was founded")
    time_to_first_funding_days: float = Field(description="Days between founding and first funding round")
    country_code: str = Field(min_length=1, max_length=8)
    region: str = Field(min_length=1)
    primary_category: str = Field(min_length=1)
    funding_total_usd: float | None = Field(default=None, ge=0, description="Required only if feature_set='full'")
    funding_rounds: int | None = Field(default=None, ge=1, description="Required only if feature_set='full'")
    funding_span_days: float | None = Field(default=None, ge=0, description="Required only if feature_set='full'")
    feature_set: Literal["clean", "full"] = DEFAULT_FEATURE_SET


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
    feature_set: str
    percentile: float
    model_version: str
    top_contributors: list[ShapContribution]


class ModelInfoResponse(BaseModel):
    default_feature_set: str
    clean_model_version: str
    full_model_version: str
    clean_test_roc_auc: float
    full_test_roc_auc: float
    clean_test_brier: float
    full_test_brier: float
    leakage_note: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    for feature_set, name in MODEL_NAMES.items():
        calibrated = joblib.load(PRODUCTION_DIR / f"{name}.joblib")
        base = joblib.load(PRODUCTION_DIR / f"{name}_base.joblib")
        reference = load_reference_distribution(PRODUCTION_DIR / f"{name}_train_score_distribution.npy")
        metadata = json.loads((PRODUCTION_DIR / f"{name}_metadata.json").read_text())
        feature_cols = metadata["numeric_features"] + metadata["categorical_features"]

        # Build (and cache) the SHAP explainer once here, at startup - not lazily
        # on the first prediction request.
        warmup_row = pd.DataFrame([{
            col: 0.0 if col in metadata["numeric_features"] else "UNKNOWN" for col in feature_cols
        }])
        explain_prediction(base, feature_cols, warmup_row, top_n=1)

        ml_state[feature_set] = {
            "calibrated": calibrated,
            "base": base,
            "reference": reference,
            "metadata": metadata,
            "feature_cols": feature_cols,
            "model_version": f"{metadata['model_name']}@{metadata['git_sha'][:8]}",
        }
    yield
    ml_state.clear()


app = FastAPI(title="Startup Success Predictor API", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _features_to_row(features: StartupFeatures, feature_cols: list[str]) -> pd.DataFrame:
    row = {
        "founded_year": features.founded_year,
        "time_to_first_funding_days": features.time_to_first_funding_days,
        "country_code": features.country_code,
        "region": features.region,
        "primary_category": features.primary_category,
    }
    if "funding_total_usd_log1p" in feature_cols:
        if features.funding_total_usd is None or features.funding_rounds is None or features.funding_span_days is None:
            raise HTTPException(422, "funding_total_usd, funding_rounds, and funding_span_days are required when feature_set='full'")
        row["funding_total_usd_log1p"] = float(np.log1p(features.funding_total_usd))
        row["funding_rounds"] = float(features.funding_rounds)
        row["funding_span_days"] = features.funding_span_days
    return pd.DataFrame([row])


def _predict_one(features: StartupFeatures) -> PredictionResponse:
    model = ml_state[features.feature_set]
    row = _features_to_row(features, model["feature_cols"])
    probability = float(model["calibrated"].predict_proba(row[model["feature_cols"]])[0, 1])
    percentile = compute_percentile(probability, model["reference"])
    contributions = explain_prediction(model["base"], model["feature_cols"], row, top_n=5)
    return PredictionResponse(
        probability=probability,
        calibrated=True,
        feature_set=features.feature_set,
        percentile=percentile,
        model_version=model["model_version"],
        top_contributors=[ShapContribution(**c) for c in contributions],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded="clean" in ml_state and "full" in ml_state)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    clean, full = ml_state["clean"]["metadata"], ml_state["full"]["metadata"]
    return ModelInfoResponse(
        default_feature_set=DEFAULT_FEATURE_SET,
        clean_model_version=ml_state["clean"]["model_version"],
        full_model_version=ml_state["full"]["model_version"],
        clean_test_roc_auc=clean["roc_auc_after"],
        full_test_roc_auc=full["roc_auc_after"],
        clean_test_brier=clean["brier_after"],
        full_test_brier=full["brier_after"],
        leakage_note=(
            "The full model's higher AUC comes from funding_total_usd/funding_rounds/funding_span_days, "
            "all measured after the outcome resolved - see MODEL_CARD.md. The clean model is the default "
            "for a reason: it's the only one honest for scoring an active, unresolved company."
        ),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StartupFeatures) -> PredictionResponse:
    return _predict_one(features)


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchPredictionRequest) -> list[PredictionResponse]:
    return [_predict_one(f) for f in request.items]
