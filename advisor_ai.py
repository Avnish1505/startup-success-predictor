"""Streamlit-facing wrapper around the local facts + retrieval advisor.
No external API, no network call, no API key."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.advisor.facts import assemble_facts_bundle, compute_cohort_stats
from src.advisor.response import build_advisor_response
from src.advisor.retrieval import load_index
from src.models.confidence import bootstrap_confidence_band

PRODUCTION_DIR = Path("models/production")
DATA_PATH = Path("data/processed/startups_features_v1.parquet")
MODEL_NAMES = {"clean": "hist_gradient_boosting_clean_calibrated", "full": "hist_gradient_boosting_full_calibrated"}


@st.cache_resource
def _load_advisor_resources():
    index = load_index(PRODUCTION_DIR / "advisor_index")
    cohort_df = pd.read_parquet(DATA_PATH)
    models = {}
    for feature_set, name in MODEL_NAMES.items():
        base = joblib.load(PRODUCTION_DIR / f"{name}_base.joblib")
        calib_data = np.load(PRODUCTION_DIR / f"{name}_calibration_raw_scores.npz")
        metadata = json.loads((PRODUCTION_DIR / f"{name}_metadata.json").read_text())
        models[feature_set] = {
            "base": base, "raw_scores": calib_data["raw_score"], "labels": calib_data["label"],
            "method": metadata["calibration_method"],
            "feature_cols": metadata["numeric_features"] + metadata["categorical_features"],
        }
    return index, models, cohort_df


def _build_prediction_context() -> dict | None:
    if "prediction_prob" not in st.session_state or "prediction_input" not in st.session_state:
        return None

    _, models, cohort_df = _load_advisor_resources()
    feature_set = st.session_state.get("prediction_feature_set", "clean")
    model = models[feature_set]

    prob_fraction = st.session_state.prediction_prob / 100.0
    snap = st.session_state.prediction_input.iloc[0]
    raw_score = float(model["base"].predict_proba(st.session_state.prediction_input[model["feature_cols"]])[0, 1])

    try:
        confidence_band = bootstrap_confidence_band(
            raw_score, model["raw_scores"], model["labels"], model["method"], point_estimate=prob_fraction, n_bootstrap=200,
        )
    except AssertionError:
        confidence_band = (prob_fraction, prob_fraction)

    funding_total_usd = float(np.expm1(snap["funding_total_usd_log1p"])) if "funding_total_usd_log1p" in snap.index else None
    cohort_stats = (
        compute_cohort_stats(cohort_df, snap["country_code"], snap["primary_category"], funding_total_usd)
        if funding_total_usd is not None else None
    )
    shap_contributions = st.session_state.get("prediction_shap_contributions", [])
    percentile = st.session_state.get("prediction_percentile", 0.0)

    return assemble_facts_bundle(
        inputs={"country_code": snap["country_code"], "primary_category": snap["primary_category"]},
        probability=prob_fraction,
        raw_probability=raw_score,
        confidence_band=confidence_band,
        shap_contributions=shap_contributions,
        percentile=percentile,
        cohort_stats=cohort_stats,
    )


def startup_advice(question: str) -> str:
    index, _, _ = _load_advisor_resources()
    facts_bundle = _build_prediction_context()
    return build_advisor_response(question, facts_bundle, index)
