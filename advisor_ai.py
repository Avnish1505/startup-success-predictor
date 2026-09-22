"""Streamlit-facing wrapper around the local facts + retrieval advisor.
No external API, no network call, no API key."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.advisor.facts import assemble_facts_bundle, compute_cohort_stats
from src.advisor.response import build_advisor_response
from src.advisor.retrieval import load_index
from src.models.confidence import lookup_confidence_band

PRODUCTION_DIR = Path("models/production")
DATA_PATH = Path("data/processed/startups_features_v1.parquet")


@st.cache_resource
def _load_advisor_resources():
    index = load_index(PRODUCTION_DIR / "advisor_index")
    confidence_bands = json.loads((PRODUCTION_DIR / "confidence_bands.json").read_text())
    cohort_df = pd.read_parquet(DATA_PATH)
    return index, confidence_bands, cohort_df


def _build_prediction_context() -> dict | None:
    if "prediction_prob" not in st.session_state or "prediction_input" not in st.session_state:
        return None

    _, confidence_bands, cohort_df = _load_advisor_resources()
    prob_fraction = st.session_state.prediction_prob / 100.0
    snap = st.session_state.prediction_input.iloc[0]

    confidence_band = lookup_confidence_band(prob_fraction, confidence_bands)
    funding_total_usd = float(np.expm1(snap["funding_total_usd_log1p"]))
    cohort_stats = compute_cohort_stats(
        cohort_df,
        country_code=snap["country_code"],
        primary_category=snap["primary_category"],
        funding_total_usd=funding_total_usd,
    )
    shap_contributions = st.session_state.get("prediction_shap_contributions", [])
    percentile = st.session_state.get("prediction_percentile", 0.0)

    return assemble_facts_bundle(
        inputs={
            "country_code": snap["country_code"],
            "primary_category": snap["primary_category"],
            "funding_total_usd": funding_total_usd,
        },
        probability=prob_fraction,
        confidence_band=confidence_band,
        shap_contributions=shap_contributions,
        percentile=percentile,
        cohort_stats=cohort_stats,
    )


def startup_advice(question: str) -> str:
    index, _, _ = _load_advisor_resources()
    facts_bundle = _build_prediction_context()
    return build_advisor_response(question, facts_bundle, index)
