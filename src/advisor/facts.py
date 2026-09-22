"""Layer 1: deterministic facts core. Every number here is either passed in
already-computed (probability, SHAP contributions, percentile - the ML
already happened upstream) or looked up from the real, committed dataset
(cohort stats). Nothing here calls a model, an explainer, or a network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import schema


def compute_cohort_stats(
    df: pd.DataFrame,
    country_code: str,
    primary_category: str,
    funding_total_usd: float,
    min_n: int = schema.MIN_COHORT_SIZE,
) -> dict | None:
    funding_band = pd.cut(
        [funding_total_usd], bins=schema.FUNDING_BIN_EDGES, labels=schema.FUNDING_BIN_LABELS
    )[0]
    if "funding_band" not in df.columns:
        raw_funding = np.expm1(df["funding_total_usd_log1p"])
        df = df.assign(funding_band=pd.cut(raw_funding, bins=schema.FUNDING_BIN_EDGES, labels=schema.FUNDING_BIN_LABELS))

    mask = (
        (df["country_code"] == country_code)
        & (df["primary_category"] == primary_category)
        & (df["funding_band"] == funding_band)
    )
    n = int(mask.sum())
    if n < min_n:
        return None
    return {
        "country_code": country_code,
        "primary_category": primary_category,
        "funding_band": str(funding_band),
        "n": n,
        "success_rate": float(df.loc[mask, "label"].mean()),
    }


def assemble_facts_bundle(
    inputs: dict,
    probability: float,
    confidence_band: tuple[float, float],
    shap_contributions: list[dict],
    percentile: float,
    cohort_stats: dict | None,
) -> dict:
    return {
        "inputs": dict(inputs),
        "probability": probability,
        "confidence_band": confidence_band,
        "top_contributions": list(shap_contributions),
        "percentile": percentile,
        "cohort": cohort_stats,
    }


def render_facts_bundle(bundle: dict) -> str:
    lines = ["## What the model says"]
    lines.append(
        f"- Calibrated probability of a positive outcome (acquired/IPO): "
        f"**{bundle['probability']:.1%}** (90% confidence band: "
        f"{bundle['confidence_band'][0]:.1%}-{bundle['confidence_band'][1]:.1%})"
    )
    lines.append(f"- Percentile vs. the training cohort: **{bundle['percentile']:.1f}%**")

    if bundle["top_contributions"]:
        lines.append("- Top signed contributions (SHAP, on the pre-calibration model):")
        for c in bundle["top_contributions"]:
            arrow = "+" if c["direction"] == "increases" else "-"
            value = c["value"]
            value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
            lines.append(f"  - {arrow} `{c['feature']}` = {value_str} ({c['direction']} the score)")

    cohort = bundle["cohort"]
    if cohort:
        lines.append(
            f"- Matching cohort ({cohort['country_code']}, {cohort['primary_category']}, "
            f"{cohort['funding_band']}, n={cohort['n']}): observed success rate "
            f"**{cohort['success_rate']:.1%}** in the training data."
        )
    else:
        lines.append(
            "- Matching cohort: not enough comparable companies in the training data "
            "(n < 30) to report a reliable rate for this exact combination."
        )
    return "\n".join(lines)
