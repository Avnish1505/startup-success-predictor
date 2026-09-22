"""Layer 1: deterministic facts core. Every number here is either passed in
already-computed (probability, SHAP contributions, percentile - the ML
already happened upstream) or looked up from the real, committed dataset
(cohort stats). Nothing here calls a model, an explainer, or a network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import schema


def _funding_band_mask(df: pd.DataFrame, funding_total_usd: float):
    funding_band = pd.cut(
        [funding_total_usd], bins=schema.FUNDING_BIN_EDGES, labels=schema.FUNDING_BIN_LABELS
    )[0]
    if "funding_band" not in df.columns:
        raw_funding = np.expm1(df["funding_total_usd_log1p"])
        df = df.assign(funding_band=pd.cut(raw_funding, bins=schema.FUNDING_BIN_EDGES, labels=schema.FUNDING_BIN_LABELS))
    return df["funding_band"] == funding_band, df


def compute_cohort_stats(
    df: pd.DataFrame,
    country_code: str,
    primary_category: str,
    funding_total_usd: float,
    min_n: int = schema.MIN_COHORT_SIZE,
) -> dict | None:
    """Progressive backoff: country+category+funding_band -> country+category
    -> country alone -> None. Always reports which level matched, so a wider
    match is shown with a stated caveat rather than refusing to answer."""
    funding_mask, df = _funding_band_mask(df, funding_total_usd)
    country_mask = df["country_code"] == country_code
    category_mask = df["primary_category"] == primary_category

    levels = [
        ("country+category+funding_band", country_mask & category_mask & funding_mask),
        ("country+category", country_mask & category_mask),
        ("country", country_mask),
    ]
    for level_name, mask in levels:
        n = int(mask.sum())
        if n >= min_n:
            return {
                "level": level_name,
                "country_code": country_code,
                "primary_category": primary_category if "category" in level_name else None,
                "n": n,
                "success_rate": float(df.loc[mask, "label"].mean()),
            }
    return None


def assemble_facts_bundle(
    inputs: dict,
    probability: float,
    raw_probability: float,
    confidence_band: tuple[float, float],
    shap_contributions: list[dict],
    percentile: float,
    cohort_stats: dict | None,
) -> dict:
    return {
        "inputs": dict(inputs),
        "probability": probability,
        "raw_probability": raw_probability,
        "confidence_band": confidence_band,
        "top_contributions": list(shap_contributions),
        "percentile": percentile,
        "cohort": cohort_stats,
    }


def _format_value(value) -> str:
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def render_facts_bundle(bundle: dict, foreground_feature: str | None = None) -> str:
    lines = ["## What the model says"]
    lines.append(
        f"- Calibrated probability of a positive outcome (acquired/IPO): "
        f"**{bundle['probability']:.1%}** (90% confidence band: "
        f"{bundle['confidence_band'][0]:.1%}-{bundle['confidence_band'][1]:.1%})"
    )
    lines.append(
        f"- Raw model score (pre-calibration, log-odds space on the uncalibrated model): "
        f"**{bundle['raw_probability']:.1%}** - the contributions below decompose this number, not the calibrated one above."
    )
    lines.append(f"- Percentile vs. the training cohort: **{bundle['percentile']:.1f}%**")

    contributions = bundle["top_contributions"]
    if contributions:
        if foreground_feature:
            match = next((c for c in contributions if c["feature"] == foreground_feature), None)
            if match:
                lines.append(
                    f"- You asked about `{foreground_feature}`: it {match['direction']} the score "
                    f"(SHAP {match['shap_value']:+.3f}, value = {_format_value(match['value'])})."
                )
        lines.append("")
        lines.append("| Feature | Value | SHAP | Direction |")
        lines.append("|---|---|---|---|")
        for c in contributions:
            lines.append(f"| `{c['feature']}` | {_format_value(c['value'])} | {c['shap_value']:+.3f} | {c['direction']} |")

    cohort = bundle["cohort"]
    if cohort:
        label = cohort["level"].replace("+", " + ").replace("_", " ")
        lines.append(
            f"- Matching cohort ({label} = {cohort['country_code']}"
            + (f", {cohort['primary_category']}" if cohort.get("primary_category") else "")
            + f", n={cohort['n']}): observed success rate **{cohort['success_rate']:.1%}** in the training data."
        )
    else:
        lines.append(
            "- Matching cohort: not enough comparable companies in the training data "
            "(n < 30 even at the country-alone level) to report a reliable rate."
        )
    return "\n".join(lines)
