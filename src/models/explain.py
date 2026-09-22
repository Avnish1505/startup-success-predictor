"""Cached SHAP TreeExplainer over the boosted model, exposing top-N signed
contributions for a single prediction in human-readable feature names."""
from __future__ import annotations

import shap
from sklearn.pipeline import Pipeline

# Keyed by id(pipeline), but each entry also holds a strong reference to the
# pipeline itself. That reference does two things: it keeps the pipeline
# alive for as long as it's cached (so its id() can't be reused by an
# unrelated, later object while the entry is still present), and it lets us
# verify identity on lookup so a stale entry - e.g. one left over from an
# earlier object that has since been garbage collected and whose id() a new
# object happens to reuse - is never mistaken for a cache hit.
_EXPLAINER_CACHE: dict[int, tuple[Pipeline, shap.TreeExplainer]] = {}


def get_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    """Build (once) or reuse a TreeExplainer for this exact pipeline object."""
    key = id(pipeline)
    cached = _EXPLAINER_CACHE.get(key)
    if cached is not None and cached[0] is pipeline:
        return cached[1]
    explainer = shap.TreeExplainer(pipeline.named_steps["clf"])
    _EXPLAINER_CACHE[key] = (pipeline, explainer)
    return explainer


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
