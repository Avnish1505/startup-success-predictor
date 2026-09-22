"""Cached SHAP TreeExplainer over the boosted model, exposing top-N signed
contributions for a single prediction in human-readable feature names."""
from __future__ import annotations

import shap
from sklearn.pipeline import Pipeline

_EXPLAINER_CACHE: dict[int, shap.TreeExplainer] = {}


def get_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    """Build (once) or reuse a TreeExplainer for this exact pipeline object.

    Cached by id(pipeline) rather than lru_cache, since we only ever want to
    key on object identity - never on equality/hash semantics of sklearn
    estimators - and never rebuild the explainer per request.
    """
    key = id(pipeline)
    if key not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[key] = shap.TreeExplainer(pipeline.named_steps["clf"])
    return _EXPLAINER_CACHE[key]


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
