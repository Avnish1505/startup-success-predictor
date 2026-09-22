import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.models.explain import explain_prediction, get_explainer


def _toy_pipeline():
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.uniform(0, 10, n)  # strongly, monotonically predictive
    x2 = rng.uniform(0, 10, n)  # noise
    y = (x1 + rng.normal(0, 0.5, n) > 5).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "cat": rng.choice(["a", "b"], n), "label": y})

    preprocessor = ColumnTransformer([
        ("num", "passthrough", ["x1", "x2"]),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["cat"]),
    ])
    pipeline = Pipeline([("preprocess", preprocessor), ("clf", HistGradientBoostingClassifier(random_state=0))])
    pipeline.fit(df[["x1", "x2", "cat"]], df["label"])
    return pipeline


def test_get_explainer_is_cached():
    pipeline = _toy_pipeline()
    e1 = get_explainer(pipeline)
    e2 = get_explainer(pipeline)
    assert e1 is e2


def test_explain_prediction_returns_sorted_signed_contributions():
    pipeline = _toy_pipeline()
    row = pd.DataFrame([{"x1": 8.0, "x2": 5.0, "cat": "a"}])
    result = explain_prediction(pipeline, ["x1", "x2", "cat"], row, top_n=3)
    assert len(result) == 3
    mags = [abs(r["shap_value"]) for r in result]
    assert mags == sorted(mags, reverse=True)
    assert {"feature", "value", "shap_value", "direction"} <= result[0].keys()


def test_explain_prediction_direction_matches_expected_relationship():
    # x1 drives the label positively - a high x1 should push the SHAP bar toward "increases"
    pipeline = _toy_pipeline()
    high = explain_prediction(pipeline, ["x1", "x2", "cat"], pd.DataFrame([{"x1": 9.5, "x2": 5.0, "cat": "a"}]), top_n=3)
    low = explain_prediction(pipeline, ["x1", "x2", "cat"], pd.DataFrame([{"x1": 0.5, "x2": 5.0, "cat": "a"}]), top_n=3)
    high_x1 = next(r for r in high if r["feature"] == "x1")
    low_x1 = next(r for r in low if r["feature"] == "x1")
    assert high_x1["shap_value"] > low_x1["shap_value"]
    assert high_x1["direction"] == "increases"
    assert low_x1["direction"] == "decreases"
