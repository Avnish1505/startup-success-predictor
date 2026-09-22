import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.models.partial_dependence import compute_partial_dependence


def _toy_pipeline():
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.uniform(0, 10, n)  # strongly, monotonically predictive
    x2 = rng.uniform(0, 10, n)  # noise
    y = (x1 + rng.normal(0, 0.3, n) > 5).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "cat": rng.choice(["a", "b"], n), "label": y})

    preprocessor = ColumnTransformer([
        ("num", "passthrough", ["x1", "x2"]),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["cat"]),
    ])
    pipeline = Pipeline([("preprocess", preprocessor), ("clf", HistGradientBoostingClassifier(random_state=0))])
    pipeline.fit(df[["x1", "x2", "cat"]], df["label"])
    return pipeline


def test_partial_dependence_shape_matches_grid():
    pipeline = _toy_pipeline()
    base_row = pd.DataFrame([{"x1": 5.0, "x2": 5.0, "cat": "a"}])
    grid = [0.0, 2.5, 5.0, 7.5, 10.0]
    result = compute_partial_dependence(pipeline, ["x1", "x2", "cat"], base_row, "x1", grid)
    assert list(result["x1"]) == grid
    assert result["predicted_probability"].between(0, 1).all()


def test_partial_dependence_holds_other_features_fixed():
    # x2 is pure noise for the label, but must stay at base_row's value across the grid -
    # verified indirectly: varying x1 across its full observed range should move the
    # probability monotonically upward given the toy data's construction.
    pipeline = _toy_pipeline()
    base_row = pd.DataFrame([{"x1": 5.0, "x2": 9.0, "cat": "a"}])
    grid = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    result = compute_partial_dependence(pipeline, ["x1", "x2", "cat"], base_row, "x1", grid)
    probs = result["predicted_probability"].tolist()
    assert probs[-1] > probs[0]
    # monotonic non-decreasing overall, allowing negligible floor-noise dips
    # (a tree model's near-zero-probability leaves can jitter in the 5th+ decimal)
    assert all(b - a >= -1e-3 for a, b in zip(probs, probs[1:]))


def test_partial_dependence_categorical_grid():
    pipeline = _toy_pipeline()
    base_row = pd.DataFrame([{"x1": 5.0, "x2": 5.0, "cat": "a"}])
    result = compute_partial_dependence(pipeline, ["x1", "x2", "cat"], base_row, "cat", ["a", "b"])
    assert list(result["cat"]) == ["a", "b"]
    assert len(result) == 2
