import numpy as np
import pandas as pd

from src.models.calibrate import choose_calibration_method, split_train_for_calibration


def test_choose_calibration_method_isotonic_when_large():
    assert choose_calibration_method(5000, threshold=1000) == "isotonic"


def test_choose_calibration_method_sigmoid_when_small():
    assert choose_calibration_method(200, threshold=1000) == "sigmoid"


def test_choose_calibration_method_boundary_is_isotonic():
    assert choose_calibration_method(1000, threshold=1000) == "isotonic"


def test_split_train_for_calibration_disjoint_and_stratified():
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({
        "label": rng.integers(0, 2, size=n),
        "x": rng.normal(size=n),
    })
    train_fit, calibration = split_train_for_calibration(df, calibration_frac=0.2, random_state=42)
    assert len(train_fit) + len(calibration) == n
    assert set(train_fit.index).isdisjoint(set(calibration.index))
    assert abs(len(calibration) / n - 0.2) < 0.02
    # stratification keeps positive rate close between the two splits
    assert abs(train_fit["label"].mean() - calibration["label"].mean()) < 0.05
