---
title: "Probability Calibration for Classifiers"
source_url: "https://scikit-learn.org/stable/modules/calibration.html"
retrieved: "2026-09-22"
---

## What Calibration Means

A classifier's raw predicted probability and a *calibrated* probability are not automatically the same thing. A model is well-calibrated if, among all the times it predicts "70% chance of the positive class," roughly 70% of those cases actually turn out positive. Many classifiers - including gradient-boosted trees - can produce probability-shaped outputs that rank examples correctly (good discrimination, e.g. a strong ROC-AUC) while still being systematically over- or under-confident at the level of the actual numeric probability. `CalibratedClassifierCV`, part of scikit-learn, exists specifically to fix that second problem without touching the first.

## Sigmoid vs. Isotonic Calibration

`CalibratedClassifierCV` supports two calibration methods. Sigmoid calibration (Platt scaling) fits a simple logistic curve mapping raw scores to calibrated probabilities - it makes a strong parametric assumption about the shape of the miscalibration, which makes it robust with limited calibration data but less flexible. Isotonic calibration fits an arbitrary non-decreasing step function instead - more flexible, and able to correct more complex miscalibration patterns, but the scikit-learn documentation explicitly cautions that it "is not advised" with too few calibration samples (informally, well under 1,000), since with limited data it tends to overfit noise in the calibration set rather than the true underlying miscalibration.

## Relevance to This Project's Own Calibration Result

This project's calibration step (`src/models/calibrate.py`, documented in `MODEL_CARD.md`/`README.md`) used isotonic calibration on 2,135 held-out samples - comfortably above the documentation's rough `<<1000` overfitting-risk floor. Even so, the measured result was that calibration did not improve, and very slightly worsened, the Brier score on the test cohort (0.1433 before, 0.1500 after). This is a real, reported finding, not a bug: `HistGradientBoostingClassifier` is trained with log-loss, which already tends to produce reasonably well-calibrated probabilities out of the box, so there was less room for isotonic calibration to improve on, and with a moderately sized (not huge) calibration set, an unconstrained isotonic step function can add variance without a clear net benefit. The general lesson, independent of this specific project: calibration is worth *measuring*, with a real before/after comparison on held-out data, rather than assumed to help just because it was applied.
