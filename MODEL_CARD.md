# Model Card: Startup Outcome Classifiers

## Headline metric: ROC-AUC / PR-AUC, not accuracy

This dataset's positive rate is **59.83%** on the train cohort and **26.71%** on the time-separated test cohort (see `DATA_CARD.md`). Accuracy on an imbalanced, non-stationary target like this rewards a classifier for leaning toward whichever class happened to be more common in a given split — a `DummyClassifier(strategy="most_frequent")` scores an accuracy that shifts with the base rate and tells you nothing about discriminative skill. Every number below is ROC-AUC, average precision (PR-AUC), Brier score, and a confusion matrix at an explicitly chosen threshold. Accuracy is not reported.

## Models trained

All five models below share the same preprocessing pattern — median imputation for numeric features, constant-fill + encoding for categoricals — fit **inside an sklearn `Pipeline`/`ColumnTransformer`, on the train split only**, never on test. `LogisticRegression` uses one-hot encoding for categoricals (linear-model-appropriate) and standardizes numeric features; `HistGradientBoostingClassifier` uses ordinal encoding (`handle_unknown="use_encoded_value", unknown_value=-1`; tree-appropriate, avoids one-hot blowup and sklearn-version-dependent native categorical handling) and no scaling. Neither model uses class rebalancing (`class_weight`) — metrics reflect the natural, imbalanced distribution, not an artificially rebalanced one.

- `dummy_most_frequent`: `DummyClassifier(strategy="most_frequent")` — the floor.
- `logistic_regression_full`: `LogisticRegression(max_iter=2000, random_state=42)` on the **full** feature set (`founded_year`, `time_to_first_funding_days`, `funding_total_usd_log1p`, `funding_rounds`, `funding_span_days`, `country_code`, `region`, `primary_category`).
- `logistic_regression_clean`: same model, **clean** feature set only (`founded_year`, `time_to_first_funding_days`, `country_code`, `region`, `primary_category` — no post-outcome funding fields).
- `hist_gradient_boosting_full`: `HistGradientBoostingClassifier(random_state=42)` on the full feature set.
- `hist_gradient_boosting_clean`: same model, clean feature set only.

Threshold for every model's confusion matrix is chosen by **maximizing F0.5 on 5-fold stratified out-of-fold predictions on the train cohort** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` + `cross_val_predict`), then applied once, fixed, to the test cohort. F0.5 (not F1) is the justified choice: in an early-stage screening use case, telling someone a startup "will succeed" when it won't (a false positive) is more costly than a missed "might succeed" (a false negative), so precision is weighted roughly twice recall.

## Measured metrics

| Model | Feature set | CV ROC-AUC (mean±std, 5×10 repeats) | Test ROC-AUC | Test PR-AUC | Test Brier | Threshold | Test confusion (tn / fp / fn / tp) | Test F0.5 |
|---|---|---|---|---|---|---|---|---|
| dummy_most_frequent | — | 0.5000 ± 0.0000 | 0.5000 | 0.2671 | 0.7329 | 0.01 | 0 / 1951 / 0 / 711 | 0.3130 |
| logistic_regression_full | full | 0.7777 ± 0.0094 | **0.8496** | 0.6522 | 0.1356 | 0.63 | 1881 / 70 / 503 / 208 | 0.5705 |
| logistic_regression_clean | clean | 0.7272 ± 0.0093 | 0.8022 | 0.5463 | 0.1542 | 0.60 | 1854 / 97 / 547 / 164 | 0.4672 |
| hist_gradient_boosting_full | full | 0.7880 ± 0.0094 | 0.8339 | 0.6564 | 0.1444 | 0.65 | 1865 / 86 / 464 / 247 | 0.6045 |
| hist_gradient_boosting_clean | clean | 0.7363 ± 0.0096 | 0.7983 | 0.5476 | 0.1635 | 0.63 | 1844 / 107 / 550 / 161 | 0.4515 |

(n_train = 10,672, n_test = 2,662 for every model. Full precision/recall/F1/F0.5 and calibration bins for each model are in `reports/metrics/<model_name>_metrics.json`; comparison curves are in `reports/figures/roc_comparison.png`, `pr_comparison.png`, `calibration_comparison.png`. Numbers here were regenerated after a small `founded_at` plausibility-check bug fix documented in `DATA_CARD.md`'s "Correction" note - the fix moved AUC values by ≤0.0003 and changed no confusion matrix.)

Raw finding, not smoothed over: **`hist_gradient_boosting_full` (0.8339) underperforms `logistic_regression_full` (0.8496) on the held-out test set**, and the clean-feature versions are essentially tied (0.7983 vs 0.8022). With only 5-8 features and largely monotonic relationships (more funding, more rounds, older company → more likely resolved positively), a regularized linear model's smoother decision boundary appears to generalize across the time gap at least as well as an untuned gradient-boosted tree ensemble, which has more capacity to fit patterns specific to the training era that don't carry forward. Neither model was hyperparameter-tuned; this is a baseline comparison, not a claim that trees can't win here with tuning.

## CV vs. holdout: both numbers, and why they don't move the way you'd expect

Every non-dummy model's **test ROC-AUC is higher than its CV ROC-AUC mean** (e.g. logistic regression full: 0.7777 CV vs 0.8496 test, a +0.072 gap in the "good" direction). That is the opposite of the naive expectation that a later, harder-to-predict cohort should score worse. ROC-AUC is a ranking metric that doesn't depend on the positive rate, and the most likely explanation is that the CV folds are drawn from the *entire*, more heterogeneous train cohort (companies founded anywhere from 1901 to 2011-12-01), while the test cohort is a comparatively narrow, more homogeneous post-2011-12-01 window — a narrower window can be easier to rank correctly even though its base rate is lower.

**PR-AUC and Brier score do move the way you'd expect**, and for a simpler reason: both are sensitive to the positive rate. Average precision's floor is the positive rate itself (0.2671 on test vs a much higher implied floor on the 59.83%-positive train folds), so a test PR-AUC of 0.65 on a 26.71%-positive set and a CV average-precision mean of 0.83 on a ~59.83%-positive set are not directly comparable numbers — the drop is largely a base-rate effect, not evidence the model got worse at ranking.

**Report both, always**: the CV mean±std tells you how stable the model is across resamples of the training era; the single test number tells you how it performs on real forward-in-time data, which is what matters for actual deployment. Neither number alone is the "real" performance.

## Clean vs. full: the leakage finding

| Model family | Full AUC | Clean AUC | Gap (full − clean) |
|---|---|---|---|
| Logistic regression | 0.8496 | 0.8022 | **0.0474** |
| HistGradientBoosting | 0.8339 | 0.7983 | **0.0356** |

Both model families lose roughly 3.6-4.7 ROC-AUC points when `funding_total_usd_log1p`, `funding_rounds`, and `funding_span_days` are removed — consistent with the diagnostic probe in `DATA_CARD.md` (0.0474 measured there with a similar logistic-regression setup on Step 1's data pipeline). **This is the finding, not an artifact**: those three columns are recorded at scrape time, after the outcome (acquisition, IPO, or closure) is already known, so a portion of the "full" model's apparent skill is unavailable in any real forward-looking use case. The clean model — weaker, and honestly so — is the only one of the two that answers the question "can we predict this company's outcome from information available early in its life."

## Context against external benchmarks

The spec that produced this model card cited published work reporting ROC-AUC around 0.86 on 34,000 Crunchbase companies, and benchmarks at a 0.78% positive rate treating F0.5 = 0.097 as a good result. **Neither is directly comparable to the numbers above.** Our labeled set is 13,334 rows (34k companies with a labelable, non-`operating` status was not what we observed here), and our test-cohort positive rate is 26.71% — two orders of magnitude higher than 0.78%. A dataset that imbalanced makes even a small positive predictive value look impressive relative to the floor; ours is far less imbalanced, so direct comparison of raw AUC or F0.5 values across the two settings would be misleading. Our `logistic_regression_full` test ROC-AUC of 0.8496 lands in the same neighborhood as the cited 0.86, which is a reasonable sanity check that nothing here is obviously broken — but it is not evidence of matching that study's setup, sample, or label definition.

## What this model cannot do

- **Cannot score an active, unresolved company honestly with the full model.** `funding_total_usd_log1p`, `funding_rounds`, and `funding_span_days` are only fully known once a company's story is over (or at least once the Crunchbase scrape happened after the fact). Scoring a currently-operating company with `hist_gradient_boosting_full` or `logistic_regression_full` in production would require feeding it its *current* funding numbers, which is a different (and easier) prediction problem than the one it was evaluated on here. Only the clean-feature models are honest for live screening, and they measurably underperform (0.80 vs 0.83-0.85 AUC).
- **Was never trained or evaluated on `operating` companies at all.** Per `DATA_CARD.md`, all 53,034 `operating` rows were dropped before training, because their outcome hasn't resolved. The model has no exposure to what an unresolved company's features look like relative to the two outcomes it was trained on; applying it to one is an extrapolation the training data provides no guarantee about.
- **Degrades on data from further outside the training window without re-validation.** The CV-vs-test section above shows real movement in both directions even within `2011-12-01` to `2015-12-05`; there's no basis here for claiming the model holds up on 2020s-era companies.
- **Calibration is approximate, not guaranteed.** `reports/figures/calibration_comparison.png` shows the real models tracking the diagonal reasonably well in the mid-range but drifting at the extremes (see the plot before trusting a specific probability value, e.g. "73% chance of success," at face value).
- **Inherits the dataset's survivorship bias.** As documented in `DATA_CARD.md`, Crunchbase under-reports failures. Even the clean model's implied success rate is calibrated against an optimistic label distribution, not the true population of all startups ever founded.
