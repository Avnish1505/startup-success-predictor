---
title: "Independent Evidence on Leakage in Crunchbase-Based Startup-Outcome Models"
source_url: "https://doi.org/10.3390/info17070702"
retrieved: "2026-09-22"
---

## A Peer-Reviewed Study on the Same Scale of Data

"A Leakage-Controlled, Calibration-First Evaluation of Machine Learning Models for Startup-Outcome Prediction: Evidence from Crunchbase," published in *Information* (MDPI) in July 2026, evaluates startup-outcome classifiers under a leakage-controlled, calibration-first protocol on Crunchbase-derived data - including a large dataset of 66,368 firms, the same scale as (and very plausibly drawn from the same public Crunchbase snapshot as) the dataset this project uses (see `DATA_CARD.md`). The paper's motivating concern is explicit: machine learning work on startup outcomes frequently reports accuracy above 0.90, and whether that reflects genuine predictive signal available before the outcome, or a methodological artifact of features that leak information from after the outcome, is "unclear and consequential for investors, accelerators, and innovation-policy agencies."

## The Measured Leakage Effect

The paper's headline finding: removing outcome-correlated, survivorship-accumulating features lowers the area under the ROC curve by 0.05 to 0.09 on the large (66,368-firm) dataset, with every paired 95% confidence interval excluding zero - i.e. a real, statistically robust effect, not noise. On a smaller, more heavily hand-engineered feature set, the drop was larger still (0.19).

## Independent Corroboration of This Project's Own Finding

This project measured its own clean-vs-full AUC gap directly (`DATA_CARD.md`, `MODEL_CARD.md`): removing `funding_total_usd`, `funding_rounds`, and `funding_span_days` (fields recorded at scrape time, after the outcome is already known) dropped test-set AUC by 0.036-0.047 depending on model family. That is the same direction and a broadly similar order of magnitude to this independently published, peer-reviewed result on a comparably-sized Crunchbase dataset - two separate analyses, on data drawn from the same underlying source, landing on the same qualitative conclusion. That agreement is evidence the leakage effect is real and not an artifact of this project's specific pipeline; it is not proof the two studies used identical features, splits, or labeling rules, and the exact magnitudes should not be treated as directly comparable.

## What This Means for Any Number This Project Reports

Any AUC, precision, or recall figure quoting the *full* feature set (as opposed to the *clean* feature set) on this kind of data should be read as an upper bound on real-world, forward-looking predictive performance, not an estimate of it - see `MODEL_CARD.md`'s "what this model cannot do" section.
