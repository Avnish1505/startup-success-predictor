# Data Card: Startup Outcome Dataset

## Source

- Kaggle dataset: [`yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase`](https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase)
- Underlying source: Crunchbase company records, scraped circa late 2015 (the latest `first_funding_at` value in the data is 2015-12-05).
- Fetched via `kagglehub.dataset_download()` in `src/data/download.py`.
- License: **CDLA-Sharing-1.0** (Community Data License Agreement – Sharing, Version 1.0). Redistribution of derived data must remain under the same license and any modifications must be shared under the same terms.

## Raw shape

- **66,368 rows × 14 columns**: `permalink, name, homepage_url, category_list, funding_total_usd, status, country_code, state_code, region, city, funding_rounds, founded_at, first_funding_at, last_funding_at`.

## Labeling

- `status == acquired` or `status == ipo` → label `1`.
- `status == closed` → label `0`.
- `status == operating` → **row dropped**, not labeled 0. An operating company's outcome hasn't resolved yet; labeling it a failure is the single most common mistake in this dataset's public notebooks, and it poisons the target with censored (not-yet-failed) companies.
- Raw status breakdown: `operating 53,034` (74.1%, dropped) / `closed 6,238` (9.4%) / `acquired 5,549` (8.4%) / `ipo 1,547` (2.3%).

## Class balance (after dropping `operating`)

- **13,334 labeled rows**: positive (acquired+ipo) = **7,096 (53.2%)**, negative (closed) = **6,238 (46.8%)**.

⚠️ **Do not read 53.2% as "the real-world success rate of startups."** See the survivorship-bias caveat below.

## Leakage control

`funding_total_usd`, `funding_rounds`, and `last_funding_at` are all measured **at scrape time, after the outcome is already known** — an acquired company's `funding_total_usd` includes funding raised after (or as part of) the acquisition. Two feature sets are built in `src/data/build.py`:

- **clean** (5 features, none post-outcome): `founded_year`, `time_to_first_funding_days`, `country_code`, `region`, `primary_category`.
- **full** (8 features = clean + 3 leaky): adds `funding_total_usd_log1p`, `funding_rounds`, `funding_span_days` (`last_funding_at − first_funding_at`).

A logistic-regression probe (`ColumnTransformer` + median/constant imputation + one-hot encoding, fit on the train split only) was scored on the held-out time-based test split:

| Feature set | Test ROC AUC |
|---|---|
| clean | 0.8021 |
| full  | 0.8495 |
| **gap (full − clean)** | **+0.0474** |

**This gap is the finding.** ~4.7 points of AUC in the "full" model come from information that would not be available at prediction time in a real forward-looking use case (a model scoring an *active* company can't see its final, post-outcome funding total). Any model reporting AUC near 0.85 on this dataset using `funding_total_usd`/`funding_rounds`/`last_funding_at` is measuring leakage, not skill.

## Time-based split

- Split key: `founded_at` when plausible (year ≥ 1900 and not later than the dataset's own max observed funding date, used as a scrape-time proxy — 114 rows dataset-wide have corrupted `founded_at` values like `1015-01-30` or `2914-01-01`); falls back to `first_funding_at` otherwise (3,754 of 13,334 labeled rows, 28.2%, use the fallback — chiefly the 3,732 rows with null `founded_at`). Every labeled row has at least one usable date; zero rows are dropped for lacking both.
- Boundary (80th percentile of the effective date): **2011-12-01**.
- Train: 10,672 rows (older cohort), positive rate **59.8%**.
- Test: 2,662 rows (newer cohort), positive rate **26.7%**.
- All transformers (imputers, scaler, one-hot encoder) are fit on the train split only, inside an sklearn `Pipeline`.

Note the large positive-rate drop from train (59.8%) to test (26.7%). This is expected, not a bug: newer companies (post-2011-12) have had less time to resolve to `acquired`/`ipo`, which usually takes years, while `closed` can happen quickly. A model evaluated on this split is being tested on a genuinely harder, more realistic task — predicting outcomes for young companies — rather than an IID resample of the same era.

## Column notes

- `funding_total_usd` is a string column; `-` is the missing sentinel (2,191 of 13,334 labeled rows, 16.4%). No comma-thousands separators were found in this dataset version, but the coercion (`coerce_funding_total_usd` in `src/data/build.py`) still strips commas defensively and **raises `ValueError`** naming any value it can't parse after handling `-` and commas — it never silently coerces unexpected garbage to NaN.
- `category_list` is pipe-delimited multi-label (e.g. `"Apps|Games|Mobile"`). The first token is extracted as `primary_category` (506 unique values in the labeled subset — one-hot-able) and used as a feature. The full `category_list` (27,296 unique combinations in the labeled subset — **not** ~58,000; that was the spec's rough estimate, this is the measured value) is preserved in the processed parquet for future multi-hot encoding but is not one-hot encoded here.

## Output

`python -m src.data.build` writes `data/processed/startups_features_v1.parquet` (13,334 rows: all clean + full feature columns, `category_list`, `label`, `permalink`, `name`, and a `split` column of `train`/`test`).

## Survivorship-bias caveat

Crunchbase is a self-reported and crowd-maintained database. Failed startups are systematically **under-reported**: a company that quietly shuts down often just stops being updated rather than having its status changed to `closed`, and many failures never had a Crunchbase profile compelling enough to attract the same scrutiny as a funded, later-successful company. That means:

- The 53.2% positive rate among *resolved* (non-operating) companies in this dataset is almost certainly **optimistic** relative to the true failure rate of startups in general — commonly cited real-world estimates put startup failure well above what this dataset's `closed` count alone would suggest.
- Any model trained on this data learns "which resolved-and-recorded companies succeeded," not "which startups succeed." Treat probability outputs as relative rankings within this reporting bias, not calibrated real-world success probabilities.
