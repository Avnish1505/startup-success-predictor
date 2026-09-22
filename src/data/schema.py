"""Static schema constants for the Crunchbase startup outcome dataset."""

KAGGLE_DATASET = "yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase"
DATA_VERSION = "v1"

RAW_COLUMNS = [
    "permalink", "name", "homepage_url", "category_list", "funding_total_usd",
    "status", "country_code", "state_code", "region", "city",
    "funding_rounds", "founded_at", "first_funding_at", "last_funding_at",
]

STATUS_ACQUIRED = "acquired"
STATUS_IPO = "ipo"
STATUS_CLOSED = "closed"
STATUS_OPERATING = "operating"
VALID_STATUSES = {STATUS_ACQUIRED, STATUS_IPO, STATUS_CLOSED, STATUS_OPERATING}

# operating is intentionally absent: it is dropped, never labeled.
LABEL_MAP = {STATUS_ACQUIRED: 1, STATUS_IPO: 1, STATUS_CLOSED: 0}

# Measured at scrape time, after the outcome is known. Full-set only.
LEAKY_COLUMNS = ["funding_total_usd", "funding_rounds", "last_funding_at"]

CLEAN_NUMERIC_FEATURES = ["founded_year", "time_to_first_funding_days"]
CLEAN_CATEGORICAL_FEATURES = ["country_code", "region", "primary_category"]

FULL_NUMERIC_FEATURES = CLEAN_NUMERIC_FEATURES + [
    "funding_total_usd_log1p", "funding_rounds", "funding_span_days",
]
FULL_CATEGORICAL_FEATURES = list(CLEAN_CATEGORICAL_FEATURES)

MIN_FOUNDED_YEAR = 1900
