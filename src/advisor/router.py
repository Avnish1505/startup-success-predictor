"""Deterministic (keyword/pattern) intent classification for the advisor.
No LLM call - this project is offline by design (see README's AI Advisor
section) - a small, testable intent table is sufficient for the four
question types this advisor actually handles."""
from __future__ import annotations

import re

GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|yo)\b", r"\bwhat can you do\b", r"\bhelp\b", r"\bwho are you\b",
]

ABOUT_PREDICTION_PATTERNS = [
    r"\b(my|this) (score|prediction|probability|result)\b", r"\bwhy (is|did|does) my\b",
    r"\bwhat hurts?\b", r"\bwhat helps?\b", r"\bdragging\b", r"\bwhat.s wrong with\b",
]

# General startup/fundraising/model-limitation topics this corpus covers -
# keywords drawn from the six real corpus documents' actual subject matter.
GENERAL_TOPIC_KEYWORDS = [
    "startup", "startups", "fundraising", "funding round", "series a", "series b", "series c",
    "pre-seed", "seed round", "survivorship", "calibration", "calibrated", "leakage",
    "investor", "traction", "venture capital", "vc ",
]

FEATURE_KEYWORDS = {
    "funding_total_usd_log1p": ["funding", "money raised", "capital", "amount raised"],
    "funding_rounds": ["rounds", "number of rounds"],
    "funding_span_days": ["funding span", "time between rounds"],
    "founded_year": ["age", "old", "founded", "founding"],
    "time_to_first_funding_days": ["time to first funding", "first funding"],
    "country_code": ["country"],
    "region": ["region", "location"],
    "primary_category": ["category", "industry", "sector"],
}


def classify_intent(question: str) -> str:
    q = question.lower().strip()
    if any(re.search(p, q) for p in GREETING_PATTERNS):
        return "greeting"
    if any(re.search(p, q) for p in ABOUT_PREDICTION_PATTERNS):
        return "about_prediction"
    if any(kw in q for kw in GENERAL_TOPIC_KEYWORDS):
        return "general"
    return "out_of_scope"


def find_named_feature(question: str) -> str | None:
    q = question.lower()
    for feature, keywords in FEATURE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return feature
    return None
