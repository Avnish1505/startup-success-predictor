from src.advisor.chunking import Chunk
from src.advisor.response import build_advisor_response
from src.advisor.retrieval import build_index


def _toy_index():
    return build_index([
        Chunk("doc::1", "doc", "Failure", "Startups fail when they run out of cash or have no product market fit.", "https://x"),
    ])


def test_greeting_returns_capability_statement_not_facts_bundle():
    index = _toy_index()
    facts_bundle = {"probability": 0.65, "raw_probability": 0.7, "confidence_band": (0.5, 0.8),
                     "top_contributions": [], "percentile": 50.0, "cohort": None, "inputs": {}}
    text = build_advisor_response("hi", facts_bundle, index)
    assert "calibrated probability" not in text.lower()  # facts bundle NOT dumped
    assert "ask" in text.lower() or "can" in text.lower()  # a capability statement


def test_off_topic_question_returns_nothing_from_retrieval():
    # regression test for item 6 - MIN_RETRIEVAL_SCORE already handles this,
    # locking it in so the router work in this task can't silently break it
    index = _toy_index()
    text = build_advisor_response("what's the weather today", None, index)
    assert "nothing" in text.lower() or "outside" in text.lower()


def test_about_prediction_with_no_facts_bundle_points_to_predictor_tab():
    index = _toy_index()
    text = build_advisor_response("why is my score low", None, index)
    assert "predictor" in text.lower()


def test_about_prediction_foregrounds_named_feature():
    index = _toy_index()
    facts_bundle = {
        "probability": 0.65, "raw_probability": 0.7, "confidence_band": (0.5, 0.8),
        "top_contributions": [
            {"feature": "primary_category", "value": "Software", "shap_value": 0.9, "direction": "increases"},
            {"feature": "funding_total_usd_log1p", "value": 13.8, "shap_value": -0.3, "direction": "decreases"},
        ],
        "percentile": 50.0, "cohort": None, "inputs": {},
    }
    text = build_advisor_response("why is my funding hurting the score", facts_bundle, index)
    assert text.index("funding_total_usd_log1p") < text.index("| Feature |")
