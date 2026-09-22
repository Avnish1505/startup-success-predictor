from src.advisor.router import classify_intent, find_named_feature


def test_classify_greeting():
    for q in ["hi", "Hello", "hey there", "what can you do", "help"]:
        assert classify_intent(q) == "greeting"


def test_classify_about_prediction():
    for q in ["why is my score low", "what hurts me most", "why did I get this probability",
              "what's dragging my score down"]:
        assert classify_intent(q) == "about_prediction"


def test_classify_general_startup_question():
    for q in ["why do startups fail", "what is survivorship bias", "what is a series A round"]:
        assert classify_intent(q) == "general"


def test_classify_out_of_scope():
    for q in ["what's the weather today", "recommend a pizza recipe", "what is the capital of France"]:
        assert classify_intent(q) == "out_of_scope"


def test_find_named_feature_funding():
    assert find_named_feature("why is my funding hurting the score") == "funding_total_usd_log1p"


def test_find_named_feature_category():
    assert find_named_feature("does my category help or hurt") == "primary_category"


def test_find_named_feature_none_when_unmentioned():
    assert find_named_feature("why is my score low") is None
