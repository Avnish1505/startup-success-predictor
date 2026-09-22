import pytest
from fastapi.testclient import TestClient

from api import app

KNOWN_INPUT = {
    "founded_year": 2013,
    "time_to_first_funding_days": 151,
    "funding_total_usd": 1_000_000,
    "funding_rounds": 2,
    "funding_span_days": 214,
    "country_code": "USA",
    "region": "SF Bay Area",
    "primary_category": "Software",
}
KNOWN_PROBABILITY = 0.42857142857142855


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["calibration_method"] == "isotonic"
    assert body["n_train_fit"] > 0
    assert 0.0 <= body["roc_auc_after_calibration"] <= 1.0


def test_predict_known_input_regression(client):
    response = client.post("/predict", json=KNOWN_INPUT)
    assert response.status_code == 200
    body = response.json()
    assert body["probability"] == pytest.approx(KNOWN_PROBABILITY, abs=0.005)
    assert body["calibrated"] is True
    assert 0.0 <= body["percentile"] <= 100.0
    assert body["model_version"].startswith("hist_gradient_boosting_full_calibrated@")
    assert len(body["top_contributors"]) == 5
    for contributor in body["top_contributors"]:
        assert contributor["direction"] in ("increases", "decreases")


def test_predict_rejects_invalid_funding_rounds(client):
    bad_input = dict(KNOWN_INPUT, funding_rounds=0)  # violates ge=1
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("funding_rounds" in str(err["loc"]) for err in detail)


def test_predict_rejects_missing_field(client):
    incomplete = dict(KNOWN_INPUT)
    del incomplete["country_code"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("country_code" in str(err["loc"]) for err in detail)


def test_predict_batch(client):
    response = client.post("/predict/batch", json={"items": [KNOWN_INPUT, KNOWN_INPUT]})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert body[0]["probability"] == pytest.approx(KNOWN_PROBABILITY, abs=0.005)


def test_predict_batch_rejects_oversized_list(client):
    oversized = {"items": [KNOWN_INPUT] * 101}
    response = client.post("/predict/batch", json=oversized)
    assert response.status_code == 422


def test_predict_batch_rejects_empty_list(client):
    response = client.post("/predict/batch", json={"items": []})
    assert response.status_code == 422
