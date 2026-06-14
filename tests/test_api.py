"""Tests for the FastAPI prediction endpoints."""

from fastapi.testclient import TestClient

from src.api import main


class FakeModel:
    """Minimal model replacement used to test the API without loading a real artifact."""

    def predict(self, X):
        assert list(X.columns) == main.FEATURE_COLUMNS
        return [38.14]


def test_health_returns_ok():
    # Checks that the API app starts and exposes the health endpoint.
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_predicted_lead_time_days(monkeypatch):
    # Checks that valid JSON is converted into model input and returned as a prediction.
    monkeypatch.setattr(main, "load_model", lambda: FakeModel())

    client = TestClient(main.app)

    payload = {
        "distance_km": 14285.36,
        "weight_mt": 237.24,
        "fuel_price_index": 2.3,
        "geopolitical_risk_score": 7.5,
        "carrier_reliability_score": 0.592,
        "copper__usd_per_mt": 9464.43,
        "origin_port": "Singapore",
        "destination_port": "Shanghai",
        "transport_mode": "Rail",
        "product_category": "Automotive",
        "weather_condition": "Storm",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"predicted_lead_time_days": 38.14}