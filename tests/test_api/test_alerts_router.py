"""Unit tests for alerts router endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


def test_alert_rules_endpoint() -> None:
    """Ensure alert rules endpoint returns rule definitions."""
    client = TestClient(app)
    response = client.get("/api/v1/alerts/rules")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert any(item.get("type") == "STOCKOUT_RISK" for item in payload)


def test_check_store_alerts_endpoint() -> None:
    """Ensure store check endpoint returns list payload or model-unavailable 503."""
    client = TestClient(app)
    response = client.post("/api/v1/alerts/check/1", params={"current_stock": 10.0})
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        payload = response.json()
        assert isinstance(payload, list)


def test_check_store_alerts_404_for_unknown_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Alerts endpoint should surface unknown store as 404."""

    class _Model:
        quantile_indices = {"p50": 0}

        def predict_for_store(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise KeyError("Store 9999 not found")

    monkeypatch.setattr(
        "src.api.routers.alerts.get_tft_model",
        lambda: (_Model(), object(), object()),
    )

    client = TestClient(app)
    response = client.post("/api/v1/alerts/check/9999", params={"current_stock": 25.0})
    assert response.status_code == 404
