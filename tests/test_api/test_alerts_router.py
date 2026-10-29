"""Tests for ``src.api.routers.alerts`` (stockout rule, CL alerts, store check, rules)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.routers import alerts as alerts_router
from tests.test_api.conftest import FakeTensor

pytestmark = pytest.mark.usefixtures("isolated_registry")

ALERTS = "src.api.routers.alerts"


def test_alert_rules_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/alerts/rules")
    assert response.status_code == 200
    assert [rule["type"] for rule in response.json()] == ["STOCKOUT_RISK", "CL_DEGRADATION"]


@pytest.mark.parametrize(
    ("forecast", "severity", "message"),
    [
        ([40.0, 50.0], None, None),  # 90 units == 90% of stock -> no alert
        ([60.0, 60.0, 999.0], "MEDIUM", "Store 2: projected deficit 20 units in 2 days."),
        ([100.0, 100.0], "HIGH", "Store 2: projected deficit 100 units in 2 days."),
    ],
    ids=["below_threshold", "medium_deficit", "high_deficit"],
)
def test_check_stockout_risk_thresholds(
    forecast: list[float], severity: str | None, message: str | None
) -> None:
    alert = alerts_router._check_stockout_risk(2, current_stock=100.0, forecast=forecast)

    if severity is None:
        assert alert is None
        return
    assert alert is not None
    assert alert.severity == severity
    assert alert.message == message
    assert alert.alert_type == "STOCKOUT_RISK"
    assert alert.store_id == 2
    assert alert.alert_id.startswith("STOCKOUT-2-")


def test_check_stockout_risk_near_capacity_reports_demand_not_negative_deficit() -> None:
    """Regression: 90-100% stock usage is a MEDIUM alert without a negative deficit."""
    alert = alerts_router._check_stockout_risk(4, current_stock=100.0, forecast=[45.0, 50.0])

    assert alert is not None
    assert alert.severity == "MEDIUM"
    assert alert.message == (
        "Store 4: projected demand 95 units in 2 days uses over 90% of current stock (100 units)."
    )


@pytest.mark.parametrize(
    ("cl", "expected"),
    [
        ({}, []),
        ({"bwt": 0.0, "intransigence_mean": 0.1}, []),
        ({"bwt": -0.2}, [("CL-DEGRADATION-001", "HIGH")]),
        ({"intransigence_mean": 0.3}, [("CL-INTRANSIGENCE-CRITICAL", "CRITICAL")]),
        (
            {"bwt": -0.5, "intransigence_mean": 0.9},
            [("CL-DEGRADATION-001", "HIGH"), ("CL-INTRANSIGENCE-CRITICAL", "CRITICAL")],
        ),
    ],
)
def test_active_alerts_from_cl_metrics(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    cl: dict[str, float],
    expected: list[tuple[str, str]],
) -> None:
    monkeypatch.setattr(f"{ALERTS}.get_cl_metrics", lambda: cl)

    response = client.get("/api/v1/alerts/active")

    assert response.status_code == 200
    payload = response.json()
    assert [(a["alert_id"], a["severity"]) for a in payload] == expected
    assert all(a["alert_type"] == "CL_DEGRADATION" and a["store_id"] is None for a in payload)


def test_check_store_alerts_503_when_model_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(f"{ALERTS}.get_tft_model", lambda: (None, None, None))
    response = client.post("/api/v1/alerts/check/1")
    assert response.status_code == 503
    assert "TFT model not loaded" in response.json()["detail"]


@pytest.mark.parametrize(
    ("exc", "status", "detail"),
    [
        (KeyError("Store 9999 not found"), 404, "Store 9999 not found"),
        (RuntimeError("shape mismatch"), 503, "Forecast generation failed: shape mismatch"),
    ],
    ids=["unknown_store", "forecast_failure"],
)
def test_check_store_alerts_maps_forecast_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, exc: Exception, status: int, detail: str
) -> None:
    class _Model:
        def predict_for_store(self, **kwargs: Any) -> Any:
            raise exc

    monkeypatch.setattr(f"{ALERTS}.get_tft_model", lambda: (_Model(), object(), object()))
    response = client.post("/api/v1/alerts/check/9999", params={"current_stock": 25.0})
    assert response.status_code == status
    assert detail in response.json()["detail"]


@pytest.mark.parametrize(
    ("predictions", "expected"),
    [
        (FakeTensor(np.tile([[1.0, 60.0, 2.0]], (7, 1))), ["MEDIUM"]),  # tensor-like output
        (np.tile([[1.0, 10.0, 2.0]], (7, 1)), []),  # plain array output
    ],
    ids=["tensor_alert", "array_no_alert"],
)
def test_check_store_alerts_evaluates_p50_forecast(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, predictions: Any, expected: list[str]
) -> None:
    seen: dict[str, Any] = {}

    class _Model:
        def predict_for_store(self, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return predictions, []

    monkeypatch.setattr(f"{ALERTS}.get_tft_model", lambda: (_Model(), "ds", "ref"))
    monkeypatch.setattr(f"{ALERTS}.get_tft_quantile_indices", lambda: {"p50": 1})

    response = client.post("/api/v1/alerts/check/12", params={"current_stock": 100.0})

    assert response.status_code == 200
    payload = response.json()
    assert [a["severity"] for a in payload] == expected
    assert all(a["store_id"] == 12 for a in payload)
    assert seen == {"store_id": 12, "horizon": 7, "training_dataset": "ds", "reference_df": "ref"}
