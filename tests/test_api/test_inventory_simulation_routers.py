"""Tests for the inventory and simulation routers (base-stock policy endpoints)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.usefixtures("isolated_registry")


@pytest.fixture()
def reference_parquet(tmp_path: Path) -> None:
    """Store 1: mean 100, std ~28.28; store 2: constant 200 (zero std)."""
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame({"Store": [1, 1, 2, 2], "sales": [80.0, 120.0, 200.0, 200.0]}).to_parquet(
        processed / "tft_reference.parquet"
    )


@pytest.mark.usefixtures("reference_parquet")
@pytest.mark.parametrize(
    ("store_id", "stock", "forecast", "order_qty", "confidence"),
    [
        (1, 1500.0, [100.0] * 7, 0.0, 0.99),  # >= 14 days of cover
        (1, 800.0, [100.0] * 7, 0.0, 0.90),  # >= 7 days
        (1, 400.0, [100.0] * 7, 328.28, 0.75),  # target 7 * 100 + std 28.28
        (1, 0.0, [], 728.28, 0.55),  # empty forecast -> store mean demand
        (2, 0.0, [200.0], 1450.0, 0.55),  # zero std -> 25% of mean as safety stock
        (99, 0.0, [10.0], 210.0, 0.55),  # unknown store defaults; capped at 21 days of demand
    ],
)
def test_inventory_action_base_stock_policy(
    client: TestClient,
    store_id: int,
    stock: float,
    forecast: list[float],
    order_qty: float,
    confidence: float,
) -> None:
    response = client.post(
        "/api/v1/inventory/action",
        json={"store_id": store_id, "current_stock": stock, "forecast": forecast},
    )

    assert response.status_code == 200
    assert response.json() == {
        "store_id": store_id,
        "order_qty": order_qty,
        "confidence": confidence,
    }


def _simulate(client: TestClient, **overrides: float) -> dict:
    body = {"store_id": 1, "n_days": 10, "initial_stock": 100.0, "risk_aversion": 0.5}
    body.update(overrides)
    response = client.post("/api/v1/simulate/", json=body)
    assert response.status_code == 200
    return response.json()


@pytest.mark.usefixtures("reference_parquet")
def test_simulation_trajectory_uses_store_demand_profile(client: TestClient) -> None:
    payload = _simulate(client, initial_stock=1000.0, risk_aversion=0.3)

    steps = payload["steps"]
    assert payload["store_id"] == 1
    assert payload["risk_aversion"] == 0.3
    assert [step["day"] for step in steps] == list(range(1, 11))
    # Demand is drawn around store 1's mean of 100, not the 5000-unit default.
    assert all(0.0 <= step["demand"] < 300.0 for step in steps)
    # Excess opening stock: no order on day 1 and the day ends overstocked.
    assert steps[0]["order_qty"] == 0.0
    assert steps[0]["status"] == "Overstock"
    assert payload["total_cost"] == pytest.approx(-sum(s["reward"] for s in steps), abs=0.01)
    assert payload["service_level"] == pytest.approx(
        sum(s["service_level"] for s in steps) / len(steps), abs=1e-4
    )


@pytest.mark.usefixtures("reference_parquet")
def test_simulation_higher_risk_aversion_raises_service_level(client: TestClient) -> None:
    cost_focus = _simulate(client, store_id=2, n_days=30, risk_aversion=0.0)
    service_focus = _simulate(client, store_id=2, n_days=30, risk_aversion=1.0)

    assert service_focus["service_level"] > cost_focus["service_level"]
    assert service_focus["service_level"] == 1.0
    assert any(step["status"] == "Stockout" for step in cost_focus["steps"])
