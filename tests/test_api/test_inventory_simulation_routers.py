"""API tests for inventory and simulation routers."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.model_registry import clear_registry, load_rl_agent


def test_inventory_action_endpoint_returns_payload() -> None:
    """Inventory action endpoint should return order and confidence fields."""
    clear_registry()
    load_rl_agent(
        config_path="configs/rl_config.yaml", demand_series=np.ones(365, dtype=float) * 50.0
    )
    client = TestClient(app)
    response = client.post(
        "/api/v1/inventory/action",
        json={"store_id": 1, "current_stock": 120.0, "forecast": [50.0] * 7},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "order_qty" in payload
    assert "confidence" in payload


def test_simulation_endpoint_returns_steps() -> None:
    """Simulation endpoint should return trajectory step list."""
    clear_registry()
    load_rl_agent(
        config_path="configs/rl_config.yaml", demand_series=np.ones(365, dtype=float) * 50.0
    )
    client = TestClient(app)
    response = client.post(
        "/api/v1/simulate/",
        json={
            "store_id": 1,
            "n_days": 5,
            "initial_stock": 100.0,
            "algorithm": "PPO",
            "risk_aversion": 0.5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "steps" in payload
    assert isinstance(payload["steps"], list)
    assert "risk_aversion" in payload
