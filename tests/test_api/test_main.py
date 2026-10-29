"""Unit tests for FastAPI app endpoints."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.model_registry import clear_registry, load_rl_agent


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Reset model registry around each API test."""
    clear_registry()
    yield
    clear_registry()


def test_health_check() -> None:
    """Verify health endpoint responds with status ok."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert "registry" in payload


def test_request_id_header_attached() -> None:
    """All API responses should include x-request-id header."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("x-request-id")


def test_request_id_echoes_when_provided() -> None:
    """Middleware should preserve caller-provided x-request-id."""
    client = TestClient(app)
    response = client.get("/health", headers={"x-request-id": "abc-123"})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "abc-123"


def test_kpi_endpoint() -> None:
    """Verify KPI endpoint response shape."""
    client = TestClient(app)
    response = client.get("/api/v1/metrics/kpis")
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        payload = response.json()
        assert "smape" in payload
        assert "service_level" in payload
        assert "fill_rate" in payload
        assert "total_cost" in payload


def test_cl_metrics_endpoint() -> None:
    """Verify CL metrics endpoint response keys."""
    client = TestClient(app)
    response = client.get("/api/v1/metrics/cl")
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        payload = response.json()
        assert "bwt" in payload
        assert "fwt" in payload


def test_stores_endpoint() -> None:
    """Verify stores endpoint returns a large store list payload."""
    client = TestClient(app)
    response = client.get("/api/v1/metrics/stores")
    assert response.status_code == 200
    payload = response.json()
    assert "stores" in payload
    assert "forecastable_stores" in payload
    assert "total" in payload
    assert "forecastable_total" in payload
    assert payload["total"] >= 1100


def test_legacy_and_v1_prefixes_both_work_for_stores() -> None:
    """Stores endpoint should remain available on both API prefixes."""
    client = TestClient(app)
    legacy = client.get("/api/metrics/stores")
    v1 = client.get("/api/v1/metrics/stores")

    assert legacy.status_code == 200
    assert v1.status_code == 200

    legacy_payload = legacy.json()
    v1_payload = v1.json()
    assert legacy_payload["total"] == v1_payload["total"]
    assert legacy_payload["forecastable_total"] == v1_payload["forecastable_total"]


def test_inventory_action_endpoint_with_loaded_agent() -> None:
    """Verify inventory endpoint returns action payload when agent is loaded."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")
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


def test_simulation_endpoint_with_loaded_agent() -> None:
    """Verify simulation endpoint returns rollout summary when agent is loaded."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")
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
    assert "total_cost" in payload
    assert "risk_aversion" in payload


def test_simulate_batch_endpoint_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch simulation endpoint should return per-store metric payloads."""

    monkeypatch.setattr(
        "src.api.routers.metrics.run_rl_simulation_metrics_async",
        lambda store_ids, n_days, risk_aversion, n_mc, request_id=None: __import__("asyncio").sleep(
            0,
            result=[
                {
                    "store_id": float(sid),
                    "service_level": 0.95,
                    "fill_rate": 0.97,
                    "total_cost": 1234.5,
                    "risk_aversion": 0.7,
                }
                for sid in store_ids
            ],
        ),
    )

    client = TestClient(app)
    response = client.get(
        "/api/v1/metrics/simulate/batch",
        params={"store_ids": "1,2", "n_days": 7, "risk_aversion": 0.7},
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 2


def test_simulate_single_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single simulation endpoint should expose risk_aversion passthrough."""
    monkeypatch.setattr(
        "src.api.routers.metrics.run_rl_simulation_metrics",
        lambda n_days, store_id, risk_aversion, n_mc, request_id=None: {
            "service_level": 0.94,
            "fill_rate": 0.96,
            "total_cost": 1111.0,
            "risk_aversion": risk_aversion,
        },
    )

    client = TestClient(app)
    response = client.get(
        "/api/v1/metrics/simulate/single", params={"store_id": 1, "n_days": 7, "risk_aversion": 0.8}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_aversion"] == 0.8
