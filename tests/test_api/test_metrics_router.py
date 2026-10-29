"""Tests for ``src.api.routers.metrics`` (CL metrics, KPIs, simulation endpoints, stores)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.usefixtures("isolated_registry")

METRICS = "src.api.routers.metrics"
ALL_STORES = list(range(1, 1116))


def _fail(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("unexpected call")


def _write_metrics_json(root: Path, payload: dict[str, Any]) -> None:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "metrics_latest.json").write_text(json.dumps(payload), encoding="utf-8")


# -----------------------------------------------------------------------
# CL metrics
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("artifact", "expected"),
    [
        (
            {
                "bwt": "-0.05",
                "fwt": 0.1,
                "plasticity": 0.7,
                "experience_accs": [0.9, "0.8"],
                "strategy": "ewc",
            },
            {
                "bwt": -0.05,
                "fwt": 0.1,
                "plasticity": 0.7,
                "stability": 0.0,
                "experience_accs": [0.9, 0.8],
                "strategy": "ewc",
            },
        ),
        (
            {},
            {
                "bwt": 0.0,
                "fwt": 0.0,
                "plasticity": 0.0,
                "stability": 0.0,
                "experience_accs": [],
                "strategy": "not_available",
            },
        ),
    ],
    ids=["artifact", "unavailable"],
)
def test_cl_metrics_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    artifact: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    monkeypatch.setattr(f"{METRICS}.load_cl_metrics_artifact", lambda: artifact)
    response = client.get("/api/v1/metrics/cl")
    assert response.status_code == 200
    assert response.json() == expected


# -----------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------


SIM = {"service_level": 0.97, "fill_rate": 0.95, "total_cost": 777.0}
# RL keys in forecast sources are placeholders and must never reach the response.
PLACEHOLDER_RL = {"service_level": 0.967, "fill_rate": 0.943, "total_cost": 18450.0}


@pytest.mark.parametrize(
    ("live_payload", "artifact", "expected"),
    [
        (
            {
                "smape": 10.0,
                "baseline_smape": 15.0,
                "rmse": 2.0,
                "coverage_80pct": 0.8,
                "model_version": "run-1",
            },
            None,
            (10.0, 15.0, 5.0, 2.0, 0.8, "run-1"),
        ),
        (
            {"smape": 12.0},
            None,
            (12.0, 12.0, 0.0, 0.0, 0.0, "mlflow_latest"),  # baseline defaults to smape
        ),
        (
            None,
            {
                "val_smape": 20.0,
                "baseline_smape": 25.0,
                "val_rmse": 4.0,
                "val_coverage_80pct": 0.75,
                "model_version": "v7",
            },
            (20.0, 25.0, 5.0, 4.0, 0.75, "v7"),
        ),
    ],
    ids=["mlflow", "mlflow_minimal", "artifact_val_aliases"],
)
def test_kpis_forecast_metrics_by_source(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    live_payload: dict[str, Any] | None,
    artifact: dict[str, Any] | None,
    expected: tuple[float, float, float, float, float, str],
) -> None:
    if artifact is not None:
        _write_metrics_json(tmp_path, {**artifact, **PLACEHOLDER_RL})
    live = None if live_payload is None else {**live_payload, **PLACEHOLDER_RL}
    monkeypatch.setattr(f"{METRICS}.fetch_live_kpi_payload", lambda: live)
    monkeypatch.setattr(f"{METRICS}.get_latest_simulation", lambda: SIM)
    monkeypatch.setattr(f"{METRICS}.load_cl_metrics_artifact", lambda: {"bwt": -0.05})

    payload = client.get("/api/v1/metrics/kpis").json()

    keys = ("smape", "baseline_smape", "smape_uplift_vs_baseline", "rmse", "coverage_80pct")
    assert tuple(payload[k] for k in keys) + (payload["model_version"],) == expected
    assert payload["bwt"] == -0.05
    assert payload["is_demo"] is False
    # Inventory KPIs come from the simulation, not from the forecast source.
    assert (payload["service_level"], payload["fill_rate"], payload["total_cost"]) == (
        pytest.approx((97.0, 95.0, 777.0))
    )


def test_kpis_503_without_live_or_artifact_metrics(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(f"{METRICS}.fetch_live_kpi_payload", lambda: None)
    response = client.get("/api/v1/metrics/kpis")
    assert response.status_code == 503
    assert "KPI metrics are not ready" in response.json()["detail"]


@pytest.mark.parametrize(
    ("cached_sim", "live_sim", "expected"),
    [
        (SIM, None, (97.0, 95.0, 777.0)),  # cached simulation; no live rollout
        ({}, {"service_level": 0.91, "fill_rate": 0.89, "total_cost": 555.0}, (91.0, 89.0, 555.0)),
        ({"service_level": 0.0}, {}, None),  # no RL agent loaded
    ],
    ids=["cached_simulation", "live_simulation", "no_agent"],
)
def test_kpis_inventory_metrics_come_from_rl_agent(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cached_sim: dict[str, float],
    live_sim: dict[str, float] | None,
    expected: tuple[float, float, float] | None,
) -> None:
    _write_metrics_json(tmp_path, {"smape": 18.0, **PLACEHOLDER_RL})
    sim_calls: list[int] = []

    def _live(**kwargs: Any) -> dict[str, float]:
        sim_calls.append(kwargs["n_days"])
        return live_sim or {}

    monkeypatch.setattr(f"{METRICS}.fetch_live_kpi_payload", lambda: None)
    monkeypatch.setattr(f"{METRICS}.get_latest_simulation", lambda: cached_sim)
    monkeypatch.setattr(
        f"{METRICS}.run_rl_simulation_metrics", _fail if live_sim is None else _live
    )
    monkeypatch.setattr(f"{METRICS}.load_cl_metrics_artifact", lambda: {})

    payload = client.get("/api/v1/metrics/kpis").json()

    rl = (payload["service_level"], payload["fill_rate"], payload["total_cost"])
    if expected is None:
        assert rl == (None, None, None)
        assert payload["rl_agent_loaded"] is False
    else:
        assert rl == pytest.approx(expected)
        assert payload["rl_agent_loaded"] is True
    assert sim_calls == ([] if live_sim is None else [90])
    assert payload["smape"] == 18.0  # forecast KPIs are still served without an agent


# -----------------------------------------------------------------------
# Simulation endpoints
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("store_ids", "detail"),
    [("1,abc", "Invalid store_ids"), (" , ,", "store_ids cannot be empty")],
)
def test_simulate_batch_rejects_bad_store_ids(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, store_ids: str, detail: str
) -> None:
    monkeypatch.setattr(f"{METRICS}.run_rl_simulation_metrics_async", _fail)
    response = client.get("/api/v1/metrics/simulate/batch", params={"store_ids": store_ids})
    assert response.status_code == 400
    assert detail in response.json()["detail"]


def test_simulate_batch_forwards_request_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def _batch(**kwargs: Any) -> list[dict[str, float]]:
        captured.update(kwargs)
        return [{"store_id": float(s), "total_cost": 1.0} for s in kwargs["store_ids"]]

    monkeypatch.setattr(f"{METRICS}.run_rl_simulation_metrics_async", _batch)

    response = client.get(
        "/api/v1/metrics/simulate/batch",
        params={"store_ids": " 3, 4 ,", "n_days": 14, "risk_aversion": 0.2},
        headers={"x-request-id": "req-9"},
    )

    assert response.status_code == 200
    assert [item["store_id"] for item in response.json()] == [3.0, 4.0]
    assert response.headers["x-request-id"] == "req-9"
    assert captured == {
        "store_ids": [3, 4],
        "n_days": 14,
        "risk_aversion": 0.2,
        "n_mc": 16,
        "request_id": "req-9",
    }


@pytest.mark.parametrize(
    ("endpoint", "params"),
    [("single", {"store_id": 3}), ("batch", {"store_ids": "3,4"})],
)
def test_simulate_503_without_rl_agent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, endpoint: str, params: dict[str, Any]
) -> None:
    # No agent: the registry returns {} (single) and the async batch wrapper raises.
    monkeypatch.setattr("src.api.model_registry.get_rl_agent", lambda: None)
    response = client.get(f"/api/v1/metrics/simulate/{endpoint}", params=params)
    assert response.status_code == 503
    assert response.json()["detail"] == "Simulation unavailable: no RL agent loaded."


# -----------------------------------------------------------------------
# Store list
# -----------------------------------------------------------------------


def _write_reference_parquet(root: Path) -> None:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame({"Store": [3, 1, 3, None, 2]}).to_parquet(processed / "tft_reference.parquet")


def _write_train_csv(root: Path) -> None:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "train.csv").write_text("Store,Sales\n7,1\n5,2\n7,3\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("write_source", "forecastable"),
    [(_write_reference_parquet, [1, 2, 3]), (_write_train_csv, [5, 7]), (None, ALL_STORES)],
    ids=["reference_parquet", "train_csv", "no_data"],
)
def test_stores_endpoint_sources(
    client: TestClient,
    tmp_path: Path,
    write_source: Callable[[Path], None] | None,
    forecastable: list[int],
) -> None:
    if write_source is not None:
        write_source(tmp_path)

    response = client.get("/api/v1/metrics/stores")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stores"] == ALL_STORES
    assert payload["total"] == 1115
    assert payload["forecastable_stores"] == forecastable
    assert payload["forecastable_total"] == len(forecastable)
