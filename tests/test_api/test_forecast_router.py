"""Tests for ``src.api.routers.forecast`` (single, batch, statistical fallback, cache)."""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.model_registry import cache_forecast_payload, get_cached_forecast
from src.api.routers import forecast as forecast_router
from tests.test_api.conftest import FakeTensor

pytestmark = pytest.mark.usefixtures("isolated_registry")

FORECAST = "src.api.routers.forecast"
# Each weekday appears twice in the history below, 7 days (70 sales units) apart.
_WEEKDAY_STD = statistics.stdev([100.0, 170.0])


def _history() -> pd.DataFrame:
    """Two weeks of daily sales for store 1 (100, 110, ...) ending Tuesday 2025-01-14."""
    dates = pd.date_range("2025-01-01", periods=14, freq="D")
    return pd.DataFrame(
        {
            "Store": 1,
            "Date": dates.strftime("%Y-%m-%d"),
            "sales": [100.0 + 10 * i for i in range(14)],
        }
    )


class _RaisingModel:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def predict_for_store(self, **kwargs: Any) -> Any:
        raise self._exc


@pytest.mark.parametrize(
    "model",
    [None, _RaisingModel(RuntimeError("CUDA exploded"))],
    ids=["no_model", "model_runtime_error"],
)
def test_forecast_falls_back_to_weekday_statistics(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, model: Any
) -> None:
    monkeypatch.setattr(f"{FORECAST}.get_tft_model", lambda: (model, object(), _history()))

    response = client.post("/api/v1/forecast/", json={"store_id": 1, "horizon": 3})

    assert response.status_code == 200
    payload = response.json()
    p50 = [135.0, 145.0, 155.0]  # mean of the two Wednesdays / Thursdays / Fridays
    assert payload["dates"] == ["2025-01-15", "2025-01-16", "2025-01-17"]
    assert payload["p50"] == p50
    assert payload["p10"] == [round(v - _WEEKDAY_STD, 2) for v in p50]
    assert payload["p90"] == [round(v + _WEEKDAY_STD, 2) for v in p50]
    assert get_cached_forecast(store_id=1, horizon=3) == payload


def test_statistical_forecast_single_row_uses_default_spread() -> None:
    df = pd.DataFrame({"Store": [9], "Date": ["2025-03-03"], "sales": [100.0]})
    response = forecast_router._generate_statistical_forecast(9, 2, df)
    assert response.p50 == [100.0, 100.0]
    assert response.p10 == [85.0, 85.0]
    assert response.p90 == [115.0, 115.0]


@pytest.mark.parametrize(
    ("model", "detail"),
    [
        (_RaisingModel(KeyError("Store 9999 not found")), "Store 9999 not found"),
        (None, "Store 9999 not found in reference data"),
    ],
    ids=["model_key_error", "fallback_unknown_store"],
)
def test_forecast_unknown_store_returns_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, model: Any, detail: str
) -> None:
    monkeypatch.setattr(f"{FORECAST}.get_tft_model", lambda: (model, object(), _history()))
    response = client.post("/api/v1/forecast/", json={"store_id": 9999, "horizon": 7})
    assert response.status_code == 404
    assert detail in response.json()["detail"]


def test_forecast_503_when_no_model_and_no_reference(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(f"{FORECAST}.get_tft_model", lambda: (None, None, None))
    response = client.post("/api/v1/forecast/", json={"store_id": 1, "horizon": 3})
    assert response.status_code == 503
    assert "Forecast model not loaded" in response.json()["detail"]


@pytest.mark.parametrize(
    ("q_idx", "scale", "expected"),
    [
        # Missing indices fall back to columns 1/3/5 at identity scale.
        ({}, 1.0, ([90.0, 90.0], [100.0, 100.0], [110.0, 110.0])),
        # Configured indices; calibration scale 2 doubles the distance from p50.
        ({"p10": 0, "p50": 2, "p90": 4}, 2.0, ([3.0, 3.0], [7.0, 7.0], [11.0, 11.0])),
    ],
    ids=["default_columns", "configured_columns_scaled"],
)
def test_forecast_model_path_selects_quantiles_and_calibrates(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    q_idx: dict[str, int],
    scale: float,
    expected: tuple[list[float], list[float], list[float]],
) -> None:
    class _Model:
        def predict_for_store(self, horizon: int, **kwargs: Any) -> Any:
            preds = np.tile([[5.0, 90.0, 7.0, 100.0, 9.0, 110.0]], (horizon, 1))
            return FakeTensor(preds), list(pd.date_range("2025-02-01", periods=horizon))

    monkeypatch.setattr(f"{FORECAST}.get_tft_model", lambda: (_Model(), object(), object()))
    monkeypatch.setattr(f"{FORECAST}.get_tft_quantile_indices", lambda: q_idx)
    monkeypatch.setattr(
        "src.api.model_registry._forecast_interval_scale", lambda config=None: scale
    )

    response = client.post("/api/v1/forecast/", json={"store_id": 8, "horizon": 2})

    assert response.status_code == 200
    payload = response.json()
    assert (payload["p10"], payload["p50"], payload["p90"]) == expected
    assert payload["dates"] == ["2025-02-01", "2025-02-02"]
    assert get_cached_forecast(store_id=8, horizon=2) == payload


def test_forecast_uses_warmup_cache_when_available(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = {
        "store_id": 1,
        "horizon": 7,
        "dates": ["2025-01-01"] * 7,
        "p10": [10.0] * 7,
        "p50": [12.0] * 7,
        "p90": [14.0] * 7,
    }
    cache_forecast_payload(store_id=1, horizon=7, payload=cached)
    monkeypatch.setattr(
        f"{FORECAST}.get_tft_model",
        lambda: pytest.fail("model must not be queried when the forecast is cached"),
    )

    response = client.post("/api/v1/forecast/", json={"store_id": 1, "horizon": 7})

    assert response.status_code == 200
    payload = response.json()
    assert payload["p50"] == [12.0] * 7
    assert payload["p90"] == [14.0] * 7


def test_forecast_batch_returns_per_store_forecasts_and_propagates_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = pd.concat([_history(), _history().assign(Store=2, sales=50.0)], ignore_index=True)
    monkeypatch.setattr(f"{FORECAST}.get_tft_model", lambda: (None, None, history))

    response = client.post("/api/v1/forecast/batch", json={"store_ids": [2, 1], "horizon": 2})

    assert response.status_code == 200
    forecasts = response.json()["forecasts"]
    assert [f["store_id"] for f in forecasts] == [2, 1]
    assert forecasts[0]["p50"] == [50.0, 50.0]
    assert forecasts[1]["p50"] == [135.0, 145.0]

    missing = client.post("/api/v1/forecast/batch", json={"store_ids": [1, 99], "horizon": 2})
    assert missing.status_code == 404
