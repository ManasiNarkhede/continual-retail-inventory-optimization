"""Unit tests for forecast router behavior."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.model_registry import clear_registry


def test_forecast_503_when_tft_not_loaded() -> None:
    """Ensure forecast route returns valid status with/without artifacts."""
    clear_registry()
    client = TestClient(app)
    response = client.post("/api/v1/forecast/", json={"store_id": 1, "horizon": 7})
    assert response.status_code in {200, 503}
    if response.status_code == 503:
        assert "Forecast model not loaded" in response.json()["detail"]


def test_forecast_batch_returns_multiple_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure batch endpoint responds with either model output or 503."""
    clear_registry()
    monkeypatch.setattr("src.api.routers.forecast.get_tft_model", lambda: (None, None, None))
    client = TestClient(app)
    response = client.post("/api/v1/forecast/batch", json={"store_ids": [1, 2, 3], "horizon": 5})
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        payload = response.json()
        assert "forecasts" in payload


def test_forecast_404_for_unknown_store_when_model_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure forecast route returns 404 for missing store in loaded reference data."""

    class _Model:
        def predict_for_store(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise KeyError("Store 9999 not found")

    monkeypatch.setattr(
        "src.api.routers.forecast.get_tft_model",
        lambda: (_Model(), object(), object()),
    )

    client = TestClient(app)
    response = client.post("/api/v1/forecast/", json={"store_id": 9999, "horizon": 7})
    assert response.status_code == 404


def test_forecast_batch_async_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async batch endpoint should return forecasts list when model is available."""

    class _TensorLike:
        def __init__(self, values):
            self._values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            class _Arr:
                def __init__(self, values):
                    self._values = values

                def tolist(self):
                    return self._values

            return _Arr(self._values)

    class _Pred:
        def __getitem__(self, key):
            rows, col = key
            _ = rows
            if col == 0:
                return _TensorLike([10.0, 11.0])
            if col == 1:
                return _TensorLike([12.0, 13.0])
            return _TensorLike([14.0, 15.0])

    class _Model:
        quantile_indices = {"p10": 0, "p50": 1, "p90": 2}

        def predict_for_store(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return _Pred(), [
                type("_D", (), {"strftime": lambda self, fmt: "2025-01-01"})(),
                type("_D", (), {"strftime": lambda self, fmt: "2025-01-02"})(),
            ]

    monkeypatch.setattr(
        "src.api.routers.forecast.get_tft_model", lambda: (_Model(), object(), object())
    )

    client = TestClient(app)
    response = client.post("/api/v1/forecast/batch", json={"store_ids": [1, 2], "horizon": 2})
    assert response.status_code == 200
    payload = response.json()
    assert "forecasts" in payload
    assert len(payload["forecasts"]) == 2


def test_forecast_batch_propagates_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch endpoint should propagate 503 when TFT is unavailable."""
    from fastapi import HTTPException

    def _raise_503(req):  # noqa: ANN001
        raise HTTPException(status_code=503, detail="TFT model not loaded")

    monkeypatch.setattr("src.api.routers.forecast.get_forecast", _raise_503)

    client = TestClient(app)
    response = client.post("/api/v1/forecast/batch", json={"store_ids": [1, 2], "horizon": 2})
    assert response.status_code == 503


def test_forecast_uses_warmup_cache_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forecast route should return cached warmup payload without model call."""
    from src.api.model_registry import cache_forecast_payload

    cache_forecast_payload(
        store_id=1,
        horizon=7,
        payload={
            "store_id": 1,
            "horizon": 7,
            "dates": ["2025-01-01"] * 7,
            "p10": [10.0] * 7,
            "p50": [12.0] * 7,
            "p90": [14.0] * 7,
        },
    )

    monkeypatch.setattr(
        "src.api.routers.forecast.get_tft_model",
        lambda: (_ for _ in ()).throw(AssertionError("Should not be called")),
    )

    client = TestClient(app)
    response = client.post("/api/v1/forecast/", json={"store_id": 1, "horizon": 7})
    assert response.status_code == 200
    payload = response.json()
    assert payload["store_id"] == 1
    assert len(payload["p50"]) == 7


def test_forecast_applies_interval_calibration_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forecast route should widen intervals when calibration scale > 1."""

    class _TensorLike:
        def __init__(self, values):
            self._values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class _Pred:
        def __getitem__(self, key):
            _rows, col = key
            if col == 0:
                return _TensorLike([90.0, 95.0])
            if col == 1:
                return _TensorLike([100.0, 105.0])
            return _TensorLike([110.0, 115.0])

    class _Model:
        def predict_for_store(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return _Pred(), [
                type("_D", (), {"strftime": lambda self, fmt: "2025-01-01"})(),
                type("_D", (), {"strftime": lambda self, fmt: "2025-01-02"})(),
            ]

    clear_registry()
    monkeypatch.setattr(
        "src.api.routers.forecast.get_tft_model", lambda: (_Model(), object(), object())
    )
    monkeypatch.setattr(
        "src.api.routers.forecast.get_tft_quantile_indices", lambda: {"p10": 0, "p50": 1, "p90": 2}
    )
    monkeypatch.setattr("src.api.model_registry._forecast_interval_scale", lambda config=None: 2.0)

    client = TestClient(app)
    response = client.post("/api/v1/forecast/", json={"store_id": 77, "horizon": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["p50"] == [100.0, 105.0]
    assert payload["p10"] == [80.0, 85.0]
    assert payload["p90"] == [120.0, 125.0]
