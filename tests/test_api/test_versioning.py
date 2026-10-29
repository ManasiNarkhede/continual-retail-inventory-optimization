"""Tests for API versioning parity between legacy and v1 routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_forecast_batch_route_available_on_both_prefixes(monkeypatch) -> None:
    """Batch forecast endpoint should exist on both /api and /api/v1."""

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
        def predict_for_store(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return _Pred(), [
                type("_D", (), {"strftime": lambda self, fmt: "2025-01-01"})(),
                type("_D", (), {"strftime": lambda self, fmt: "2025-01-02"})(),
            ]

    monkeypatch.setattr(
        "src.api.routers.forecast.get_tft_model", lambda: (_Model(), object(), object())
    )
    monkeypatch.setattr(
        "src.api.routers.forecast.get_tft_quantile_indices", lambda: {"p10": 0, "p50": 1, "p90": 2}
    )

    client = TestClient(app)
    payload = {"store_ids": [1, 2], "horizon": 7}
    legacy = client.post("/api/forecast/batch", json=payload)
    v1 = client.post("/api/v1/forecast/batch", json=payload)

    assert legacy.status_code == 200
    assert v1.status_code == 200
    assert legacy.json() == v1.json()


def test_alerts_active_route_available_on_both_prefixes() -> None:
    """Alerts endpoint should be backward compatible on both route prefixes."""
    client = TestClient(app)
    legacy = client.get("/api/alerts/active")
    v1 = client.get("/api/v1/alerts/active")

    assert legacy.status_code in {200, 503}
    assert v1.status_code in {200, 503}
