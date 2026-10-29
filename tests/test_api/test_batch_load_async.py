"""Lightweight async load test for batch forecast endpoint."""

from __future__ import annotations

import asyncio

import pytest

from src.api.main import app
from src.api.model_registry import clear_registry
from src.api.schemas import ForecastResponse


@pytest.mark.asyncio
async def test_batch_forecast_endpoint_concurrent_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue 50 concurrent batch requests and ensure endpoint responds."""
    httpx = pytest.importorskip("httpx")

    clear_registry()

    def _fake_forecast(req):  # noqa: ANN001
        dates = [f"2025-01-{day:02d}" for day in range(1, req.horizon + 1)]
        p10 = [10.0] * req.horizon
        p50 = [12.0] * req.horizon
        p90 = [14.0] * req.horizon
        return ForecastResponse(
            store_id=req.store_id, horizon=req.horizon, dates=dates, p10=p10, p50=p50, p90=p90
        )

    monkeypatch.setattr("src.api.routers.forecast.get_forecast", _fake_forecast)

    payload = {"store_ids": list(range(1, 51)), "horizon": 7}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", timeout=15.0
    ) as client:
        tasks = [client.post("/api/v1/forecast/batch", json=payload) for _ in range(50)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    ok = 0
    unavailable = 0
    for response in responses:
        if isinstance(response, Exception):
            continue
        if response.status_code == 200:
            ok += 1
        elif response.status_code == 503:
            unavailable += 1

    assert ok + unavailable >= 40
