"""Rate limit tests for compute-heavy simulation endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.limiter import limiter
from src.api.main import app


def test_batch_simulation_rate_limit_and_headers(monkeypatch) -> None:
    """Batch endpoint should enforce 10/min and expose rate limit headers."""

    async def _fake_batch(*args, **kwargs):  # noqa: ANN001
        _ = args, kwargs
        return [
            {
                "store_id": 1.0,
                "service_level": 0.95,
                "fill_rate": 0.97,
                "total_cost": 1000.0,
                "risk_aversion": 0.5,
            }
        ]

    monkeypatch.setattr("src.api.routers.metrics.run_rl_simulation_metrics_async", _fake_batch)

    limiter.reset()
    with TestClient(app) as client:
        statuses = []
        headers = []
        for _ in range(11):
            response = client.get(
                "/api/v1/metrics/simulate/batch",
                params={"store_ids": "1", "n_days": 7, "risk_aversion": 0.5},
            )
            statuses.append(response.status_code)
            headers.append(response.headers)

    assert statuses[:10] == [200] * 10
    assert statuses[10] == 429
    assert any("x-ratelimit-limit" in {k.lower() for k in hdr.keys()} for hdr in headers)


def test_single_simulation_rate_limit(monkeypatch) -> None:
    """Single endpoint should allow 30/min then return 429."""

    def _fake_single(*args, **kwargs):  # noqa: ANN001
        _ = args, kwargs
        return {
            "service_level": 0.94,
            "fill_rate": 0.96,
            "total_cost": 1111.0,
            "risk_aversion": 0.8,
        }

    monkeypatch.setattr("src.api.routers.metrics.run_rl_simulation_metrics", _fake_single)

    limiter.reset()
    with TestClient(app) as client:
        last_status = 200
        for _ in range(31):
            response = client.get(
                "/api/v1/metrics/simulate/single",
                params={"store_id": 1, "n_days": 7, "risk_aversion": 0.8},
            )
            last_status = response.status_code

    assert last_status == 429
