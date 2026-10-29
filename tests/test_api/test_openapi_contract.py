"""Contract-style tests for OpenAPI surface stability."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_openapi_contains_expected_key_paths() -> None:
    """Guard key route contracts against accidental removal/rename."""
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200

    payload = response.json()
    paths = payload.get("paths", {})
    expected = {
        "/api/v1/forecast/",
        "/api/v1/forecast/batch",
        "/api/v1/metrics/kpis",
        "/api/v1/metrics/simulate/single",
        "/api/v1/metrics/simulate/batch",
        "/api/v1/simulate/",
    }
    for path in expected:
        assert path in paths


def test_openapi_kpi_response_shape_contract() -> None:
    """KPI schema should retain core response fields expected by frontend."""
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200

    payload = response.json()
    schemas = payload.get("components", {}).get("schemas", {})
    kpi_schema = schemas.get("KpiResponse", {})
    properties = kpi_schema.get("properties", {})

    for key in [
        "smape",
        "service_level",
        "fill_rate",
        "total_cost",
        "model_version",
        "timestamp_utc",
    ]:
        assert key in properties
