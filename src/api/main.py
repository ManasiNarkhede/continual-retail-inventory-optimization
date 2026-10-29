"""FastAPI application entry point."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from configs.validator import load_yaml_config, validate_config_schema, validate_configs
from src.api.limiter import limiter
from src.api.model_registry import (
    load_rl_agent,
    load_tft_from_artifacts,
    model_registry_ready,
    warmup_forecast_cache,
)
from src.api.routers import alerts, forecast, inventory, metrics, simulation
from src.utils.json_logger import log_json

logger = logging.getLogger(__name__)


def _load_api_config(path: str = "configs/api_config.yaml") -> dict[str, Any]:
    """Load API configuration from YAML file.

    Args:
        path: Path to API config file.

    Returns:
        Parsed API configuration dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        return {
            "api": {
                "allowed_origins": ["http://localhost:5173"],
                "title": "Retail Inventory Optimization API",
                "version": "0.1.0",
            }
        }
    validate_config_schema(path, "configs/schemas/api_schema.json")
    payload = load_yaml_config(path)
    return payload if isinstance(payload, dict) else {}


api_cfg = _load_api_config()["api"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load runtime dependencies on startup using lifespan protocol.

    Args:
        app: FastAPI application instance.
    """
    validate_configs(
        tft_config_path="configs/tft_config.yaml", rl_config_path="configs/rl_config.yaml"
    )

    preload = os.getenv("API_PRELOAD_MODELS", "0") == "1"
    if preload:
        synthetic_demand = np.ones(365, dtype=float) * 50.0
        try:
            load_rl_agent(config_path="configs/rl_config.yaml", demand_series=synthetic_demand)
        except Exception as exc:  # pragma: no cover
            logger.warning("RL agent auto-load skipped during startup: %s", exc)

        try:
            load_tft_from_artifacts(
                checkpoint_path="checkpoints/tft/tft_latest.pt",
                reference_path="data/processed/tft_reference.parquet",
                config_path="data/processed/tft_runtime_config.yaml",
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("TFT model auto-load skipped during startup: %s", exc)

        try:
            cfg = _load_api_config().get("models", {})
            warmed = warmup_forecast_cache(
                top_n=int(cfg.get("forecast_warmup_top_n", 10)),
                horizon=int(cfg.get("forecast_warmup_horizon", 14)),
            )
            model_version = str(cfg.get("model_version", "runtime"))
            log_json(
                logger,
                logging.INFO,
                "forecast_warmup_completed",
                model_version=model_version,
                warmed_stores=warmed,
            )
        except (ValueError, RuntimeError) as exc:
            log_json(
                logger,
                logging.WARNING,
                "forecast_warmup_failed",
                model_version="runtime",
                error=str(exc),
            )
    else:
        logger.info("Model preloading disabled; using lazy-loading from model registry.")

    yield


app = FastAPI(
    title=api_cfg.get("title", "Retail Inventory Optimization API"),
    description="TFT + Continual Learning + RL inventory control",
    version=api_cfg.get("version", "0.1.0"),
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context_middleware(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    """Attach request-id header and normalized error response envelope."""
    request_id = request.headers.get("x-request-id") or os.urandom(8).hex()
    try:
        response = cast(Response, await call_next(request))
    except Exception:  # pragma: no cover
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"x-request-id": request_id},
        )

    response.headers["x-request-id"] = request_id
    return response


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(
        {
            *api_cfg.get("allowed_origins", ["http://localhost:5173"]),
            "http://localhost:5173",
        }
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for version_prefix in ("/api/v1", "/api"):
    app.include_router(forecast.router, prefix=f"{version_prefix}/forecast", tags=["Forecasting"])
    app.include_router(inventory.router, prefix=f"{version_prefix}/inventory", tags=["Inventory"])
    app.include_router(simulation.router, prefix=f"{version_prefix}/simulate", tags=["Simulation"])
    app.include_router(metrics.router, prefix=f"{version_prefix}/metrics", tags=["Metrics"])
    app.include_router(alerts.router, prefix=f"{version_prefix}/alerts", tags=["Alerts"])


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple health-check endpoint.

    Returns:
        Service status and version.
    """
    ready, message = model_registry_ready()
    return {
        "status": "ok" if ready else "degraded",
        "version": api_cfg.get("version", "0.1.0"),
        "registry": message,
    }
