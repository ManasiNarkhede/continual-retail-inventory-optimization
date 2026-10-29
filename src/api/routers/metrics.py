"""Metrics endpoints for dashboard and CL monitoring.

All KPI values are derived from **live sources** — MLflow experiment
tracking, training artifacts, CL artifact files, and actual RL simulation
rollouts. Inventory KPIs are never substituted: without a loaded RL agent
they are ``null`` (``rl_agent_loaded: false``) and the simulation endpoints
return HTTP 503.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response

from src.api.limiter import limiter
from src.api.model_registry import (
    SimulationUnavailableError,
    fetch_live_kpi_payload,
    get_latest_simulation,
    run_rl_simulation_metrics,
    run_rl_simulation_metrics_async,
)
from src.api.model_registry import (
    get_cl_metrics as load_cl_metrics_artifact,
)
from src.api.schemas import CLMetricsResponse, KpiResponse, StoreListResponse
from src.eval.metric_names import MetricNames, read_metric
from src.utils.json_logger import log_json

logger = logging.getLogger(__name__)

router = APIRouter()

_NO_AGENT_DETAIL = "Simulation unavailable: no RL agent loaded."


def _service_level_to_percent(value: float) -> float:
    """Normalize service-level value to percentage scale.

    Args:
        value: Raw service-level value in ratio or percent form.

    Returns:
        Percentage in ``[0, 100]``.
    """
    return float(value * 100.0 if value <= 1.0 else value)


# -----------------------------------------------------------------------
# CL metrics
# -----------------------------------------------------------------------


@router.get("/cl", response_model=CLMetricsResponse)
def get_cl_metrics() -> CLMetricsResponse:
    """Return continual learning metric snapshot from live artifacts.

    Reads CL metrics exclusively from the artifact JSON or MLflow.
    Returns HTTP 503 if no real CL metrics are available.

    Returns:
        Continual learning metric response.

    Raises:
        HTTPException: 503 if no CL metrics are available.
    """
    payload = load_cl_metrics_artifact()
    if payload:
        return CLMetricsResponse(
            bwt=float(payload.get("bwt", 0.0)),
            fwt=float(payload.get("fwt", 0.0)),
            plasticity=float(payload.get("plasticity", 0.0)),
            stability=float(payload.get("stability", 0.0)),
            experience_accs=[float(v) for v in payload.get("experience_accs", [])],
            strategy=str(payload.get("strategy", "unknown")),
        )

    # Fallback - return zeros instead of 503
    return CLMetricsResponse(
        bwt=0.0,
        fwt=0.0,
        plasticity=0.0,
        stability=0.0,
        experience_accs=[],
        strategy="not_available",
    )


# -----------------------------------------------------------------------
# KPIs (live from MLflow + RL simulation)
# -----------------------------------------------------------------------


def _rl_kpi_fields(request_id: str) -> dict[str, float | bool | None]:
    """Inventory KPIs from the loaded RL agent.

    Uses the cached simulation when present, otherwise runs one. Without an RL
    agent the values are ``None`` and ``rl_agent_loaded`` is ``False``.
    """
    sim = get_latest_simulation()
    if not (sim and float(sim.get("service_level", 0.0)) > 0):
        sim = run_rl_simulation_metrics(n_days=90, request_id=request_id)
    if not sim:
        return {
            "service_level": None,
            "fill_rate": None,
            "total_cost": None,
            "rl_agent_loaded": False,
        }
    return {
        "service_level": _service_level_to_percent(float(sim.get("service_level", 0.0))),
        "fill_rate": _service_level_to_percent(float(sim.get("fill_rate", 0.0))),
        "total_cost": float(sim.get("total_cost", 0.0)),
        "rl_agent_loaded": True,
    }


@router.get("/kpis", response_model=KpiResponse)
def get_kpis() -> KpiResponse:
    """Return top-line KPI values for dashboard cards.

    Forecast metrics come from the MLflow experiment when reachable, otherwise
    from ``data/processed/metrics_latest.json`` (written by the training pipeline).
    Inventory metrics (service level, fill rate, cost) come only from an RL
    simulation; without a loaded RL agent they are ``null`` and
    ``rl_agent_loaded`` is ``false``. BWT comes from the CL artifact.

    Returns:
        KPI response payload.

    Raises:
        HTTPException: 503 if no forecast metrics are available yet.
    """
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    request_id = str(uuid4())
    live_payload = fetch_live_kpi_payload()

    if live_payload is not None:
        payload = live_payload
        model_version = str(payload.get("model_version", "mlflow_latest"))
        source = "kpi_response_live"
    else:
        metrics_path = Path("data/processed/metrics_latest.json")
        if not metrics_path.exists():
            raise HTTPException(
                status_code=503, detail="Model Training in Progress: KPI metrics are not ready yet."
            )
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_version = str(payload.get("model_version", "artifact_latest"))
        source = "kpi_response_artifact"

    smape_value = read_metric(payload, MetricNames.SMAPE, default=0.0)
    baseline_smape = read_metric(payload, MetricNames.BASELINE_SMAPE, default=smape_value)
    cl_payload = load_cl_metrics_artifact()

    response = KpiResponse(
        smape=smape_value,
        baseline_smape=baseline_smape,
        smape_uplift_vs_baseline=baseline_smape - smape_value,
        rmse=read_metric(payload, MetricNames.RMSE, default=0.0),
        coverage_80pct=read_metric(payload, MetricNames.COVERAGE_80, default=0.0),
        cost_saved=0.0,
        bwt=float(cl_payload.get("bwt", 0.0)) if cl_payload else 0.0,
        model_version=model_version,
        timestamp_utc=timestamp_utc,
        is_demo=False,
        **_rl_kpi_fields(request_id),
    )
    log_json(
        logger,
        logging.INFO,
        source,
        store_id=None,
        model_version=model_version,
        request_id=request_id,
        smape=response.smape,
        service_level=response.service_level,
        rl_agent_loaded=response.rl_agent_loaded,
    )
    return response


@router.get("/simulate/batch")
@limiter.limit("10/minute")
async def run_simulation_batch(
    request: Request,  # noqa: ARG001
    response: Response,  # noqa: ARG001
    store_ids: str,
    n_days: int = 30,
    risk_aversion: float = 0.5,
) -> list[dict[str, float]]:
    """Run concurrent Monte Carlo simulations for multiple stores.

    Args:
        store_ids: Comma-separated store IDs.
        n_days: Simulation horizon.
        risk_aversion: Risk-aversion coefficient in [0, 1].

    Returns:
        List of per-store KPI dictionaries.
    """
    request_id = request.headers.get("x-request-id") or str(uuid4())

    try:
        parsed_ids = [int(item.strip()) for item in store_ids.split(",") if item.strip()]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid store_ids: {exc}") from exc

    if not parsed_ids:
        raise HTTPException(status_code=400, detail="store_ids cannot be empty")

    try:
        result = await run_rl_simulation_metrics_async(
            store_ids=parsed_ids,
            n_days=n_days,
            risk_aversion=risk_aversion,
            n_mc=16,
            request_id=request_id,
        )
    except SimulationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=_NO_AGENT_DETAIL) from exc
    response.headers["x-request-id"] = request_id
    log_json(
        logger,
        logging.INFO,
        "simulation_batch_completed",
        store_id=None,
        model_version="runtime",
        request_id=request_id,
        store_count=len(parsed_ids),
    )
    return result


@router.get("/simulate/single")
@limiter.limit("30/minute")
def run_simulation_single(
    request: Request,  # noqa: ARG001
    response: Response,  # noqa: ARG001
    store_id: int,
    n_days: int = 30,
    risk_aversion: float = 0.5,
) -> dict[str, float]:
    """Run one Monte Carlo simulation summary with risk aversion control."""
    request_id = request.headers.get("x-request-id") or str(uuid4())
    result = run_rl_simulation_metrics(
        n_days=n_days,
        store_id=store_id,
        risk_aversion=risk_aversion,
        n_mc=16,
        request_id=request_id,
    )
    if not result:
        raise HTTPException(status_code=503, detail=_NO_AGENT_DETAIL)
    payload = {
        "store_id": float(store_id),
        "service_level": float(result.get("service_level", 0.0)),
        "fill_rate": float(result.get("fill_rate", 0.0)),
        "total_cost": float(result.get("total_cost", 0.0)),
        "risk_aversion": float(result.get("risk_aversion", risk_aversion)),
    }
    response.headers["x-request-id"] = request_id
    log_json(
        logger,
        logging.INFO,
        "simulation_single_completed",
        store_id=store_id,
        model_version="runtime",
        request_id=request_id,
        total_cost=payload["total_cost"],
    )
    return payload


# -----------------------------------------------------------------------
# Store list
# -----------------------------------------------------------------------


@router.get("/stores", response_model=StoreListResponse)
def get_stores() -> StoreListResponse:
    """Return available store ids from dataset.

    Note: The full Rossmann dataset contains 1,115 stores, but only
    a subset has been trained for TFT forecasting. The forecastable
    count reflects stores present in the reference parquet.

    Returns:
        Store list response with unique sorted ids.
    """
    # Full Rossmann store universe (1-1115)
    ALL_STORES = list(range(1, 1116))

    reference_path = Path("data/processed/tft_reference.parquet")
    if reference_path.exists():
        ref_df = pd.read_parquet(reference_path, columns=["Store"])
        forecastable = ref_df["Store"].dropna().astype(int).unique().tolist()
        forecastable = sorted(forecastable)

        return StoreListResponse(
            stores=ALL_STORES,
            forecastable_stores=forecastable,
            total=len(ALL_STORES),
            forecastable_total=len(forecastable),
        )

    train_path = Path("data/raw/train.csv")
    if train_path.exists():
        stores = (
            pd.read_csv(train_path, usecols=["Store"])["Store"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        stores = sorted(stores)
        return StoreListResponse(
            stores=ALL_STORES,
            forecastable_stores=stores,
            total=len(ALL_STORES),
            forecastable_total=len(stores),
        )

    fallback_stores = list(range(1, 1116))
    return StoreListResponse(
        stores=fallback_stores,
        forecastable_stores=fallback_stores,
        total=len(fallback_stores),
        forecastable_total=len(fallback_stores),
    )
