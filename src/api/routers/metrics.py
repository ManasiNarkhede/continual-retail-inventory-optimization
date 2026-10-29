"""Metrics endpoints for dashboard and CL monitoring.

All KPI values are derived from **live sources** — MLflow experiment
tracking, CL artifact files, and actual RL simulation rollouts.
Hardcoded placeholder values have been removed.  When no real data is
available the endpoint returns ``is_demo: true`` with explicit zero
values so the frontend can show a clear unavailable state.
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


@router.get("/kpis", response_model=KpiResponse)
def get_kpis() -> KpiResponse:
    """Return top-line KPI values for dashboard cards.

    Data sources (in priority order):
    1. ``data/processed/metrics_latest.json`` — written by the training
       pipeline with forecast SMAPE, RMSE, and coverage.
    2. MLflow experiment query — fallback for forecast metrics.
    3. Live RL simulation — provides actual service-level and fill-rate.
    4. CL artifact — provides real BWT.

    When none of these are available, returns ``is_demo: true`` with zero
    values so the frontend can display a clear "no data" state.

    Returns:
        KPI response payload.
    """
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    request_id = str(uuid4())
    response: KpiResponse | None = None
    live_payload = fetch_live_kpi_payload()

    if live_payload is not None:
        smape_value = read_metric(live_payload, MetricNames.SMAPE, default=0.0)
        baseline_smape = read_metric(live_payload, MetricNames.BASELINE_SMAPE, default=smape_value)
        model_rmse = read_metric(live_payload, MetricNames.RMSE, default=0.0)
        coverage = read_metric(live_payload, MetricNames.COVERAGE_80, default=0.0)
        model_version = str(live_payload.get("model_version", "mlflow_latest"))
        service_level = read_metric(live_payload, MetricNames.SERVICE_LEVEL, default=0.0)
        fill_rate = read_metric(live_payload, MetricNames.FILL_RATE, default=0.0)
        total_cost_live = read_metric(live_payload, MetricNames.TOTAL_COST, default=0.0)
        if fill_rate <= 0.0 or total_cost_live <= 0.0:
            sim_live = run_rl_simulation_metrics(n_days=90, request_id=request_id)
            fill_rate = float(sim_live.get("fill_rate", fill_rate))
            total_cost_live = float(sim_live.get("total_cost", total_cost_live))
        cl_payload = load_cl_metrics_artifact()
        bwt_value = float(cl_payload.get("bwt", 0.0)) if cl_payload else 0.0

        uplift = baseline_smape - smape_value

        response = KpiResponse(
            smape=smape_value,
            baseline_smape=baseline_smape,
            smape_uplift_vs_baseline=uplift,
            rmse=model_rmse,
            coverage_80pct=coverage,
            service_level=_service_level_to_percent(service_level),
            fill_rate=_service_level_to_percent(fill_rate),
            total_cost=total_cost_live,
            cost_saved=0.0,
            bwt=bwt_value,
            model_version=model_version,
            timestamp_utc=timestamp_utc,
            is_demo=False,
        )
        response_model_version = model_version
        log_json(
            logger,
            logging.INFO,
            "kpi_response_live",
            store_id=None,
            model_version=response_model_version,
            request_id=request_id,
            smape=response.smape,
            service_level=response.service_level,
        )
        return response

    # Fallback path for environments without MLflow server
    metrics_path = Path("data/processed/metrics_latest.json")
    if not metrics_path.exists():
        raise HTTPException(
            status_code=503, detail="Model Training in Progress: KPI metrics are not ready yet."
        )

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    smape_value = read_metric(payload, MetricNames.SMAPE, default=0.0)
    baseline_smape = read_metric(payload, MetricNames.BASELINE_SMAPE, default=smape_value)
    model_rmse = read_metric(payload, MetricNames.RMSE, default=0.0)
    coverage = read_metric(payload, MetricNames.COVERAGE_80, default=0.0)
    model_version = str(payload.get("model_version", "artifact_latest"))

    # Read RL metrics from JSON if available
    service_level = float(payload.get("service_level", 0.0))
    fill_rate = float(payload.get("fill_rate", 0.0))
    total_cost = float(payload.get("total_cost", 0.0))

    # --- Source 3: RL simulation (only as fallback when JSON has no RL metrics) ---
    if service_level <= 0.0 or fill_rate <= 0.0:
        sim = get_latest_simulation()
        if sim and sim.get("service_level", 0.0) > 0:
            service_level = sim.get("service_level", service_level)
            fill_rate = sim.get("fill_rate", fill_rate)
            total_cost = sim.get("total_cost", total_cost)
        else:
            sim_live = run_rl_simulation_metrics(n_days=90, request_id=request_id)
            if sim_live:
                service_level = float(sim_live.get("service_level", service_level))
                fill_rate = float(sim_live.get("fill_rate", fill_rate))
                total_cost = float(sim_live.get("total_cost", total_cost))

    # --- Source 4: CL metrics (graceful fallback to 0.0 when unavailable) ---
    cl_payload = load_cl_metrics_artifact()
    bwt_value = float(cl_payload.get("bwt", 0.0)) if cl_payload else 0.0

    uplift = baseline_smape - smape_value

    response = KpiResponse(
        smape=smape_value,
        baseline_smape=baseline_smape,
        smape_uplift_vs_baseline=uplift,
        rmse=model_rmse,
        coverage_80pct=coverage,
        service_level=_service_level_to_percent(service_level),
        fill_rate=_service_level_to_percent(fill_rate),
        total_cost=float(total_cost),
        cost_saved=0.0,
        bwt=bwt_value,
        model_version=model_version,
        timestamp_utc=timestamp_utc,
        is_demo=False,
    )
    log_json(
        logger,
        logging.INFO,
        "kpi_response_artifact",
        store_id=None,
        model_version=model_version,
        request_id=request_id,
        smape=response.smape,
        service_level=response.service_level,
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

    result = await run_rl_simulation_metrics_async(
        store_ids=parsed_ids,
        n_days=n_days,
        risk_aversion=risk_aversion,
        n_mc=16,
        request_id=request_id,
    )
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
        raise HTTPException(status_code=503, detail="Simulation unavailable")
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
