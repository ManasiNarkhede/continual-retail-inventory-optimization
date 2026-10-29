"""Demand forecast endpoints."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.model_registry import (
    cache_forecast_payload,
    calibrate_prediction_interval,
    get_cached_forecast,
    get_tft_model,
    get_tft_quantile_indices,
)
from src.api.schemas import (
    ForecastBatchRequest,
    ForecastBatchResponse,
    ForecastRequest,
    ForecastResponse,
)

router = APIRouter()


def _to_numpy_1d(values: Any) -> np.ndarray:
    """Convert tensor-like or list-like payload into 1D float array."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    return np.asarray(values, dtype=float).reshape(-1)


def _generate_statistical_forecast(
    store_id: int, horizon: int, reference_df: pd.DataFrame
) -> ForecastResponse:
    """Generate a simple statistical forecast from historical data when ML model is unavailable.

    Computes mean + std per day-of-week for the requested store.
    """
    store_data = reference_df[reference_df["Store"] == store_id]
    if store_data.empty:
        raise HTTPException(status_code=404, detail=f"Store {store_id} not found in reference data")

    sales = store_data["sales"].astype(float)
    mean_sales = float(sales.mean())
    std_sales = float(sales.std()) if sales.std() > 0 else mean_sales * 0.15

    # Generate future dates starting from tomorrow
    last_date = pd.to_datetime(store_data["Date"]).max()
    dates = [last_date + timedelta(days=i + 1) for i in range(horizon)]

    # Simple forecast: mean ± std with slight daily variation based on day-of-week
    p50_values = []
    p10_values = []
    p90_values = []

    for i in range(horizon):
        dow = dates[i].weekday()
        dow_data = store_data[pd.to_datetime(store_data["Date"]).dt.weekday == dow]["sales"].astype(
            float
        )
        if len(dow_data) > 0:
            day_mean = float(dow_data.mean())
            day_std = float(dow_data.std()) if dow_data.std() > 0 else std_sales
        else:
            day_mean = mean_sales
            day_std = std_sales

        p50_values.append(day_mean)
        p10_values.append(max(day_mean - day_std, 0))
        p90_values.append(day_mean + day_std)

    return ForecastResponse(
        store_id=store_id,
        horizon=horizon,
        dates=[d.strftime("%Y-%m-%d") for d in dates],
        p10=[round(v, 2) for v in p10_values],
        p50=[round(v, 2) for v in p50_values],
        p90=[round(v, 2) for v in p90_values],
    )


@router.post("/", response_model=ForecastResponse)
def get_forecast(req: ForecastRequest) -> ForecastResponse:
    """Generate demand forecast for a single store.

    Args:
        req: Forecast request payload.

    Returns:
        Forecast response with selected quantiles.
    """
    cached = get_cached_forecast(store_id=req.store_id, horizon=req.horizon)
    if cached is not None:
        return ForecastResponse(**cached)

    model, dataset, reference_df = get_tft_model()

    # Fallback to statistical forecast if ML model is not loaded
    if model is None or dataset is None or reference_df is None:
        if reference_df is not None:
            response = _generate_statistical_forecast(req.store_id, req.horizon, reference_df)
            cache_forecast_payload(
                store_id=req.store_id, horizon=req.horizon, payload=response.model_dump()
            )
            return response
        raise HTTPException(
            status_code=503,
            detail="Forecast model not loaded and no reference data available. Please check logs.",
        )

    try:
        predictions, dates = model.predict_for_store(
            store_id=req.store_id,
            horizon=req.horizon,
            training_dataset=dataset,
            reference_df=reference_df,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError:
        # Fallback to statistical on model runtime errors too
        response = _generate_statistical_forecast(req.store_id, req.horizon, reference_df)
        cache_forecast_payload(
            store_id=req.store_id, horizon=req.horizon, payload=response.model_dump()
        )
        return response

    # Resolve quantile column indices dynamically from config
    q_idx = get_tft_quantile_indices()
    idx_p10 = q_idx.get("p10", 1)
    idx_p50 = q_idx.get("p50", 3)
    idx_p90 = q_idx.get("p90", 5)
    p10_raw = _to_numpy_1d(predictions[:, idx_p10].detach().cpu().numpy())
    p50_raw = _to_numpy_1d(predictions[:, idx_p50].detach().cpu().numpy())
    p90_raw = _to_numpy_1d(predictions[:, idx_p90].detach().cpu().numpy())
    p10, p50, p90 = calibrate_prediction_interval(p10=p10_raw, p50=p50_raw, p90=p90_raw)

    response = ForecastResponse(
        store_id=req.store_id,
        horizon=req.horizon,
        dates=[date.strftime("%Y-%m-%d") for date in dates],
        p10=p10.tolist(),
        p50=p50.tolist(),
        p90=p90.tolist(),
    )
    cache_forecast_payload(
        store_id=req.store_id, horizon=req.horizon, payload=response.model_dump()
    )
    return response


@router.post("/batch", response_model=ForecastBatchResponse)
async def get_forecast_batch(req: ForecastBatchRequest) -> ForecastBatchResponse:
    """Generate demand forecasts for multiple stores.

    Args:
        req: Batch forecast request payload.

    Returns:
        Batch response containing per-store forecasts.
    """

    async def _one(store_id: int) -> ForecastResponse:
        return await asyncio.to_thread(
            get_forecast,
            ForecastRequest(store_id=store_id, horizon=req.horizon),
        )

    responses = await asyncio.gather(*[_one(store_id) for store_id in req.store_ids])
    return ForecastBatchResponse(forecasts=responses)
