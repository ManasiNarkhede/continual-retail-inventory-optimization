"""Pydantic request and response schemas for API endpoints."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Forecast request payload.

    Attributes:
        store_id: Rossmann store identifier.
        horizon: Forecast horizon in days.
    """

    store_id: int = Field(..., description="Rossmann store identifier (1-1115)")
    horizon: int = Field(7, ge=1, le=28, description="Forecast horizon in days")


class ForecastResponse(BaseModel):
    """Forecast response payload."""

    store_id: int
    horizon: int
    dates: List[str]
    p10: List[float]
    p50: List[float]
    p90: List[float]


class ForecastBatchRequest(BaseModel):
    """Batch forecast request payload."""

    store_ids: List[int] = Field(..., min_length=1, description="Store identifiers")
    horizon: int = Field(7, ge=1, le=28, description="Forecast horizon in days")


class ForecastBatchResponse(BaseModel):
    """Batch forecast response payload."""

    forecasts: List[ForecastResponse]


class InventoryStateRequest(BaseModel):
    """Inventory action request payload."""

    store_id: int
    current_stock: float
    forecast: List[float]


class InventoryActionResponse(BaseModel):
    """Inventory action recommendation response."""

    store_id: int
    order_qty: float
    confidence: float


class SimulationRequest(BaseModel):
    """Simulation execution request payload."""

    store_id: int
    n_days: int = Field(30, ge=1, le=365)
    initial_stock: float = 100.0
    algorithm: str = "PPO"
    risk_aversion: float = Field(0.5, ge=0.0, le=1.0)


class SimulationStep(BaseModel):
    """Single simulation step details."""

    day: int
    stock: float
    demand: float
    order_qty: float
    reward: float
    service_level: float = 1.0
    status: str = "Good"  # Good, Stockout, Overstock


class SimulationResponse(BaseModel):
    """Simulation response payload."""

    store_id: int
    steps: List[SimulationStep]
    total_cost: float
    service_level: float
    risk_aversion: float = 0.5


class CLMetricsResponse(BaseModel):
    """Continual learning metrics response."""

    bwt: float
    fwt: float
    plasticity: float
    stability: float
    experience_accs: List[float]
    strategy: str


class KpiResponse(BaseModel):
    """Dashboard KPI response."""

    smape: float
    baseline_smape: float
    smape_uplift_vs_baseline: float
    rmse: float
    coverage_80pct: float
    # Inventory KPIs come from RL simulation only; None when no RL agent is loaded.
    service_level: Optional[float] = None
    fill_rate: Optional[float] = None
    total_cost: Optional[float] = None
    rl_agent_loaded: bool = False
    cost_saved: float
    bwt: float
    model_version: str
    timestamp_utc: str
    is_demo: bool

    model_config = {"protected_namespaces": ()}


class StoreListResponse(BaseModel):
    """Store list response payload."""

    stores: List[int]
    forecastable_stores: List[int]
    total: int
    forecastable_total: int
