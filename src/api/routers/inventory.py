"""Inventory action endpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter

from src.api.schemas import InventoryActionResponse, InventoryStateRequest

router = APIRouter()


def _load_store_demand_stats(store_id: int):
    """Load demand mean and std for a store from reference data."""
    ref_path = "data/processed/tft_reference.parquet"
    try:
        df = pd.read_parquet(ref_path, columns=["Store", "sales"])
        store_data = df[df["Store"] == store_id]["sales"].astype(float)
        if len(store_data) > 0:
            return float(store_data.mean()), float(store_data.std())
    except Exception:
        pass
    return 5000.0, 1500.0


def _fallback_inventory_action(req: InventoryStateRequest) -> InventoryActionResponse:
    """Return a store-aware order recommendation without a trained RL agent.

    Uses a base-stock policy: order enough to bring inventory up to
    forecasted demand + safety stock calibrated to the store's demand variance.
    """
    mean_demand, std_demand = _load_store_demand_stats(req.store_id)
    if std_demand <= 0:
        std_demand = mean_demand * 0.25

    forecast = np.asarray(req.forecast, dtype=np.float32)
    if forecast.size > 0:
        avg_demand = float(np.mean(forecast))
    else:
        avg_demand = mean_demand

    # Safety stock covers ~84% of demand spikes (z = 1.0)
    safety_stock = 1.0 * std_demand
    target_stock = avg_demand * 7 + safety_stock  # 1 week cover + safety
    order_qty = max(0.0, target_stock - req.current_stock)

    # Cap at ~3 weeks of average demand
    order_qty = min(order_qty, avg_demand * 21)

    # Confidence scales with how well current stock covers near-term demand
    days_of_cover = req.current_stock / max(avg_demand, 1.0)
    if days_of_cover >= 14:
        confidence = 0.99  # Very well stocked
    elif days_of_cover >= 7:
        confidence = 0.90
    elif days_of_cover >= 3:
        confidence = 0.75
    else:
        confidence = 0.55  # Critically low

    return InventoryActionResponse(
        store_id=req.store_id,
        order_qty=round(order_qty, 2),
        confidence=round(confidence, 2),
    )


@router.post("/action", response_model=InventoryActionResponse)
def recommend_inventory_action(req: InventoryStateRequest) -> InventoryActionResponse:
    """Return recommended order action for current state.

    Args:
        req: Current inventory state and forecast.

    Returns:
        Recommended action quantity and confidence.
    """
    # Always use the deterministic base-stock fallback rather than a
    # potentially broken RL agent so recommendations are explainable
    # and vary sensibly with store demand patterns.
    return _fallback_inventory_action(req)
