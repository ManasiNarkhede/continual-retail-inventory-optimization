"""Simulation endpoints for rollout trajectories."""

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request, Response

from src.api.limiter import limiter
from src.api.schemas import SimulationRequest, SimulationResponse, SimulationStep

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
    # Fallback defaults
    return 5000.0, 1500.0


def _generate_mock_simulation(req: SimulationRequest) -> SimulationResponse:
    """Generate a simulated trajectory using a base-stock inventory policy.

    Safety stock scales with risk_aversion (0 = cost focus, 1 = service focus).
    """
    # Load store-specific demand stats
    mean_demand, std_demand = _load_store_demand_stats(req.store_id)
    if std_demand <= 0:
        std_demand = mean_demand * 0.25

    # Service-level oriented z_score: even cost focus gets some safety
    z_score = 0.5 + req.risk_aversion * 3.5
    lead_time = 1
    base_stock = mean_demand * lead_time + z_score * std_demand * np.sqrt(lead_time)
    base_stock = max(base_stock, mean_demand * 1.2)

    # Cost factors
    holding_cost_factor = 1.0 - req.risk_aversion * 0.5
    stockout_cost_factor = 0.5 + req.risk_aversion * 1.5

    steps: list[SimulationStep] = []
    stock = max(float(req.initial_stock), base_stock * 0.5)
    service_level_total = 0.0
    total_cost = 0.0

    for day in range(req.n_days):
        day_seed = req.store_id * 10000 + day
        day_rng = np.random.RandomState(day_seed)
        demand = max(0.0, float(day_rng.normal(mean_demand, std_demand)))

        order_qty = max(0.0, base_stock - stock)
        stock += order_qty

        fulfilled = min(stock, demand)
        stock -= fulfilled
        unmet = demand - fulfilled

        holding_cost = stock * 0.1 * holding_cost_factor
        stockout_cost = unmet * 2.0 * stockout_cost_factor
        ordering_cost = order_qty * 0.05
        day_cost = holding_cost + stockout_cost + ordering_cost
        total_cost += day_cost

        day_sl = 1.0 if unmet <= 0 else fulfilled / demand
        service_level_total += day_sl

        if unmet > 0:
            status = "Stockout"
        elif stock > mean_demand * 1.5:
            status = "Overstock"
        else:
            status = "Good"

        steps.append(
            SimulationStep(
                day=day + 1,
                stock=max(0.0, stock),
                demand=demand,
                order_qty=order_qty,
                reward=-(day_cost),
                service_level=day_sl,
                status=status,
            )
        )

    avg_service_level = service_level_total / max(len(steps), 1)

    return SimulationResponse(
        store_id=req.store_id,
        steps=steps,
        total_cost=round(total_cost, 2),
        service_level=round(avg_service_level, 4),
        risk_aversion=float(req.risk_aversion),
    )


@router.post("/", response_model=SimulationResponse)
@limiter.limit("20/minute")
def run_simulation(
    request: Request,
    response: Response,  # noqa: ARG001
    req: SimulationRequest,
) -> SimulationResponse:
    """Simulate an inventory trajectory using a base-stock policy.

    Args:
        req: Simulation request payload.

    Returns:
        Simulation response with daily steps and summary metrics.
    """
    # Always use the deterministic base-stock policy for consistent,
    # explainable results rather than a potentially broken RL agent.
    return _generate_mock_simulation(req)
