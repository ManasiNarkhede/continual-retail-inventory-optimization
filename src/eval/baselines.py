"""Baseline models for forecast and inventory comparisons."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from src.eval.forecast_metrics import mase, rmse, smape
from src.eval.harness import ModelResult


def _sales_series(val_df: pd.DataFrame) -> np.ndarray:
    """Extract demand series from evaluation dataframe."""
    if "Sales" in val_df.columns:
        return val_df["Sales"].to_numpy(dtype=float)
    if "sales" in val_df.columns:
        return val_df["sales"].to_numpy(dtype=float)
    raise KeyError("Validation dataframe must contain 'Sales' or 'sales'.")


def _simulate_inventory(
    demand: np.ndarray,
    order_fn: Callable[[int, float], float],
    initial_stock: float = 120.0,
    lead_time: int = 2,
    holding_cost: float = 0.5,
    stockout_cost: float = 5.0,
    order_cost: float = 10.0,
) -> tuple[float, float, float]:
    """Run a rolling inventory simulation and return (service, fill, cost)."""
    n = len(demand)
    stock = float(initial_stock)
    pipeline = [0.0] * max(int(lead_time), 1)

    stockout_days = 0
    total_demand = 0.0
    total_fulfilled = 0.0
    total_cost = 0.0

    for t in range(n):
        arriving = pipeline.pop(0)
        stock += arriving

        d_t = float(max(demand[t], 0.0))
        fulfilled = min(stock, d_t)
        shortfall = max(0.0, d_t - stock)
        stock = max(0.0, stock - d_t)

        if shortfall > 0.0:
            stockout_days += 1
        total_demand += d_t
        total_fulfilled += fulfilled

        inv_position = stock + sum(pipeline)
        order_qty = float(max(0.0, order_fn(t=t, inv_position=inv_position)))
        pipeline.append(order_qty)

        total_cost += holding_cost * stock
        total_cost += stockout_cost * shortfall
        if order_qty > 0.5:
            total_cost += order_cost

    service = float((n - stockout_days) / max(n, 1))
    fill = float(min(1.0, total_fulfilled / max(total_demand, 1e-8)))
    return service, fill, float(round(total_cost, 2))


class NaiveSeasonalBaseline:
    """Seasonal naive forecaster using lag-m copy."""

    def __init__(self, seasonality: int = 7) -> None:
        self.seasonality = int(seasonality)

    def predict(self, values: np.ndarray) -> np.ndarray:
        """Create lag-based predictions."""
        preds = np.full(values.shape, np.nan, dtype=float)
        preds[self.seasonality :] = values[: -self.seasonality]
        return preds

    def evaluate(self, val_df: pd.DataFrame) -> ModelResult:
        """Evaluate naive baseline on a validation frame."""
        y_true = _sales_series(val_df)
        y_pred = self.predict(y_true)
        mask = ~np.isnan(y_pred)

        warm_start = np.where(np.isnan(y_pred), np.mean(y_true[: self.seasonality]), y_pred)

        def _order_fn(t: int, inv_position: float) -> float:
            return max(0.0, float(warm_start[t]) - inv_position)

        service, fill, total_cost = _simulate_inventory(y_true, _order_fn)

        return ModelResult(
            model_name=f"Naive seasonal (lag={self.seasonality})",
            smape=float(smape(y_true[mask], y_pred[mask])) if np.any(mask) else 0.0,
            rmse=float(rmse(y_true[mask], y_pred[mask])) if np.any(mask) else 0.0,
            mase=(
                float(mase(y_true[mask], y_pred[mask], m=self.seasonality)) if np.any(mask) else 0.0
            ),
            coverage=0.0,
            service_level=service,
            fill_rate=fill,
            total_cost=total_cost,
            latency_ms=0.01,
        )


class EOQBaseline:
    """Classical EOQ inventory-policy baseline."""

    def __init__(
        self,
        order_cost: float = 10.0,
        holding_cost: float = 182.5,
        lead_time: int = 2,
        service_z: float = 1.645,
    ) -> None:
        self.order_cost = float(order_cost)
        self.holding_cost = float(holding_cost)
        self.lead_time = int(lead_time)
        self.service_z = float(service_z)

    def compute_eoq(self, annual_demand: float) -> float:
        """Compute EOQ quantity."""
        return float(np.sqrt(2.0 * annual_demand * self.order_cost / max(self.holding_cost, 1e-8)))

    def compute_rop(self, demand_mean: float, demand_std: float) -> float:
        """Compute reorder point with safety stock."""
        return float(
            demand_mean * self.lead_time + self.service_z * demand_std * np.sqrt(self.lead_time)
        )

    def evaluate(self, val_df: pd.DataFrame) -> ModelResult:
        """Simulate EOQ policy over validation demand."""
        demand = _sales_series(val_df)
        mean = float(np.mean(demand))
        std = float(np.std(demand))
        annual = mean * 365.0
        q_star = self.compute_eoq(annual)
        reorder_point = self.compute_rop(mean, std)

        def _order_fn(t: int, inv_position: float) -> float:
            if inv_position <= reorder_point:
                return q_star
            return 0.0

        service, fill, total_cost = _simulate_inventory(
            demand=demand,
            order_fn=_order_fn,
            initial_stock=max(2.0 * q_star, reorder_point),
            lead_time=self.lead_time,
            holding_cost=self.holding_cost / 365.0,
            stockout_cost=5.0,
            order_cost=self.order_cost,
        )

        return ModelResult(
            model_name="EOQ policy (baseline)",
            smape=0.0,
            rmse=0.0,
            mase=0.0,
            coverage=0.0,
            service_level=service,
            fill_rate=fill,
            total_cost=float(round(total_cost, 2)),
            latency_ms=0.05,
        )


class BaseStockBaseline:
    """Order-up-to baseline calibrated on historical demand statistics."""

    def __init__(
        self,
        lead_time: int = 2,
        service_z: float = 1.645,
        holding_cost: float = 0.5,
        stockout_cost: float = 5.0,
        order_cost: float = 10.0,
    ) -> None:
        self.lead_time = int(lead_time)
        self.service_z = float(service_z)
        self.holding_cost = float(holding_cost)
        self.stockout_cost = float(stockout_cost)
        self.order_cost = float(order_cost)

    def evaluate(self, val_df: pd.DataFrame) -> ModelResult:
        """Evaluate base-stock inventory policy on demand series."""
        demand = _sales_series(val_df)
        if demand.size == 0:
            return ModelResult(model_name="Base-stock policy (baseline)")

        mean = float(np.mean(demand))
        std = float(np.std(demand))
        base_stock_level = mean * self.lead_time + self.service_z * std * math.sqrt(self.lead_time)
        base_stock_level = max(base_stock_level, mean)

        def _order_fn(t: int, inv_position: float) -> float:
            return max(0.0, base_stock_level - inv_position)

        service, fill, total_cost = _simulate_inventory(
            demand=demand,
            order_fn=_order_fn,
            initial_stock=max(base_stock_level, mean * 2.0),
            lead_time=self.lead_time,
            holding_cost=self.holding_cost,
            stockout_cost=self.stockout_cost,
            order_cost=self.order_cost,
        )

        return ModelResult(
            model_name="Base-stock policy (baseline)",
            smape=0.0,
            rmse=0.0,
            mase=0.0,
            coverage=0.0,
            service_level=service,
            fill_rate=fill,
            total_cost=total_cost,
            latency_ms=0.03,
            extra={"base_stock_level": float(base_stock_level)},
        )
