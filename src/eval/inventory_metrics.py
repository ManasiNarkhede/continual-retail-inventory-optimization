"""Inventory control evaluation metrics."""

from __future__ import annotations

import numpy as np


def service_level_rate(demand: np.ndarray, stock: np.ndarray) -> float:
    """Compute service level from demand and available stock arrays.

    Args:
        demand: Demand values by period.
        stock: Available stock values by period.

    Returns:
        Fraction of periods without stockout.
    """
    if demand.size == 0:
        return 1.0
    return float(np.mean(stock >= demand))


def service_level(shortfalls: np.ndarray) -> float:
    """Compute service level as fraction of no-stockout periods.

    Args:
        shortfalls: Array of shortfall quantities by period.

    Returns:
        Service level fraction.
    """
    if shortfalls.size == 0:
        return 1.0
    return float(np.mean(shortfalls <= 0))


def fill_rate(demand: np.ndarray, shortfalls: np.ndarray) -> float:
    """Compute fill rate based on fulfilled demand volume.

    Args:
        demand: Demand quantities.
        shortfalls: Unmet demand quantities.

    Returns:
        Fill rate in range [0, 1].
    """
    total_demand = float(np.sum(demand))
    if total_demand <= 0:
        return 1.0
    fulfilled = total_demand - float(np.sum(np.maximum(shortfalls, 0)))
    return float(max(0.0, fulfilled / total_demand))


def total_inventory_cost(
    excess: np.ndarray,
    shortfalls: np.ndarray,
    orders: np.ndarray,
    holding_cost: float,
    stockout_cost: float,
    order_cost: float,
) -> float:
    """Compute total inventory cost.

    Args:
        excess: Excess inventory per period.
        shortfalls: Unmet demand per period.
        orders: Binary or quantity indicator of order placement.
        holding_cost: Holding cost coefficient.
        stockout_cost: Stockout cost coefficient.
        order_cost: Fixed order cost coefficient.

    Returns:
        Scalar total cost.
    """
    return float(
        holding_cost * np.sum(np.maximum(excess, 0))
        + stockout_cost * np.sum(np.maximum(shortfalls, 0))
        + order_cost * np.sum(orders > 0)
    )
