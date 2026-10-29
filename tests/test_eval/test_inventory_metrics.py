"""Unit tests for inventory metric functions."""

from __future__ import annotations

import numpy as np

from src.eval.inventory_metrics import (
    fill_rate,
    service_level,
    service_level_rate,
    total_inventory_cost,
)


def test_service_level_basic_case() -> None:
    """Ensure service level matches non-stockout fraction."""
    shortfalls = np.array([0.0, 2.0, 0.0, 1.0, 0.0])
    assert np.isclose(service_level(shortfalls), 3 / 5)


def test_fill_rate_bounds() -> None:
    """Ensure fill rate result is bounded in [0, 1]."""
    demand = np.array([10.0, 10.0, 10.0])
    shortfalls = np.array([0.0, 2.0, 1.0])
    value = fill_rate(demand, shortfalls)
    assert 0.0 <= value <= 1.0


def test_total_inventory_cost_combines_terms() -> None:
    """Ensure total cost includes holding, stockout, and order terms."""
    excess = np.array([5.0, 0.0, 2.0])
    shortfalls = np.array([0.0, 3.0, 0.0])
    orders = np.array([1.0, 0.0, 1.0])
    cost = total_inventory_cost(
        excess=excess,
        shortfalls=shortfalls,
        orders=orders,
        holding_cost=0.5,
        stockout_cost=5.0,
        order_cost=10.0,
    )
    expected = (5.0 + 2.0) * 0.5 + 3.0 * 5.0 + 2 * 10.0
    assert np.isclose(cost, expected)


def test_service_level_rate_matches_expected() -> None:
    """Ensure demand/stock-based service level computes correctly."""
    demand = np.array([10.0, 20.0, 15.0, 12.0])
    stock = np.array([10.0, 18.0, 15.0, 15.0])
    assert np.isclose(service_level_rate(demand, stock), 3 / 4)
