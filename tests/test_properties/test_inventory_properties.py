"""Property-based tests for inventory metric invariants."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval.inventory_metrics import fill_rate, service_level_rate, total_inventory_cost


@settings(max_examples=40)
@given(
    demand=st.lists(
        st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=40,
    ),
    stock=st.lists(
        st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=40,
    ),
)
def test_service_level_rate_is_probability(demand: list[float], stock: list[float]) -> None:
    """Service level must be in [0, 1] for valid nonnegative arrays."""
    size = min(len(demand), len(stock))
    value = service_level_rate(
        np.asarray(demand[:size], dtype=float), np.asarray(stock[:size], dtype=float)
    )
    assert 0.0 <= value <= 1.0


@settings(max_examples=40)
@given(
    demand=st.lists(
        st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=40,
    ),
    shortfall=st.lists(
        st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=40,
    ),
)
def test_fill_rate_is_probability(demand: list[float], shortfall: list[float]) -> None:
    """Fill rate must stay in [0, 1] for nonnegative demand/shortfall."""
    size = min(len(demand), len(shortfall))
    value = fill_rate(
        np.asarray(demand[:size], dtype=float), np.asarray(shortfall[:size], dtype=float)
    )
    assert 0.0 <= value <= 1.0


@settings(max_examples=30)
@given(
    excess=st.lists(
        st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=30,
    ),
    shortfall=st.lists(
        st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=30,
    ),
    orders=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=30,
    ),
)
def test_total_inventory_cost_non_negative(
    excess: list[float], shortfall: list[float], orders: list[float]
) -> None:
    """Total inventory cost must remain nonnegative."""
    size = min(len(excess), len(shortfall), len(orders))
    cost = total_inventory_cost(
        excess=np.asarray(excess[:size], dtype=float),
        shortfalls=np.asarray(shortfall[:size], dtype=float),
        orders=np.asarray(orders[:size], dtype=float),
        holding_cost=0.5,
        stockout_cost=5.0,
        order_cost=10.0,
    )
    assert cost >= 0.0
