"""Property-based tests for reward and metric invariants."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.cl.metrics import compute_bwt
from src.eval.baselines import EOQBaseline
from src.eval.inventory_metrics import fill_rate, service_level_rate
from src.rl.reward import BaseReward, ShapedReward


@given(
    stock=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    demand=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    order_placed=st.booleans(),
)
@settings(max_examples=200)
def test_base_reward_nonpositive(stock: float, demand: float, order_placed: bool) -> None:
    """Base reward should always be non-positive under pure cost terms."""
    reward_fn = BaseReward(holding_cost=0.5, stockout_cost=5.0, order_cost=10.0)
    excess = max(0.0, stock - demand)
    shortfall = max(0.0, demand - stock)
    reward = reward_fn(excess_units=excess, shortfall_units=shortfall, order_placed=order_placed)
    assert reward <= 1e-9


@given(
    stock_before=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    stock_after=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    stock=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    demand=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=150)
def test_shaped_reward_bounded(
    stock_before: float,
    stock_after: float,
    stock: float,
    demand: float,
) -> None:
    """Shaped reward should stay finite and bounded for valid numeric inputs."""
    reward_fn = ShapedReward(gamma=0.99, target_stock=100.0, shaping_alpha=0.1)
    excess = max(0.0, stock - demand)
    shortfall = max(0.0, demand - stock)
    reward = reward_fn.shaped(
        stock_before=stock_before,
        stock_after=stock_after,
        excess_units=excess,
        shortfall_units=shortfall,
        order_placed=False,
    )
    assert np.isfinite(reward)
    assert abs(reward) < 1e6


@given(annual_demand=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False))
@settings(max_examples=120)
def test_eoq_positive(annual_demand: float) -> None:
    """EOQ should always be strictly positive for positive annual demand."""
    baseline = EOQBaseline(order_cost=10.0, holding_cost=182.5)
    quantity = baseline.compute_eoq(annual_demand)
    assert quantity > 0.0


@given(
    demand=st.lists(
        st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100,
    ),
    stock=st.lists(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100,
    ),
)
@settings(max_examples=120)
def test_service_and_fill_rate_in_range(demand: list[float], stock: list[float]) -> None:
    """Service level and fill rate must be probabilities in [0, 1]."""
    assume(len(demand) == len(stock))
    d = np.asarray(demand, dtype=float)
    s = np.asarray(stock, dtype=float)
    sl = service_level_rate(d, s)
    shortfall = np.maximum(0.0, d - s)
    fr = fill_rate(d, shortfall)
    assert 0.0 <= sl <= 1.0
    assert 0.0 <= fr <= 1.0


@given(t=st.integers(min_value=2, max_value=20))
@settings(max_examples=40)
def test_packnet_bwt_zero_matrix_property(t: int) -> None:
    """Zero-forgetting matrix construction yields BWT of exactly zero."""
    diagonal = np.linspace(0.6, 0.95, t)
    result_matrix = np.zeros((t, t), dtype=float)
    for idx in range(t):
        result_matrix[idx, : idx + 1] = diagonal[: idx + 1]
    assert abs(compute_bwt(result_matrix)) < 1e-12
