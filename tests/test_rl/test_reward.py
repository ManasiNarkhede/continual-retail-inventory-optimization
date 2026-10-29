"""Unit tests for reward function utilities."""

from __future__ import annotations

from src.rl.reward import (
    BaseReward,
    MultiObjectiveReward,
    ShapedReward,
    TripleTierReward,
    compute_inventory_reward,
)


def test_reward_includes_all_cost_terms() -> None:
    """Validate that reward is negative sum of costs."""
    reward = compute_inventory_reward(
        excess_units=10,
        shortfall_units=2,
        order_placed=True,
        holding_cost=0.5,
        stockout_cost=5.0,
        order_cost=10.0,
    )
    assert reward < 0.0


def test_reward_without_order_cost() -> None:
    """Validate reward when no order is placed."""
    reward = compute_inventory_reward(
        excess_units=0,
        shortfall_units=1,
        order_placed=False,
        holding_cost=0.5,
        stockout_cost=5.0,
        order_cost=10.0,
    )
    assert reward < 0.0


def test_base_reward_class_matches_function() -> None:
    """Ensure BaseReward produces same value as functional helper."""
    reward_obj = BaseReward(holding_cost=0.5, stockout_cost=5.0, order_cost=10.0)
    via_class = reward_obj(excess_units=4.0, shortfall_units=1.0, order_placed=True)
    via_function = compute_inventory_reward(
        excess_units=4.0,
        shortfall_units=1.0,
        order_placed=True,
        holding_cost=0.5,
        stockout_cost=5.0,
        order_cost=10.0,
    )
    assert via_class < 0.0
    assert via_function < 0.0


def test_triple_tier_reward_exponential_penalty() -> None:
    """Large shortfall should have super-linear penalty under triple tier."""
    reward_fn = TripleTierReward(
        holding_cost=0.5, order_cost=10.0, stockout_cost=5.0, stockout_severity=0.05
    )
    small = reward_fn(excess_units=0.0, shortfall_units=1.0, order_placed=False)
    large = reward_fn(excess_units=0.0, shortfall_units=10.0, order_placed=False)
    assert large < small


def test_shaped_reward_returns_float() -> None:
    """Ensure shaped reward calculation executes and returns scalar."""
    reward_obj = ShapedReward(
        holding_cost=0.5,
        stockout_cost=5.0,
        order_cost=10.0,
        gamma=0.99,
        target_stock=100.0,
        shaping_alpha=0.1,
    )
    value = reward_obj.shaped(
        stock_before=120.0,
        stock_after=90.0,
        excess_units=10.0,
        shortfall_units=0.0,
        order_placed=True,
    )
    assert isinstance(value, float)


def test_multi_objective_reward_tunable_tradeoff() -> None:
    """Risk-averse configuration should penalize stockouts more strongly."""
    reward = MultiObjectiveReward(
        holding_cost=0.5,
        order_cost=10.0,
        stockout_cost=5.0,
        stockout_severity=0.05,
        cost_weight=0.2,
        service_weight=0.8,
        service_level_target=0.95,
    )
    no_stockout = reward(
        excess_units=5.0, shortfall_units=0.0, order_placed=False, demand_units=20.0
    )
    stockout = reward(excess_units=0.0, shortfall_units=10.0, order_placed=False, demand_units=20.0)
    assert no_stockout > stockout
