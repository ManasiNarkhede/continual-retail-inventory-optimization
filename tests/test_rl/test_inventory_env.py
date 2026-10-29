"""Unit tests for RetailInventoryEnv: observations, rewards, lead times and episode metrics."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.rl import inventory_env as env_module
from src.rl.inventory_env import RetailInventoryEnv
from src.rl.reward import BaseReward, MultiObjectiveReward, ShapedReward, TripleTierReward

HORIZON = 7
NO_ORDER = np.array([-1.0])
FULL_ORDER = np.array([1.0])


def _config(tmp_path: Path, **overrides) -> str:  # noqa: ANN003
    env_cfg = {
        "initial_stock": 100,
        "max_stock": 500,
        "max_order": 200,
        "lead_time": 2,
        "holding_cost": 0.5,
        "stockout_cost": 5.0,
        "order_cost": 10.0,
        "episode_length": 4,
        "forecast_horizon": HORIZON,
    }
    env_cfg.update(overrides)
    cfg = {"environment": env_cfg, "agent": {"algorithm": "PPO", "total_timesteps": 10}}
    path = tmp_path / "rl_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def _env(tmp_path: Path, demand=None, cfg: dict | None = None, **kwargs):  # noqa: ANN001, ANN202
    series = np.full(10, 50.0) if demand is None else demand
    return RetailInventoryEnv(_config(tmp_path, **(cfg or {})), demand_series=series, **kwargs)


def test_observation_layout_is_clipped_and_exposed(tmp_path: Path) -> None:
    """Reset returns horizon + 9 features clipped to [-1, 1]; get_observation matches it."""
    env = _env(tmp_path, cfg={"stockout_cost": 20.0})
    obs, info = env.reset()
    assert info == {}
    assert obs.shape == (HORIZON + 9,)
    assert obs[0] == pytest.approx(100 / 500)
    assert obs[HORIZON + 5] == 1.0  # stockout_cost / 10 = 2.0, clipped
    assert np.all((obs >= -1.0) & (obs <= 1.0))
    np.testing.assert_array_equal(env.get_observation(), obs)


# -----------------------------------------------------------------------
# Reward selection
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cfg", "reward_type", "expected"),
    [
        ({}, None, TripleTierReward),
        ({"use_multi_objective_reward": True}, None, MultiObjectiveReward),
        ({"use_multi_objective_reward": True}, "triple_tier", TripleTierReward),
        (
            {"use_multi_objective_reward": True, "reward_class": "TripleTierReward"},
            None,
            TripleTierReward,
        ),
    ],
    ids=["default", "multi-objective-flag", "reward-type-override", "reward-class-wins"],
)
def test_default_reward_selection(
    tmp_path: Path, cfg: dict, reward_type, expected
) -> None:  # noqa: ANN001
    """reward_type overrides config; an explicit reward_class beats the multi-objective flag."""
    env = _env(tmp_path, cfg=cfg, reward_type=reward_type)
    assert type(env._default_reward) is expected


def test_multi_objective_reward_type_uses_demand_units(tmp_path: Path) -> None:
    """reward_type='multi_objective' weights cost and service by risk aversion."""
    env = _env(tmp_path, reward_type="multi_objective", risk_aversion=0.25)
    assert isinstance(env._default_reward, MultiObjectiveReward)
    assert env._default_reward.cost_weight == 0.75
    assert env._default_reward.service_weight == 0.25
    env.reset()
    _, reward, _, _, info = env.step(NO_ORDER)
    # excess 50 -> triple-tier reward -25; service signal 1.0 vs target 0.95.
    assert reward == pytest.approx(0.75 * -25.0 + 0.25 * (1.0 - 0.95))
    assert info["excess"] == 50.0 and info["shortfall"] == 0.0


@pytest.mark.parametrize("reward_class", [BaseReward, ShapedReward])
def test_base_reward_classes_construct_and_score_linearly(
    tmp_path: Path, reward_class
) -> None:  # noqa: ANN001
    """Regression: reward_class "BaseReward" used to crash with TypeError (stockout_severity).

    BaseReward-family classes are built from holding/stockout/order cost only and score a
    step linearly: stock 100 vs demand 120 with a 200-unit order gives shortfall 20, so
    reward = -(5.0 * 20 + 10.0) = -110 (triple-tier would give -(5 * (e^1 - 1) + 10)).
    """
    env = _env(tmp_path, demand=np.full(10, 120.0), cfg={"reward_class": reward_class.__name__})
    assert type(env._default_reward) is reward_class
    assert (env._default_reward.holding_cost, env._default_reward.stockout_cost) == (0.5, 5.0)
    assert env._default_reward.order_cost == 10.0
    env.reset()
    _, reward, _, _, info = env.step(FULL_ORDER)
    assert (info["excess"], info["shortfall"], info["order_qty"]) == (0.0, 20.0, 200.0)
    assert reward == pytest.approx(-110.0)


def test_custom_reward_with_shaped_method(tmp_path: Path) -> None:
    """A reward_fn exposing shaped() receives stock before/after and step terms."""

    class _Shaped:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def shaped(self, **kwargs):  # noqa: ANN003, ANN201
            self.calls.append(kwargs)
            return 7.5

    reward_fn = _Shaped()
    env = _env(tmp_path, reward_fn=reward_fn)
    assert env._default_reward is None
    env.reset()
    _, reward, _, _, _ = env.step(FULL_ORDER)
    assert reward == 7.5
    assert reward_fn.calls == [
        {
            "stock_before": 100.0,
            "stock_after": 50.0,
            "excess_units": 50.0,
            "shortfall_units": 0.0,
            "order_placed": True,
        }
    ]


def test_custom_callable_reward(tmp_path: Path) -> None:
    """A plain callable reward_fn is called with excess/shortfall/order flag."""
    calls: list[dict] = []

    def _reward(**kwargs):  # noqa: ANN003, ANN202
        calls.append(kwargs)
        return -1.25

    env = _env(tmp_path, demand=np.full(10, 120.0), reward_fn=_reward)
    env.reset()
    _, reward, _, _, _ = env.step(NO_ORDER)
    assert reward == -1.25
    assert calls == [{"excess_units": 0.0, "shortfall_units": 20.0, "order_placed": False}]


def test_functional_reward_fallback(tmp_path: Path) -> None:
    """Without any reward object the functional triple-tier helper is used."""
    env = _env(tmp_path, demand=np.full(10, 120.0))
    env._default_reward = None
    env.reset()
    _, reward, _, _, _ = env.step(NO_ORDER)
    expected = TripleTierReward(0.5, 10.0, 5.0, 0.05)(0.0, 20.0, False)
    assert reward == pytest.approx(expected)


# -----------------------------------------------------------------------
# Lead time, store-open status, forecasts
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "cfg", "samples", "expected", "expected_kwargs"),
    [
        ("poisson", {"lead_time_poisson_lambda": 3.0}, [4, 0], [4, 1], {"lam": 3.0}),
        (
            "normal",
            {"lead_time_normal_std": 0.0},
            [3.4, -2.0],
            [3, 1],
            {"loc": 2.0, "scale": 1e-6},
        ),
    ],
)
def test_stochastic_lead_time_rounds_and_floors_at_one(
    tmp_path: Path,
    monkeypatch,
    mode: str,
    cfg: dict,
    samples,
    expected,
    expected_kwargs,  # noqa: ANN001
) -> None:
    """Stochastic lead-time samples are rounded to int and never drop below 1."""
    env = _env(tmp_path, cfg={"lead_time_mode": mode, **cfg})
    draws = iter(samples)
    seen: list[dict] = []

    def _draw(**kwargs):  # noqa: ANN003, ANN202
        seen.append(kwargs)
        return next(draws)

    monkeypatch.setattr(env_module.np.random, mode, _draw)
    assert [env._sample_lead_time(), env._sample_lead_time()] == expected
    assert seen == [expected_kwargs, expected_kwargs]


def test_lead_time_change_keeps_in_transit_orders(tmp_path: Path, monkeypatch) -> None:
    """A changed lead time only affects the new order; in-transit orders still arrive.

    t=0 orders 200 with lead 2 (due at t=2); t=1 orders 200 with lead 1 (also due at t=2).
    Shrinking the lead time used to evict the first order from the queue.
    """
    env = _env(tmp_path, demand=np.zeros(10), cfg={"initial_stock": 0})
    env.reset()
    leads = iter([2, 1, 3])
    monkeypatch.setattr(env, "_sample_lead_time", lambda: next(leads))
    infos = [env.step(action)[4] for action in (FULL_ORDER, FULL_ORDER, NO_ORDER)]
    assert [info["lead_time"] for info in infos] == [2.0, 1.0, 3.0]
    assert [info["stock"] for info in infos] == [0.0, 0.0, 400.0]
    assert list(env.order_queue) == [0.0, 0.0, 0.0]


def test_store_open_series_drives_observation(tmp_path: Path) -> None:
    """Observation slot horizon+2 mirrors the cyclic store_open_series."""
    env = _env(tmp_path, store_open_series=[1, 0, 1])
    obs, _ = env.reset()
    assert env._get_store_open(4) == 0.0
    values = [obs[HORIZON + 2]]
    for _ in range(3):
        obs, *_ = env.step(NO_ORDER)
        values.append(obs[HORIZON + 2])
    assert values == [1.0, 0.0, 1.0, 1.0]


def test_default_store_open_closed_on_day_six(tmp_path: Path) -> None:
    """Without a series the store is closed on every 7th day only."""
    env = _env(tmp_path)
    assert [env._get_store_open(t) for t in range(8)] == [1, 1, 1, 1, 1, 1, 0, 1]


def test_quantile_forecast_padding_and_uncertainty(tmp_path: Path) -> None:
    """Short p50 vectors are padded with their last value; width is mean(p90-p10)."""
    demand = np.full(10, 100.0)
    p10 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    p90 = p10 + np.array([10.0, 20.0, -5.0, 0.0, 10.0, 20.0, 10.0, 999.0])
    p50 = np.array([30.0, 40.0, 50.0])
    seen_t: list[int] = []

    def _quantiles(t: int):  # noqa: ANN202
        seen_t.append(t)
        return p10, p50, p90

    env = _env(tmp_path, demand=demand, forecast_quantiles_fn=_quantiles)
    obs, _ = env.reset()
    forecast, width = env._get_forecast_with_uncertainty()
    np.testing.assert_allclose(forecast, [30, 40, 50, 50, 50, 50, 50])
    assert width == pytest.approx((10 + 20 + 0 + 0 + 10 + 20 + 10) / 7)
    np.testing.assert_allclose(obs[1 : HORIZON + 1], np.array(forecast) / 100.0, rtol=1e-6)
    assert obs[HORIZON + 1] == pytest.approx(width / 100.0, rel=1e-6)
    assert seen_t[0] == 0


def test_quantile_forecast_long_and_empty_inputs(tmp_path: Path) -> None:
    """Long p50 is truncated; empty quantiles give zero forecast and width."""
    long_p50 = np.arange(10, dtype=float)
    env = _env(tmp_path, forecast_quantiles_fn=lambda t: (long_p50, long_p50, long_p50 + 2))
    forecast, width = env._get_forecast_with_uncertainty()
    np.testing.assert_allclose(forecast, np.arange(HORIZON))
    assert width == pytest.approx(2.0)

    empty = np.array([], dtype=float)
    env = _env(tmp_path, forecast_quantiles_fn=lambda t: (empty, empty, empty))
    forecast, width = env._get_forecast_with_uncertainty()
    np.testing.assert_allclose(forecast, np.zeros(HORIZON))
    assert width == 0.0


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (list(range(1, 11)), list(range(1, 8))),
        ([4.0, 6.0], [4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),
        ([], [0.0] * HORIZON),
    ],
)
def test_point_forecast_fn_truncates_pads_or_zeros(
    tmp_path: Path, values, expected  # noqa: ANN001
) -> None:
    """forecast_fn output is fitted to the configured horizon."""
    env = _env(tmp_path, forecast_fn=lambda t: values)
    np.testing.assert_allclose(env._get_forecast(), expected)
    _, width = env._get_forecast_with_uncertainty()
    assert width == 0.0  # constant demand series -> zero std


def test_default_forecast_wraps_demand_series(tmp_path: Path) -> None:
    """Without a forecast callback the demand series is read cyclically."""
    env = _env(tmp_path, demand=np.arange(1.0, 5.0))
    np.testing.assert_allclose(env._get_forecast(), [1, 2, 3, 4, 1, 2, 3])


# -----------------------------------------------------------------------
# Episode metrics
# -----------------------------------------------------------------------


def test_episode_metrics_without_orders(tmp_path: Path) -> None:
    """Hand-computed metrics for a no-order episode with demand 50 and stock 100."""
    env = _env(tmp_path)
    env.reset()
    assert (env.service_level(), env.fill_rate(), env.total_cost()) == (1.0, 1.0, 0.0)
    flags = []
    terminated = False
    while not terminated:
        _, _, terminated, truncated, info = env.step(NO_ORDER)
        assert truncated is False
        flags.append(info["service_level"])
    assert flags == [1.0, 1.0, 0.0, 0.0]
    assert env.t == 4
    assert env.service_level() == pytest.approx(0.5)
    assert env.fill_rate() == pytest.approx(0.5)
    assert env.total_cost() == pytest.approx(0.5 * 50 + 5.0 * 100)


def test_orders_arrive_after_lead_time(tmp_path: Path) -> None:
    """A full order placed at t=0 arrives at the start of step t=2 (lead time 2)."""
    env = _env(tmp_path)
    env.reset()
    stocks = [env.step(a)[4]["stock"] for a in (FULL_ORDER, NO_ORDER, NO_ORDER)]
    # 100-50=50, 50-50=0, then +200 arrival -50 = 150.
    assert stocks == [50.0, 0.0, 150.0]
    assert env.total_orders == 10.0


def test_fill_rate_counts_arriving_order_once(tmp_path: Path) -> None:
    """Regression: an order arriving this step is counted once in fulfilled demand.

    Zero initial stock; action 0.0 orders 0.5 * max_order = 30 units at t=0 (zero demand),
    which arrive at t=1 (lead time 1) against demand 50: 30 of 50 units are filled, so
    fill rate = 0.6. The old formula added the arrival twice and reported 1.0.
    """
    env = _env(
        tmp_path,
        demand=np.array([0.0, 50.0]),
        cfg={"initial_stock": 0, "lead_time": 1, "max_order": 60},
    )
    env.reset()
    assert env.step(np.array([0.0]))[4]["order_qty"] == 30.0
    info = env.step(NO_ORDER)[4]
    assert (info["shortfall"], info["stock"]) == (20.0, 0.0)
    assert env.fill_rate() == pytest.approx(0.6)


# -----------------------------------------------------------------------
# Default demand series loading
# -----------------------------------------------------------------------


def _fake_registry(monkeypatch, series) -> list:  # noqa: ANN001
    calls: list = []
    module = types.ModuleType("src.api.model_registry")

    def get_reference_demand_series(min_length: int):  # noqa: ANN202
        calls.append(min_length)
        return series

    module.get_reference_demand_series = get_reference_demand_series
    monkeypatch.setitem(sys.modules, "src.api.model_registry", module)
    return calls


def test_default_demand_series_loaded_from_registry(tmp_path: Path, monkeypatch) -> None:
    """Missing demand_series falls back to the registry reference series."""
    calls = _fake_registry(monkeypatch, np.array([5, 6, 7], dtype=int))
    env = RetailInventoryEnv(_config(tmp_path))
    assert calls == [365]
    assert env.demand_series.dtype == float
    np.testing.assert_allclose(env.demand_series, [5.0, 6.0, 7.0])


def test_default_demand_series_unavailable_raises(tmp_path: Path, monkeypatch) -> None:
    """An empty registry series raises FileNotFoundError."""
    _fake_registry(monkeypatch, np.array([]))
    with pytest.raises(FileNotFoundError, match="Reference demand series"):
        RetailInventoryEnv(_config(tmp_path))
