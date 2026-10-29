"""Custom Gymnasium environment for retail inventory control.

The observation space includes:
- Normalised current stock level
- Forecast demand vector (horizon steps)
- Uncertainty width (p90 − p10) normalised by max demand
- Store open status (1.0 if open, 0.0 if closed)
- Lead time (normalised)
- Holding and stockout costs (normalised)
- Cyclic day-of-week encoding (sin, cos)
- Bi-weekly phase flag

The environment uses the ``TripleTierReward`` by default, which applies
an exponential stockout penalty.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from configs.validator import load_yaml_config, validate_config_schema
from src.rl.reward import (
    BaseReward,
    MultiObjectiveReward,
    ShapedReward,
    TripleTierReward,
    compute_inventory_reward,
)

_REWARD_REGISTRY = {
    "BaseReward": BaseReward,
    "ShapedReward": ShapedReward,
    "TripleTierReward": TripleTierReward,
    "MultiObjectiveReward": MultiObjectiveReward,
}


class RetailInventoryEnv(gym.Env):
    """Gymnasium environment for single-store inventory optimization.

    Args:
        config_path: Path to rl_config.yaml.
        demand_series: Historical demand series.
        forecast_fn: Optional forecast callback returning horizon vector.
        forecast_quantiles_fn: Optional callback returning (p10, p50, p90)
            arrays for uncertainty-aware decision making.
        reward_fn: Optional reward object with ``__call__`` or ``shaped``.
        store_open_series: Optional binary series indicating store open status.
    """

    metadata = {"render_modes": []}

    # Number of auxiliary features beyond the forecast horizon:
    # stock, uncertainty_width, open_status, lead_time, h_cost, s_cost,
    # dow_sin, dow_cos, promo_phase
    _AUX_FEATURES = 9

    def __init__(
        self,
        config_path: str = "configs/rl_config.yaml",
        demand_series: np.ndarray | None = None,
        forecast_fn: Callable[[int], np.ndarray] | None = None,
        forecast_quantiles_fn: (
            Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]] | None
        ) = None,
        reward_fn: Any | None = None,
        store_open_series: np.ndarray | None = None,
        risk_aversion: float | None = None,
        reward_type: str | None = None,
    ) -> None:
        """Initialize environment state and spaces.

        Args:
            config_path: Path to RL configuration YAML.
            demand_series: Optional historical demand sequence.
            forecast_fn: Optional callable returning point-forecast vector.
            forecast_quantiles_fn: Optional callable returning (p10, p50, p90) arrays.
            reward_fn: Optional reward object with ``__call__`` or ``shaped``.
            store_open_series: Optional binary open/closed status per timestep.
            risk_aversion: Optional override for risk-aversion coefficient.
            reward_type: Optional reward selector among ``auto``, ``multi_objective``,
                ``triple_tier``.
        """
        super().__init__()
        validate_config_schema(config_path, "configs/schemas/rl_schema.json")
        self.cfg = load_yaml_config(config_path).get("environment", {})

        self.max_stock = float(self.cfg["max_stock"])
        self.max_order = float(self.cfg["max_order"])
        self.lead_time = int(self.cfg["lead_time"])
        self.lead_time_mode = str(self.cfg.get("lead_time_mode", "fixed")).lower()
        self.lead_time_poisson_lambda = float(
            self.cfg.get("lead_time_poisson_lambda", max(self.lead_time, 1))
        )
        self.lead_time_normal_mean = float(self.cfg.get("lead_time_normal_mean", self.lead_time))
        self.lead_time_normal_std = float(self.cfg.get("lead_time_normal_std", 0.5))
        self.holding_cost = float(self.cfg["holding_cost"])
        self.stockout_cost = float(self.cfg["stockout_cost"])
        self.order_cost = float(self.cfg["order_cost"])
        self.risk_aversion = (
            float(risk_aversion)
            if risk_aversion is not None
            else float(self.cfg.get("risk_aversion", 0.5))
        )
        self.service_level_target = float(self.cfg.get("service_level_target", 0.95))
        self.episode_length = int(self.cfg["episode_length"])
        self.horizon = int(self.cfg["forecast_horizon"])

        self.demand_series = (
            np.asarray(demand_series, dtype=float)
            if demand_series is not None
            else self._load_default_demand_series()
        )
        self.forecast_fn = forecast_fn
        self.forecast_quantiles_fn = forecast_quantiles_fn
        self.reward_fn = reward_fn
        self.store_open_series = (
            np.asarray(store_open_series, dtype=float) if store_open_series is not None else None
        )

        selected_reward = str(reward_type or "auto").lower()
        configured_reward = str(self.cfg.get("reward_class", "")).strip()
        if configured_reward == "":
            configured_reward = (
                "MultiObjectiveReward"
                if bool(self.cfg.get("use_multi_objective_reward", False))
                else "TripleTierReward"
            )

        if selected_reward == "multi_objective":
            configured_reward = "MultiObjectiveReward"
        elif selected_reward == "triple_tier":
            configured_reward = "TripleTierReward"

        if self.reward_fn is None:
            reward_class = _REWARD_REGISTRY.get(configured_reward, TripleTierReward)
            if reward_class is MultiObjectiveReward:
                self._default_reward = reward_class(
                    holding_cost=self.holding_cost,
                    order_cost=self.order_cost,
                    stockout_cost=self.stockout_cost,
                    stockout_severity=float(self.cfg.get("stockout_severity", 0.05)),
                    cost_weight=float(1.0 - self.risk_aversion),
                    service_weight=float(self.risk_aversion),
                    service_level_target=self.service_level_target,
                )
            elif reward_class is ShapedReward:
                self._default_reward = reward_class(
                    holding_cost=self.holding_cost,
                    stockout_cost=self.stockout_cost,
                    order_cost=self.order_cost,
                )
            else:
                self._default_reward = reward_class(
                    holding_cost=self.holding_cost,
                    order_cost=self.order_cost,
                    stockout_cost=self.stockout_cost,
                    stockout_severity=float(self.cfg.get("stockout_severity", 0.05)),
                )
        else:
            self._default_reward = None

        obs_dim = self.horizon + self._AUX_FEATURES
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._reset_state()

    def _load_default_demand_series(self) -> np.ndarray:
        """Load default demand profile from registry-backed artifacts."""
        from src.api.model_registry import get_reference_demand_series

        series = get_reference_demand_series(min_length=max(self.episode_length, 365))
        if isinstance(series, np.ndarray) and series.size > 0:
            return series.astype(float)
        raise FileNotFoundError(
            "Reference demand series unavailable for RetailInventoryEnv startup."
        )

    def _sample_lead_time(self) -> int:
        """Sample lead time from configured distribution."""
        mode = self.lead_time_mode
        if mode == "poisson":
            sample = int(np.random.poisson(lam=max(self.lead_time_poisson_lambda, 1e-3)))
            return max(1, sample)
        if mode == "normal":
            sample = int(
                round(
                    np.random.normal(
                        loc=self.lead_time_normal_mean, scale=max(self.lead_time_normal_std, 1e-6)
                    )
                )
            )
            return max(1, sample)
        return max(1, int(self.lead_time))

    def _reset_state(self) -> None:
        """Reset internal state variables."""
        self.t = 0
        self.stock = float(self.cfg["initial_stock"])
        self.current_lead_time = self._sample_lead_time()
        self.order_queue: deque[float] = deque(
            [0.0] * self.current_lead_time, maxlen=self.current_lead_time
        )
        self.total_holding = 0.0
        self.total_stockout = 0.0
        self.total_orders = 0.0
        self.stockout_days = 0
        self.total_demand = 0.0
        self.total_fulfilled = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed.
            options: Unused options dictionary.

        Returns:
            Initial observation and info dictionary.
        """
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Scaled action in range [-1, 1].

        Returns:
            Observation, reward, terminated, truncated, info tuple.
        """
        arriving_order = self.order_queue.popleft()
        self.stock += arriving_order
        stock_before = float(self.stock)

        demand = self._get_demand(self.t)
        excess = max(0.0, self.stock - demand)
        shortfall = max(0.0, demand - self.stock)
        self.stock = excess

        clipped_action = float(np.clip(float(action[0]), -1.0, 1.0))
        order_qty = (clipped_action + 1.0) / 2.0 * self.max_order
        order_placed = order_qty > 0.5

        sampled_lead = self._sample_lead_time()
        if sampled_lead != self.current_lead_time:
            old_queue = list(self.order_queue)
            if sampled_lead > self.current_lead_time:
                old_queue.extend([0.0] * (sampled_lead - self.current_lead_time))
            else:
                old_queue = old_queue[:sampled_lead]
            self.current_lead_time = sampled_lead
            self.order_queue = deque(old_queue, maxlen=sampled_lead)

        if self.reward_fn is not None and hasattr(self.reward_fn, "shaped"):
            reward = float(
                self.reward_fn.shaped(
                    stock_before=stock_before,
                    stock_after=self.stock,
                    excess_units=excess,
                    shortfall_units=shortfall,
                    order_placed=order_placed,
                )
            )
        elif self.reward_fn is not None:
            reward = float(
                self.reward_fn(
                    excess_units=excess,
                    shortfall_units=shortfall,
                    order_placed=order_placed,
                )
            )
        elif self._default_reward is not None:
            if isinstance(self._default_reward, MultiObjectiveReward):
                reward = float(
                    self._default_reward(
                        excess_units=excess,
                        shortfall_units=shortfall,
                        order_placed=order_placed,
                        demand_units=demand,
                    )
                )
            else:
                reward = float(
                    self._default_reward(
                        excess_units=excess,
                        shortfall_units=shortfall,
                        order_placed=order_placed,
                    )
                )
        else:
            reward = compute_inventory_reward(
                excess_units=excess,
                shortfall_units=shortfall,
                order_placed=order_placed,
                holding_cost=self.holding_cost,
                stockout_cost=self.stockout_cost,
                order_cost=self.order_cost,
            )

        self.total_holding += self.holding_cost * excess
        self.total_stockout += self.stockout_cost * shortfall
        self.total_orders += self.order_cost * float(order_placed)
        self.total_demand += demand
        self.total_fulfilled += min(demand, stock_before + arriving_order)
        if shortfall > 0:
            self.stockout_days += 1

        self.order_queue.append(order_qty)
        self.t += 1

        terminated = self.t >= self.episode_length
        truncated = False
        info = {
            "stock": float(self.stock),
            "demand": float(demand),
            "order_qty": float(order_qty),
            "excess": float(excess),
            "shortfall": float(shortfall),
            "service_level": float(shortfall == 0),
            "lead_time": float(self.current_lead_time),
        }
        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self) -> np.ndarray:
        """Build normalized observation vector.

        Layout:
            [0]               stock / max_stock
            [1..horizon]      forecast / max_demand
            [horizon+1]       uncertainty_width / max_demand
            [horizon+2]       store_open (1.0 or 0.0)
            [horizon+3]       lead_time / 7
            [horizon+4]       holding_cost / 10
            [horizon+5]       stockout_cost / 10
            [horizon+6]       sin(2π · dow / 7)
            [horizon+7]       cos(2π · dow / 7)
            [horizon+8]       bi-weekly promo phase flag

        Returns:
            Observation array clipped to [-1, 1].
        """
        forecast, uncertainty_width = self._get_forecast_with_uncertainty()
        max_demand = max(float(np.max(self.demand_series)), 1.0)
        day_of_week = self.t % 7
        store_open = self._get_store_open(self.t)

        obs = np.array(
            [
                self.stock / self.max_stock,
                *list(forecast / max_demand),
                uncertainty_width / max_demand,
                store_open,
                self.current_lead_time / 7.0,
                self.holding_cost / 10.0,
                self.stockout_cost / 10.0,
                np.sin(2 * np.pi * day_of_week / 7),
                np.cos(2 * np.pi * day_of_week / 7),
                float(self.t % 14 < 7),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def get_observation(self) -> np.ndarray:
        """Public observation getter used by API and registry helpers."""
        return self._obs()

    def _get_demand(self, timestep: int) -> float:
        """Read demand value at a specific timestep.

        Args:
            timestep: Current step index.

        Returns:
            Demand value.
        """
        idx = timestep % len(self.demand_series)
        return float(self.demand_series[idx])

    def _get_store_open(self, timestep: int) -> float:
        """Read store open status at a specific timestep.

        Args:
            timestep: Current step index.

        Returns:
            1.0 if open, 0.0 if closed.
        """
        if self.store_open_series is not None:
            idx = timestep % len(self.store_open_series)
            return float(self.store_open_series[idx])
        # Default: open every day except Sunday (day_of_week == 6)
        return 0.0 if (timestep % 7 == 6) else 1.0

    def _get_forecast_with_uncertainty(self) -> tuple[np.ndarray, float]:
        """Get forecast vector and scalar uncertainty width (p90 − p10).

        Returns:
            Tuple of (forecast_vector, mean_uncertainty_width).
        """
        if self.forecast_quantiles_fn is not None:
            p10, p50, p90 = self.forecast_quantiles_fn(self.t)
            p10 = np.asarray(p10, dtype=float)
            p50 = np.asarray(p50, dtype=float)
            p90 = np.asarray(p90, dtype=float)
            forecast = (
                p50[: self.horizon]
                if len(p50) >= self.horizon
                else np.pad(
                    p50,
                    (0, self.horizon - len(p50)),
                    constant_values=p50[-1] if len(p50) > 0 else 0,
                )
            )
            width = (
                float(np.mean(np.maximum(p90[: self.horizon] - p10[: self.horizon], 0.0)))
                if len(p10) > 0
                else 0.0
            )
            return forecast, width

        forecast = self._get_forecast()
        # Without quantile info, estimate uncertainty from demand variability
        width = float(np.std(self.demand_series) * 2.0)
        return forecast, width

    def _get_forecast(self) -> np.ndarray:
        """Read demand forecast for current timestep.

        Returns:
            Forecast vector with length equal to horizon.
        """
        if self.forecast_fn is not None:
            forecast = np.asarray(self.forecast_fn(self.t), dtype=float)
            if forecast.shape[0] >= self.horizon:
                return forecast[: self.horizon]
            if forecast.shape[0] == 0:
                return np.zeros(self.horizon, dtype=float)
            padding = np.repeat(forecast[-1], self.horizon - forecast.shape[0])
            return np.concatenate([forecast, padding])
        idxs = [(self.t + offset) % len(self.demand_series) for offset in range(self.horizon)]
        return self.demand_series[idxs]

    def service_level(self) -> float:
        """Compute episode service level as non-stockout ratio.

        Returns:
            Service level in range [0, 1].
        """
        if self.t == 0:
            return 1.0
        return float((self.t - self.stockout_days) / self.t)

    def fill_rate(self) -> float:
        """Compute episode fill rate as fulfilled-demand ratio.

        Returns:
            Fill rate in range [0, 1].
        """
        if self.total_demand <= 0.0:
            return 1.0
        return float(min(1.0, self.total_fulfilled / self.total_demand))

    def total_cost(self) -> float:
        """Compute total accumulated episode cost.

        Returns:
            Total cost.
        """
        return float(self.total_holding + self.total_stockout + self.total_orders)
