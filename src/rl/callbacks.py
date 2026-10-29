"""Stable Baselines3 callbacks for MLflow metric logging."""

from __future__ import annotations

import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.rl.inventory_env import RetailInventoryEnv


class InventoryMetricsCallback(BaseCallback):
    """Log inventory-specific evaluation metrics during RL training.

    Args:
        eval_env: Environment used for periodic evaluation.
        eval_freq: Number of calls between evaluations.
        n_eval_episodes: Episodes to average per evaluation.
    """

    def __init__(
        self, eval_env: RetailInventoryEnv, eval_freq: int = 5000, n_eval_episodes: int = 5
    ) -> None:
        """Initialize callback for periodic inventory metric evaluation.

        Args:
            eval_env: Environment used for deterministic rollouts.
            eval_freq: Number of callback steps between evaluations.
            n_eval_episodes: Episodes to average in each evaluation cycle.
        """
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        """Run periodic evaluation and log mean metrics.

        Returns:
            Always True to continue training.
        """
        if self.n_calls % self.eval_freq != 0:
            return True

        service_levels: list[float] = []
        fill_rates: list[float] = []
        total_costs: list[float] = []
        for _ in range(self.n_eval_episodes):
            observation, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = self.eval_env.step(action)
                done = bool(terminated or truncated)
            service_levels.append(self.eval_env.service_level())
            fill_rates.append(self.eval_env.fill_rate())
            total_costs.append(self.eval_env.total_cost())

        if mlflow.active_run() is not None:
            mlflow.log_metric(
                "mean_service_level", float(np.mean(service_levels)), step=self.n_calls
            )
            mlflow.log_metric("mean_fill_rate", float(np.mean(fill_rates)), step=self.n_calls)
            mlflow.log_metric("mean_total_cost", float(np.mean(total_costs)), step=self.n_calls)
        return True
