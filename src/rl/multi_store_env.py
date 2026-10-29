"""Vectorized multi-store inventory environment for fast RL rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.rl.inventory_env import RetailInventoryEnv


@dataclass
class VectorStepResult:
    """Batch step result container."""

    observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list[dict[str, Any]]


class VectorizedMultiStoreEnv:
    """Manage multiple ``RetailInventoryEnv`` instances in one batched API.

    This provides vectorized reset/step interfaces so agents can process
    observations for many stores in one tensor operation.
    """

    def __init__(self, envs: list[RetailInventoryEnv]) -> None:
        if not envs:
            raise ValueError("At least one store environment is required.")
        self.envs = envs
        self.n_envs = len(envs)
        self.obs_dim = int(envs[0].observation_space.shape[0])

    def reset(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Reset all envs and return stacked observations."""
        observations: list[np.ndarray] = []
        infos: list[dict[str, Any]] = []
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        return np.stack(observations).astype(np.float32), infos

    def step(self, actions: np.ndarray) -> VectorStepResult:
        """Step all envs with action batch of shape ``(n_envs, action_dim)``."""
        if actions.ndim == 1:
            actions = actions.reshape(self.n_envs, -1)
        if actions.shape[0] != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} actions, got {actions.shape[0]}.")

        obs_list: list[np.ndarray] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        infos: list[dict[str, Any]] = []

        for idx, env in enumerate(self.envs):
            obs, reward, done, trunc, info = env.step(actions[idx])
            obs_list.append(obs)
            rewards.append(float(reward))
            terminated.append(bool(done))
            truncated.append(bool(trunc))
            infos.append(info)

        return VectorStepResult(
            observations=np.stack(obs_list).astype(np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            terminated=np.asarray(terminated, dtype=bool),
            truncated=np.asarray(truncated, dtype=bool),
            infos=infos,
        )


def build_vectorized_multi_store_env(
    train_df,
    store_ids: list[int],
    config_path: str = "configs/rl_config.yaml",
) -> VectorizedMultiStoreEnv:
    """Build vectorized environment from Rossmann store histories."""
    sales_col = "Sales" if "Sales" in train_df.columns else "sales"
    if sales_col not in train_df.columns:
        raise ValueError("train_df must contain a sales column ('Sales' or 'sales').")

    envs: list[RetailInventoryEnv] = []
    for sid in store_ids:
        demand = (
            train_df.loc[train_df["Store"] == sid]
            .sort_values("Date")[sales_col]
            .to_numpy(dtype=float)
        )
        if demand.size < 30:
            continue
        env = RetailInventoryEnv(config_path=config_path, demand_series=demand)
        env.store_id = int(sid)
        envs.append(env)

    if not envs:
        raise ValueError("No valid stores found for vectorized environment.")
    return VectorizedMultiStoreEnv(envs)
