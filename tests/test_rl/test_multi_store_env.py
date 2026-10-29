"""Unit tests for multi-store vector env helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.agent import RLInventoryAgent
from src.rl.multi_store_env import build_vectorized_multi_store_env


def _train_df() -> pd.DataFrame:
    stores = np.repeat([1, 2], 40)
    dates = pd.date_range("2015-01-01", periods=40).tolist() * 2
    sales = np.random.uniform(100.0, 300.0, size=len(stores))
    return pd.DataFrame({"Store": stores, "Date": pd.to_datetime(dates), "Sales": sales})


def _train_df_lowercase() -> pd.DataFrame:
    stores = np.repeat([1, 2], 40)
    dates = pd.date_range("2015-01-01", periods=40).tolist() * 2
    sales = np.random.uniform(100.0, 300.0, size=len(stores))
    return pd.DataFrame({"Store": stores, "Date": pd.to_datetime(dates), "sales": sales})


def test_build_multi_store_vec_env() -> None:
    """Helper should return vectorized env for valid stores."""
    env = build_vectorized_multi_store_env(_train_df(), store_ids=[1, 2])
    assert env is not None
    obs, _ = env.reset()
    assert obs.shape[0] == 2
    actions = np.zeros((2, 1), dtype=float)
    step = env.step(actions)
    assert step.observations.shape[0] == 2


def test_rl_agent_build_uses_all_vectorized_envs() -> None:
    """Agent should create an SB3 vector env spanning all stores."""
    env = build_vectorized_multi_store_env(_train_df(), store_ids=[1, 2])
    agent = RLInventoryAgent(config_path="configs/rl_config.yaml", env=env)
    agent.build()

    assert isinstance(agent.model.get_env(), DummyVecEnv)
    assert int(agent.model.get_env().num_envs) == 2


def test_build_multi_store_vec_env_accepts_lowercase_sales() -> None:
    """Vectorized env helper should accept normalized sales column."""
    env = build_vectorized_multi_store_env(_train_df_lowercase(), store_ids=[1, 2])
    obs, _ = env.reset()
    assert obs.shape[0] == 2
