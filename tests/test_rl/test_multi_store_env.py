"""Unit tests for multi-store vector env helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("stable_baselines3")
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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


@pytest.mark.parametrize("normalize_reward", [True, False])
def test_rl_agent_build_uses_all_vectorized_envs(tmp_path, normalize_reward: bool) -> None:
    """Agent spans all stores; reward normalization wraps the vec env without touching obs."""
    cfg = yaml.safe_load(open("configs/rl_config.yaml", encoding="utf-8"))
    cfg["agent"]["normalize_reward"] = normalize_reward
    config_path = tmp_path / "rl_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = build_vectorized_multi_store_env(_train_df(), store_ids=[1, 2])
    agent = RLInventoryAgent(config_path=str(config_path), env=env)
    agent.build()

    vec_env = agent.model.get_env()
    assert int(vec_env.num_envs) == 2
    if normalize_reward:
        assert isinstance(vec_env, VecNormalize)
        assert vec_env.norm_reward and not vec_env.norm_obs
        assert isinstance(vec_env.venv, DummyVecEnv)
    else:
        assert isinstance(vec_env, DummyVecEnv)


def test_build_multi_store_vec_env_accepts_lowercase_sales() -> None:
    """Vectorized env helper should accept normalized sales column."""
    env = build_vectorized_multi_store_env(_train_df_lowercase(), store_ids=[1, 2])
    obs, _ = env.reset()
    assert obs.shape[0] == 2
