"""Unit tests for RetailInventoryEnv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def _write_test_config(tmp_path: Path) -> str:
    """Write test RL config file.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        Config file path.
    """
    cfg = {
        "environment": {
            "initial_stock": 100,
            "max_stock": 500,
            "max_order": 200,
            "lead_time": 2,
            "holding_cost": 0.5,
            "stockout_cost": 5.0,
            "order_cost": 10.0,
            "episode_length": 30,
            "forecast_horizon": 7,
        },
        "agent": {
            "algorithm": "PPO",
            "policy": "MlpPolicy",
            "total_timesteps": 1000,
            "learning_rate": 3e-4,
            "n_steps": 64,
            "batch_size": 32,
            "n_epochs": 3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 0,
            "tensorboard_log": "./logs/tb",
        },
    }
    path = tmp_path / "rl_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_env_reset_returns_correct_obs_shape(tmp_path: Path) -> None:
    """Ensure reset returns expected observation size.

    Args:
        tmp_path: Temporary directory fixture.
    """
    from src.rl.inventory_env import RetailInventoryEnv

    config_path = _write_test_config(tmp_path)
    env = RetailInventoryEnv(config_path=config_path, demand_series=np.ones(365) * 50)
    observation, _ = env.reset()
    assert observation.shape == (16,)


def test_env_step_returns_five_tuple(tmp_path: Path) -> None:
    """Ensure step returns Gymnasium five-tuple.

    Args:
        tmp_path: Temporary directory fixture.
    """
    from src.rl.inventory_env import RetailInventoryEnv

    config_path = _write_test_config(tmp_path)
    env = RetailInventoryEnv(config_path=config_path, demand_series=np.ones(365) * 50)
    env.reset()
    result = env.step(np.array([0.0]))
    assert len(result) == 5


def test_observation_within_bounds(tmp_path: Path) -> None:
    """Ensure observation values remain in [-1, 1].

    Args:
        tmp_path: Temporary directory fixture.
    """
    from src.rl.inventory_env import RetailInventoryEnv

    config_path = _write_test_config(tmp_path)
    demand = np.random.randint(30, 80, size=365).astype(float)
    env = RetailInventoryEnv(config_path=config_path, demand_series=demand)
    observation, _ = env.reset()
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)


def test_reward_non_positive(tmp_path: Path) -> None:
    """Ensure reward remains non-positive under cost formulation.

    Args:
        tmp_path: Temporary directory fixture.
    """
    from src.rl.inventory_env import RetailInventoryEnv

    config_path = _write_test_config(tmp_path)
    env = RetailInventoryEnv(config_path=config_path, demand_series=np.ones(365) * 50)
    env.reset()
    _, reward, _, _, _ = env.step(np.array([0.5]))
    assert reward <= 0


def test_public_observation_getter_matches_shape(tmp_path: Path) -> None:
    """Public observation getter should return same expected observation shape."""
    from src.rl.inventory_env import RetailInventoryEnv

    config_path = _write_test_config(tmp_path)
    env = RetailInventoryEnv(config_path=config_path, demand_series=np.ones(365) * 50)
    env.reset()
    observation = env.get_observation()
    assert observation.shape == (16,)


def test_stochastic_lead_time_modes(tmp_path: Path) -> None:
    """Lead time should support stochastic modes without crashing step loop."""
    from src.rl.inventory_env import RetailInventoryEnv

    config_path = _write_test_config(tmp_path)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cfg["environment"]["lead_time_mode"] = "poisson"
    cfg["environment"]["lead_time_poisson_lambda"] = 2.0
    Path(config_path).write_text(yaml.safe_dump(cfg), encoding="utf-8")

    env = RetailInventoryEnv(config_path=config_path, demand_series=np.ones(365) * 50)
    env.reset()
    for _ in range(5):
        _obs, _reward, _done, _trunc, info = env.step(np.array([0.0]))
        assert info["lead_time"] >= 1.0


def test_reward_class_config_controls_default_reward(tmp_path: Path) -> None:
    """Environment should honor configured reward_class from rl_config."""
    from src.rl.inventory_env import RetailInventoryEnv
    from src.rl.reward import TripleTierReward

    config_path = _write_test_config(tmp_path)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cfg["environment"]["reward_class"] = "TripleTierReward"
    cfg["environment"]["use_multi_objective_reward"] = True
    Path(config_path).write_text(yaml.safe_dump(cfg), encoding="utf-8")

    env = RetailInventoryEnv(config_path=config_path, demand_series=np.ones(365) * 50)
    assert isinstance(env._default_reward, TripleTierReward)
