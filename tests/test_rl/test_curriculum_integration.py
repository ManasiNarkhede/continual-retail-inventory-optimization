"""Integration-style tests for curriculum flow in RL agent training."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")

from src.rl.agent import RLInventoryAgent
from src.rl.inventory_env import RetailInventoryEnv


def _build_agent(enable_curriculum: bool, levels: int = 3) -> RLInventoryAgent:
    demand = np.ones(365, dtype=float) * 50.0
    env = RetailInventoryEnv(config_path="configs/rl_config.yaml", demand_series=demand)
    agent = RLInventoryAgent(config_path="configs/rl_config.yaml", env=env)
    agent.agent_cfg = dict(agent.agent_cfg)
    agent.agent_cfg["total_timesteps"] = 30
    agent.agent_cfg["enable_curriculum"] = bool(enable_curriculum)
    agent.agent_cfg["curriculum_levels"] = int(levels)
    return agent


def _patch_mlflow(monkeypatch) -> None:
    monkeypatch.setattr("src.rl.agent.mlflow.set_tracking_uri", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.rl.agent.mlflow.set_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.rl.agent.mlflow.start_run", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("src.rl.agent.mlflow.log_params", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.rl.agent.mlflow.log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.rl.agent.mlflow.log_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.rl.agent.mlflow.log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.rl.agent.mlflow.set_tag", lambda *args, **kwargs: None)


def test_train_uses_single_learn_when_curriculum_disabled(monkeypatch) -> None:
    """Disabled curriculum should run one learn call with full timesteps."""
    _patch_mlflow(monkeypatch)
    agent = _build_agent(enable_curriculum=False)

    calls: list[int] = []

    class _Model:
        def learn(self, total_timesteps, callback=None, progress_bar=False):  # noqa: ANN001
            _ = callback, progress_bar
            calls.append(int(total_timesteps))

        def save(self, *_args, **_kwargs):
            return None

        def get_env(self):
            class _Env:
                @staticmethod
                def get_attr(_name):
                    return [np.ones(365, dtype=float) * 50.0]

            return _Env()

    agent.model = _Model()
    monkeypatch.setattr("src.rl.agent.Path.exists", lambda self: False)

    agent.train()
    assert calls == [30]


def test_train_splits_timesteps_when_curriculum_enabled(monkeypatch) -> None:
    """Enabled curriculum should split timesteps across configured levels."""
    _patch_mlflow(monkeypatch)
    metric_calls: list[tuple[str, float, int]] = []
    monkeypatch.setattr(
        "src.rl.agent.mlflow.log_metric",
        lambda name, value, step=0: metric_calls.append((str(name), float(value), int(step))),
    )
    agent = _build_agent(enable_curriculum=True, levels=3)

    calls: list[int] = []

    class _Model:
        def learn(self, total_timesteps, callback=None, progress_bar=False):  # noqa: ANN001
            _ = callback, progress_bar
            calls.append(int(total_timesteps))

        def save(self, *_args, **_kwargs):
            return None

        def get_env(self):
            class _Env:
                @staticmethod
                def get_attr(_name):
                    return [np.ones(365, dtype=float) * 50.0]

            return _Env()

    class _Level:
        def __init__(self, level_id):
            self.level_id = level_id
            self.store_ids = [1]

    class _Scheduler:
        n_levels = 3

        def __init__(self, *_args, **_kwargs):
            self._idx = 0

        def current_level(self):
            return _Level(self._idx + 1)

        def record_episode(self, service_level):  # noqa: ANN001
            _ = service_level
            self._idx += 1
            return True

        def is_complete(self):
            return self._idx >= 3

    agent.model = _Model()
    monkeypatch.setattr(
        "src.rl.agent.Path.exists",
        lambda self: str(self).replace("\\", "/").endswith("data/raw/train.csv"),
    )
    monkeypatch.setattr(
        "pandas.read_csv",
        lambda *args, **kwargs: pd.DataFrame(
            {"Store": [1, 1, 2, 2], "Sales": [10.0, 11.0, 12.0, 13.0]}
        ),
    )
    monkeypatch.setattr("src.rl.agent.CurriculumScheduler", _Scheduler)

    agent.train()
    assert calls == [10, 10, 10]
    logged_keys = {name for (name, _value, _step) in metric_calls}
    assert "curriculum_level" in logged_keys
    assert "curriculum_service_level" in logged_keys
