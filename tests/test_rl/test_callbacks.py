"""Unit tests for InventoryMetricsCallback periodic evaluation and MLflow logging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.rl import callbacks as callbacks_module
from src.rl.callbacks import InventoryMetricsCallback
from src.rl.inventory_env import RetailInventoryEnv


def _env(tmp_path: Path) -> RetailInventoryEnv:
    cfg = {
        "environment": {
            "initial_stock": 100,
            "max_stock": 500,
            "max_order": 200,
            "lead_time": 2,
            "holding_cost": 0.5,
            "stockout_cost": 5.0,
            "order_cost": 10.0,
            "episode_length": 4,
            "forecast_horizon": 7,
        },
        "agent": {"algorithm": "PPO", "total_timesteps": 10},
    }
    path = tmp_path / "rl_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return RetailInventoryEnv(config_path=str(path), demand_series=np.full(10, 50.0))


class _NoOrderModel:
    """Policy that always outputs action -1 (order quantity 0)."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, observation, deterministic):  # noqa: ANN001, ANN201
        assert deterministic is True
        self.calls += 1
        return np.array([-1.0], dtype=np.float32), None


def _callback(tmp_path: Path, n_calls: int, eval_freq: int = 5, episodes: int = 2):  # noqa: ANN202
    callback = InventoryMetricsCallback(
        _env(tmp_path), eval_freq=eval_freq, n_eval_episodes=episodes
    )
    callback.model = _NoOrderModel()
    callback.n_calls = n_calls
    return callback


def _record_mlflow(monkeypatch, active: bool) -> list:  # noqa: ANN001
    logged: list = []
    monkeypatch.setattr(callbacks_module.mlflow, "active_run", lambda: object() if active else None)
    monkeypatch.setattr(
        callbacks_module.mlflow,
        "log_metric",
        lambda key, value, step=None: logged.append((key, value, step)),
    )
    return logged


def test_on_step_skips_between_evaluations(tmp_path: Path, monkeypatch) -> None:
    """Off-frequency steps neither roll out episodes nor log metrics."""
    logged = _record_mlflow(monkeypatch, active=True)
    callback = _callback(tmp_path, n_calls=3)
    assert callback._on_step() is True
    assert callback.model.calls == 0
    assert logged == []


@pytest.mark.parametrize("active", [True, False], ids=["active-run", "no-active-run"])
def test_on_step_evaluates_and_logs_mean_metrics(tmp_path: Path, monkeypatch, active: bool) -> None:
    """On-frequency steps roll out episodes; mean metrics are logged only with an active run.

    No-order episode (demand 50, stock 100): stock 100 -> 50 -> 0, then two stockout days of
    50 units each, so service = 2/4, fill = 100/200, cost = 0.5*50 + 5*(50+50) = 525.
    """
    logged = _record_mlflow(monkeypatch, active=active)
    callback = _callback(tmp_path, n_calls=10)

    assert callback._on_step() is True

    assert callback.model.calls == 2 * 4
    assert callback.eval_env.t == 4
    expected = [
        ("mean_service_level", pytest.approx(0.5), 10),
        ("mean_fill_rate", pytest.approx(0.5), 10),
        ("mean_total_cost", pytest.approx(525.0), 10),
    ]
    assert logged == (expected if active else [])
