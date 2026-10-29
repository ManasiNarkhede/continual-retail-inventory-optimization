"""Tests for risk-aversion handling in RL simulation metrics."""

from __future__ import annotations

import numpy as np

from src.api import model_registry


def _patch_runtime_dependencies(
    monkeypatch, captured_inits: list[tuple[float | None, str, int]]
) -> None:
    class _Agent:
        def __init__(self) -> None:
            self.env = type(
                "_Wrap", (), {"env": type("_E", (), {"horizon": 7, "episode_length": 30})()}
            )()

        @staticmethod
        def predict(obs, deterministic=True):  # noqa: ANN001, ARG004
            _ = obs
            return np.array([0.0], dtype=float)

    class _Model:
        @staticmethod
        def predict_for_store(
            store_id, horizon, training_dataset, reference_df
        ):  # noqa: ANN001, ARG004
            pred = np.tile(np.array([[10.0, 12.0, 14.0]], dtype=float), (int(horizon), 1))
            return pred, []

    class _EvalEnv:
        episode_length = 10
        _counter = 0

        def __init__(
            self, config_path, demand_series, risk_aversion=None, reward_type=None
        ):  # noqa: ANN001
            _ = config_path, demand_series
            _EvalEnv._counter += 1
            self.env_id = _EvalEnv._counter
            captured_inits.append((risk_aversion, str(reward_type), int(self.env_id)))
            self._locked = False
            self._default_reward = object()
            self._step_count = 0
            self._locked = True

        def __setattr__(self, name, value):  # noqa: ANN001
            if getattr(self, "_locked", False) and name == "_default_reward":
                raise AssertionError("_default_reward should not be mutated after env construction")
            object.__setattr__(self, name, value)

        @staticmethod
        def reset():
            return np.array([0.0], dtype=float), {}

        def step(self, action):  # noqa: ANN001, ARG002
            self._step_count += 1
            done = self._step_count >= 1
            return np.array([0.0], dtype=float), 0.0, done, False, {}

        @staticmethod
        def service_level() -> float:
            return 0.95

        @staticmethod
        def fill_rate() -> float:
            return 0.97

        @staticmethod
        def total_cost() -> float:
            return 1234.0

    monkeypatch.setattr(model_registry, "get_rl_agent", lambda: _Agent())
    monkeypatch.setattr(model_registry, "get_tft_model", lambda: (_Model(), object(), object()))
    monkeypatch.setattr(
        model_registry, "get_tft_quantile_indices", lambda: {"p10": 0, "p50": 1, "p90": 2}
    )
    monkeypatch.setattr("src.rl.inventory_env.RetailInventoryEnv", _EvalEnv)


def test_run_rl_simulation_uses_constructor_risk_aversion(monkeypatch) -> None:
    """Simulation should pass risk aversion to environment constructor."""
    captured: list[tuple[float | None, str, int]] = []
    _patch_runtime_dependencies(monkeypatch, captured)

    result = model_registry.run_rl_simulation_metrics(
        n_days=3, store_id=1, risk_aversion=0.8, n_mc=3
    )

    assert len(captured) == 3
    assert all(item[0] == 0.8 for item in captured)
    assert all(item[1] == "multi_objective" for item in captured)
    assert result["risk_aversion"] == 0.8


def test_run_rl_simulation_defaults_to_auto_reward_type(monkeypatch) -> None:
    """Without risk aversion override, simulation should keep auto reward mode."""
    captured: list[tuple[float | None, str, int]] = []
    _patch_runtime_dependencies(monkeypatch, captured)

    result = model_registry.run_rl_simulation_metrics(
        n_days=3, store_id=1, risk_aversion=None, n_mc=2
    )

    assert len(captured) == 2
    assert all(item[0] is None for item in captured)
    assert all(item[1] == "auto" for item in captured)
    assert result["risk_aversion"] == 0.5


def test_run_rl_simulation_reuses_env_pool_across_calls(monkeypatch) -> None:
    """Repeated calls with same signature should reuse env instances from pool."""
    captured: list[tuple[float | None, str, int]] = []
    _patch_runtime_dependencies(monkeypatch, captured)

    _ = model_registry.run_rl_simulation_metrics(n_days=3, store_id=1, risk_aversion=0.8, n_mc=2)
    first_ids = [item[2] for item in captured]
    captured.clear()

    _ = model_registry.run_rl_simulation_metrics(n_days=3, store_id=1, risk_aversion=0.8, n_mc=2)
    second_ids = [item[2] for item in captured]

    assert first_ids == [1, 2]
    assert second_ids == []


def test_simulation_request_id_passthrough(monkeypatch) -> None:
    """Request id should be accepted and not break simulation execution."""
    captured: list[tuple[float | None, str, int]] = []
    _patch_runtime_dependencies(monkeypatch, captured)

    result = model_registry.run_rl_simulation_metrics(
        n_days=3,
        store_id=1,
        risk_aversion=0.5,
        n_mc=1,
        request_id="req-123",
    )

    assert result["risk_aversion"] == 0.5


def test_run_rl_simulation_clips_out_of_range_risk_aversion(monkeypatch) -> None:
    """Risk aversion should be clipped into [0, 1] before env construction."""
    captured: list[tuple[float | None, str, int]] = []
    _patch_runtime_dependencies(monkeypatch, captured)

    result = model_registry.run_rl_simulation_metrics(
        n_days=3, store_id=1, risk_aversion=1.8, n_mc=2
    )

    assert len(captured) == 2
    assert all(item[0] == 1.0 for item in captured)
    assert result["risk_aversion"] == 1.0
