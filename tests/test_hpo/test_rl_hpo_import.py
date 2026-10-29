"""Tests for RL HPO import side effects and runtime setup."""

from __future__ import annotations

from pathlib import Path

from src.hpo import rl_hpo


def test_run_rl_hpo_creates_study_dir_at_runtime(tmp_path: Path, monkeypatch) -> None:
    """run_rl_hpo should create optuna_studies only when invoked."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "optuna_studies"
    assert not target.exists()

    class _BestTrial:
        number = 1
        params = {"learning_rate": 1e-4}

    class _Study:
        best_trial = _BestTrial()
        best_value = 0.1

        def optimize(self, fn, n_trials):  # noqa: ANN001
            _ = n_trials
            fn(object())

    monkeypatch.setattr(rl_hpo, "rl_objective", lambda *args, **kwargs: 0.1)
    monkeypatch.setattr(rl_hpo.optuna, "create_study", lambda **kwargs: _Study())
    monkeypatch.setattr(rl_hpo.mlflow, "set_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(rl_hpo.mlflow, "active_run", lambda: object())
    monkeypatch.setattr(rl_hpo.mlflow, "start_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(rl_hpo.mlflow, "log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(rl_hpo.mlflow, "log_metric", lambda *args, **kwargs: None)

    study = rl_hpo.run_rl_hpo(demand_series=object(), n_trials=1)
    assert isinstance(study, _Study)
    assert target.exists()
