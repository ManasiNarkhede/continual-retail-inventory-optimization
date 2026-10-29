"""Unit tests for TFT HPO runtime setup and epoch cap wiring."""

from __future__ import annotations

from pathlib import Path

from src.hpo import tft_hpo


class _DummyTrial:
    def __init__(self) -> None:
        self.attrs = {}

    def set_user_attr(self, key, value):  # noqa: ANN001
        self.attrs[key] = value


def test_run_tft_hpo_creates_study_dir_at_runtime(tmp_path: Path, monkeypatch) -> None:
    """run_tft_hpo should create optuna_studies inside current working dir."""
    monkeypatch.chdir(tmp_path)
    created_dir = tmp_path / "optuna_studies"
    assert not created_dir.exists()

    class _BestTrial:
        number = 1
        params = {"hidden_size": 64}

    class _Study:
        best_trial = _BestTrial()
        best_value = 0.123

        def optimize(self, fn, n_trials, callbacks):  # noqa: ANN001
            _ = n_trials
            trial = object()
            fn(trial)
            for callback in callbacks:
                callback(self, type("_T", (), {"number": 0, "value": 0.123, "state": "COMPLETE"})())

    monkeypatch.setattr(tft_hpo, "tft_objective_with_epochs", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(tft_hpo.optuna, "create_study", lambda **kwargs: _Study())

    class _RunCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tft_hpo.mlflow, "set_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(tft_hpo.mlflow, "start_run", lambda *args, **kwargs: _RunCtx())
    monkeypatch.setattr(tft_hpo.mlflow, "log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(tft_hpo.mlflow, "log_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(tft_hpo.mlflow, "active_run", lambda: object())

    study = tft_hpo.run_tft_hpo(
        train_dataset=object(), val_dataset=object(), n_trials=1, trial_max_epochs=5
    )
    assert isinstance(study, _Study)
    assert created_dir.exists()
    assert created_dir.is_dir()


def test_tft_objective_with_epochs_sets_minimum_epoch_cap(monkeypatch) -> None:
    """Epoch cap user attr should be clamped to at least 20."""
    trial = _DummyTrial()

    def _fake_objective(passed_trial, train_dataset, val_dataset):  # noqa: ANN001, ARG001
        assert passed_trial is trial
        return 1.23

    monkeypatch.setattr(tft_hpo, "tft_objective", _fake_objective)

    result = tft_hpo.tft_objective_with_epochs(trial, object(), object(), trial_max_epochs=7)
    assert result == 1.23
    assert trial.attrs["trial_epoch_cap"] == 20


def test_tft_objective_with_epochs_respects_higher_epoch_cap(monkeypatch) -> None:
    """Epoch cap should preserve values above 20."""
    trial = _DummyTrial()

    def _fake_objective(passed_trial, train_dataset, val_dataset):  # noqa: ANN001, ARG001
        assert passed_trial is trial
        return 0.42

    monkeypatch.setattr(tft_hpo, "tft_objective", _fake_objective)

    result = tft_hpo.tft_objective_with_epochs(trial, object(), object(), trial_max_epochs=64)
    assert result == 0.42
    assert trial.attrs["trial_epoch_cap"] == 64


def test_run_tft_hpo_uses_active_mlflow_run_without_starting_nested(
    monkeypatch, tmp_path: Path
) -> None:
    """run_tft_hpo should not call mlflow.start_run when a run is already active."""
    monkeypatch.chdir(tmp_path)

    class _BestTrial:
        number = 1
        params = {"hidden_size": 64}

    class _Study:
        best_trial = _BestTrial()
        best_value = 0.25

        def optimize(self, fn, n_trials, callbacks):  # noqa: ANN001
            _ = n_trials
            trial = object()
            fn(trial)
            for callback in callbacks:
                callback(self, type("_T", (), {"number": 0, "value": 0.25, "state": "COMPLETE"})())

    monkeypatch.setattr(tft_hpo, "tft_objective_with_epochs", lambda *args, **kwargs: 0.25)
    monkeypatch.setattr(tft_hpo.optuna, "create_study", lambda **kwargs: _Study())

    calls = {"start_run": 0}

    def _start_run(*args, **kwargs):  # noqa: ANN001
        calls["start_run"] += 1
        raise AssertionError("start_run should not be called with active run")

    monkeypatch.setattr(tft_hpo.mlflow, "set_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(tft_hpo.mlflow, "start_run", _start_run)
    monkeypatch.setattr(tft_hpo.mlflow, "log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(tft_hpo.mlflow, "log_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(tft_hpo.mlflow, "active_run", lambda: object())

    study = tft_hpo.run_tft_hpo(
        train_dataset=object(), val_dataset=object(), n_trials=1, trial_max_epochs=10
    )
    assert isinstance(study, _Study)
    assert calls["start_run"] == 0
