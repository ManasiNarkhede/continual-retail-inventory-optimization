"""Unit tests for TFT/RL HPO objectives, pruning callback and study runners."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import optuna
import pytest
import pytorch_forecasting
import torch
from optuna.trial import FixedTrial

from src.hpo import rl_hpo, tft_hpo

TFT_PARAMS = {
    "hidden_size": 64,
    "learning_rate": 1e-3,
    "lstm_layers": 2,
    "attention_head_size": 4,
    "dropout": 0.1,
    "hidden_continuous_size": 32,
    "max_epochs": 25,
}
RL_PARAMS = {
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "n_steps": 512,
    "batch_size": 64,
    "ent_coef": 0.01,
}


class _MlflowRecorder:
    """Record mlflow calls made through a module's ``mlflow`` reference."""

    def __init__(self, module, monkeypatch, active: bool = False) -> None:  # noqa: ANN001
        self.params: dict = {}
        self.metrics: list[tuple] = []
        self.experiments: list[str] = []
        self.started: list[str] = []
        recorder = self

        class _Run:
            def __enter__(self):  # noqa: ANN204
                return self

            def __exit__(self, *exc):  # noqa: ANN002, ANN204
                return False

        def _start_run(run_name=None):  # noqa: ANN001, ANN202
            recorder.started.append(run_name)
            return _Run()

        monkeypatch.setattr(module.mlflow, "active_run", lambda: object() if active else None)
        monkeypatch.setattr(module.mlflow, "set_experiment", self.experiments.append)
        monkeypatch.setattr(module.mlflow, "start_run", _start_run)
        monkeypatch.setattr(module.mlflow, "log_param", self.params.__setitem__)
        monkeypatch.setattr(
            module.mlflow,
            "log_metric",
            lambda key, value, step=None: self.metrics.append((key, value, step)),
        )


def test_hpo_imports_do_not_create_optuna_dir(tmp_path: Path, monkeypatch) -> None:
    """Importing HPO modules must not create optuna_studies (it is created at run time)."""
    monkeypatch.chdir(tmp_path)
    for name in ("src.hpo.tft_hpo", "src.hpo.rl_hpo"):
        monkeypatch.delitem(sys.modules, name)
        importlib.import_module(name)
    assert not (tmp_path / "optuna_studies").exists()


# -----------------------------------------------------------------------
# tft_hpo
# -----------------------------------------------------------------------


class _Trial:
    def __init__(self, number: int, value, state: str = "COMPLETE") -> None:  # noqa: ANN001
        self.number = number
        self.value = value
        self.state = state


@pytest.mark.parametrize(
    ("active", "expected_metrics", "expected_params"),
    [
        (False, [], {}),
        (
            True,
            [("trial_objective", 0.5, 3)],
            {"trial_3_state": "COMPLETE", "trial_4_state": "PRUNED"},
        ),
    ],
    ids=["no-active-run", "active-run"],
)
def test_trial_logger(
    monkeypatch, active: bool, expected_metrics: list, expected_params: dict
) -> None:
    """With an active run, trials log objective (if any) at their step plus their state."""
    recorder = _MlflowRecorder(tft_hpo, monkeypatch, active=active)
    tft_hpo._trial_logger(None, _Trial(3, 0.5))
    tft_hpo._trial_logger(None, _Trial(4, None, state="PRUNED"))
    assert recorder.metrics == expected_metrics
    assert recorder.params == expected_params


class _ReportingTrial:
    def __init__(self, prune: bool) -> None:
        self.prune = prune
        self.reports: list[tuple] = []

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return self.prune


class _Trainer:
    def __init__(self, metrics: dict, epoch: int = 0) -> None:
        self.callback_metrics = metrics
        self.current_epoch = epoch


def test_pruning_callback_reports_tensor_and_float_metrics() -> None:
    """Monitored metric is reported per epoch; missing metric is ignored."""
    trial = _ReportingTrial(prune=False)
    callback = tft_hpo.OptunaPruningCallback(trial, monitor="val_loss")
    callback.on_validation_epoch_end(_Trainer({}), None)
    callback.on_validation_epoch_end(_Trainer({"val_loss": torch.tensor(0.5)}, epoch=2), None)
    callback.on_validation_epoch_end(_Trainer({"val_loss": 0.25}, epoch=3), None)
    assert trial.reports == [(0.5, 2), (0.25, 3)]


def test_pruning_callback_raises_when_trial_should_prune() -> None:
    """A prunable trial raises optuna.TrialPruned."""
    trial = _ReportingTrial(prune=True)
    callback = tft_hpo.OptunaPruningCallback(trial)
    with pytest.raises(optuna.TrialPruned):
        callback.on_validation_epoch_end(_Trainer({"val_loss": 1.0}, epoch=1), None)
    assert trial.reports == [(1.0, 1)]


@pytest.mark.parametrize(("requested", "cap"), [(7, 20), (64, 64)])
def test_tft_objective_with_epochs_sets_epoch_cap_of_at_least_20(
    monkeypatch, requested: int, cap: int
) -> None:
    """The epoch cap user attr is max(requested, 20) and the inner objective is returned."""
    trial = FixedTrial({})
    seen: list = []

    def _objective(passed_trial, train_dataset, val_dataset):  # noqa: ANN001, ANN202
        seen.append(passed_trial)
        return 0.42

    monkeypatch.setattr(tft_hpo, "tft_objective", _objective)

    assert tft_hpo.tft_objective_with_epochs(trial, object(), object(), requested) == 0.42
    assert seen == [trial]
    assert trial.user_attrs["trial_epoch_cap"] == cap


class _TftDataset:
    def __init__(self) -> None:
        self.loader_kwargs: dict = {}

    def to_dataloader(self, **kwargs):  # noqa: ANN201
        self.loader_kwargs = kwargs
        return f"loader-train={kwargs['train']}"


def test_tft_objective_builds_model_and_returns_val_loss(monkeypatch) -> None:
    """Sampled params configure the model/trainer and val_loss is returned."""
    captured: dict = {}

    def _from_dataset(dataset, **kwargs):  # noqa: ANN001
        captured["model_kwargs"] = kwargs
        return "model"

    class _FakeTrainer:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            captured["trainer_kwargs"] = kwargs
            self.callback_metrics: dict = {}

        def fit(self, model, train_dataloaders, val_dataloaders):  # noqa: ANN001
            captured["fit"] = (model, train_dataloaders, val_dataloaders)
            self.callback_metrics["val_loss"] = torch.tensor(0.375)

    monkeypatch.setattr(
        pytorch_forecasting.TemporalFusionTransformer, "from_dataset", _from_dataset
    )
    monkeypatch.setattr(tft_hpo.pl, "Trainer", _FakeTrainer)
    train_ds, val_ds = _TftDataset(), _TftDataset()

    score = tft_hpo.tft_objective_with_epochs(
        FixedTrial(TFT_PARAMS), train_ds, val_ds, trial_max_epochs=30
    )

    assert score == pytest.approx(0.375)
    kwargs = captured["model_kwargs"]
    assert kwargs["hidden_size"] == 64
    assert kwargs["lstm_layers"] == 2
    assert kwargs["attention_head_size"] == 4
    assert kwargs["hidden_continuous_size"] == 32
    assert kwargs["output_size"] == 7
    assert captured["trainer_kwargs"]["max_epochs"] == 25
    callbacks = captured["trainer_kwargs"]["callbacks"]
    assert len(callbacks) == 1 and isinstance(callbacks[0], tft_hpo.OptunaPruningCallback)
    assert captured["fit"] == ("model", "loader-train=True", "loader-train=False")
    assert train_ds.loader_kwargs == {"train": True, "batch_size": 64, "num_workers": 0}
    assert val_ds.loader_kwargs == {"train": False, "batch_size": 128, "num_workers": 0}


@pytest.mark.parametrize("active", [False, True], ids=["starts-run", "reuses-active-run"])
def test_run_tft_hpo_real_study_logs_best(tmp_path: Path, monkeypatch, active: bool) -> None:
    """A real 2-trial study creates its DB at run time and logs best params and metrics.

    A master run is started only when no MLflow run is already active.
    """
    monkeypatch.chdir(tmp_path)
    recorder = _MlflowRecorder(tft_hpo, monkeypatch, active=active)
    epochs_seen: list[int] = []

    def _objective(trial, train_dataset, val_dataset, trial_max_epochs):  # noqa: ANN001
        epochs_seen.append(trial_max_epochs)
        x = trial.suggest_float("x", 0.0, 1.0)
        return (x - 0.3) ** 2

    monkeypatch.setattr(tft_hpo, "tft_objective_with_epochs", _objective)

    study = tft_hpo.run_tft_hpo(object(), object(), n_trials=2, trial_max_epochs=0)

    assert (tmp_path / "optuna_studies" / "tft_study.db").exists()
    assert len(study.trials) == 2
    assert epochs_seen == [1, 1]
    assert recorder.experiments == ["retail_tft_hpo"]
    assert recorder.started == ([] if active else ["tft_hpo_master"])
    assert recorder.params["trial_max_epochs"] == 1
    assert recorder.params["best_trial"] == study.best_trial.number
    assert recorder.params["best_x"] == study.best_trial.params["x"]
    assert ("best_objective", study.best_value, None) in recorder.metrics


# -----------------------------------------------------------------------
# rl_hpo
# -----------------------------------------------------------------------


class _FakeEnv:
    """Three-step episode env whose reward is -(step index + 1)."""

    instances: list = []

    def __init__(self, config_path: str, demand_series) -> None:  # noqa: ANN001
        self.config_path = config_path
        self.demand_series = demand_series
        self.t = 0
        self.resets = 0
        _FakeEnv.instances.append(self)

    def reset(self):  # noqa: ANN201
        self.t = 0
        self.resets += 1
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):  # noqa: ANN001, ANN201
        self.t += 1
        return np.zeros(2, dtype=np.float32), -float(self.t), self.t >= 3, False, {}


class _FakePPO:
    instances: list = []

    def __init__(self, policy, env, **kwargs) -> None:  # noqa: ANN001, ANN003
        self.policy = policy
        self.env = env
        self.kwargs = kwargs
        self.learned: list = []
        _FakePPO.instances.append(self)

    def learn(self, total_timesteps, progress_bar):  # noqa: ANN001
        self.learned.append((total_timesteps, progress_bar))

    def predict(self, obs, deterministic):  # noqa: ANN001, ANN201
        assert deterministic is True
        return np.array([0.0]), None


@pytest.fixture
def fake_rl(monkeypatch):  # noqa: ANN201
    """Replace env and PPO in rl_hpo with fast fakes."""
    _FakeEnv.instances = []
    _FakePPO.instances = []
    monkeypatch.setattr(rl_hpo, "RetailInventoryEnv", _FakeEnv)
    monkeypatch.setattr(rl_hpo, "PPO", _FakePPO)


def test_rl_objective_returns_negative_mean_reward(fake_rl) -> None:  # noqa: ANN001
    """Objective is the negated mean reward of a 200-step deterministic rollout."""
    demand = np.ones(10)
    objective = rl_hpo.rl_objective(FixedTrial(RL_PARAMS), demand)

    # Rewards cycle -1, -2, -3 (reset every 3 steps): 66 full cycles + (-1, -2).
    expected = -((66 * -6.0) + (-1.0) + (-2.0)) / 200
    assert objective == pytest.approx(expected)
    env = _FakeEnv.instances[0]
    assert env.config_path == "configs/rl_config.yaml"
    assert env.demand_series is demand
    assert env.resets == 1 + 66
    ppo = _FakePPO.instances[0]
    assert ppo.policy == "MlpPolicy" and ppo.env is env
    assert ppo.kwargs == {**RL_PARAMS, "verbose": 0}
    assert ppo.learned == [(20_000, False)]


def test_rl_objective_raises_when_pruned(fake_rl) -> None:  # noqa: ANN001
    """A trial flagged for pruning raises optuna.TrialPruned after reporting."""

    class _PruneTrial(FixedTrial):
        def __init__(self) -> None:
            super().__init__(RL_PARAMS)
            self.reported: list = []

        def report(self, value, step):  # noqa: ANN001
            self.reported.append((value, step))

        def should_prune(self) -> bool:
            return True

    trial = _PruneTrial()
    with pytest.raises(optuna.TrialPruned):
        rl_hpo.rl_objective(trial, np.ones(5))
    assert trial.reported == [(pytest.approx(399 / 200), 0)]


@pytest.mark.parametrize("active", [False, True], ids=["starts-run", "reuses-active-run"])
def test_run_rl_hpo_real_study_logs_best(tmp_path: Path, monkeypatch, active: bool) -> None:
    """A real 2-trial RL study creates its DB at run time and logs best results."""
    monkeypatch.chdir(tmp_path)
    recorder = _MlflowRecorder(rl_hpo, monkeypatch, active=active)
    seen: list = []
    demand = np.arange(3)

    def _objective(trial, demand_series):  # noqa: ANN001
        seen.append(demand_series)
        return trial.suggest_float("gamma", 0.9, 0.99)

    monkeypatch.setattr(rl_hpo, "rl_objective", _objective)

    study = rl_hpo.run_rl_hpo(demand, n_trials=2)

    assert (tmp_path / "optuna_studies" / "rl_study.db").exists()
    assert len(study.trials) == 2 and all(item is demand for item in seen)
    assert recorder.experiments == ["retail_rl_hpo"]
    assert recorder.started == ([] if active else ["rl_hpo_master"])
    assert recorder.params["best_trial"] == study.best_trial.number
    assert recorder.params["best_gamma"] == study.best_trial.params["gamma"]
    assert recorder.metrics == [("best_objective", study.best_value, None)]
