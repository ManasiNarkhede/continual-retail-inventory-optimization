"""Optuna hyperparameter optimization for TFT models."""

from __future__ import annotations

from contextlib import nullcontext

import lightning.pytorch as pl
import mlflow
import optuna
from lightning.pytorch.callbacks import Callback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def _ensure_optuna_study_dir() -> str:
    """Ensure local Optuna study directory exists at runtime.

    Returns:
        SQLite storage URL for the TFT study.
    """
    from pathlib import Path

    study_dir = Path("optuna_studies")
    study_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(study_dir / 'tft_study.db').as_posix()}"


def _trial_logger(study: optuna.Study, trial: optuna.Trial) -> None:
    """Log trial summary metrics to MLflow when run is active."""
    if mlflow.active_run() is None:
        return
    if trial.value is not None:
        mlflow.log_metric("trial_objective", float(trial.value), step=int(trial.number))
    mlflow.log_param(f"trial_{trial.number}_state", str(trial.state))


class OptunaPruningCallback(Callback):
    """Lightning callback to prune unpromising Optuna trials early."""

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss") -> None:
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        score = float(metric.detach().cpu().item()) if hasattr(metric, "detach") else float(metric)
        epoch = int(getattr(trainer, "current_epoch", 0))
        self.trial.report(score, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def tft_objective(trial: optuna.Trial, train_dataset, val_dataset) -> float:
    """Objective function for TFT Optuna search.

    Args:
        trial: Optuna trial object.
        train_dataset: Training TimeSeriesDataSet.
        val_dataset: Validation TimeSeriesDataSet.

    Returns:
        Validation proxy score (SMAPE-like objective).
    """
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    attention_head_size = trial.suggest_categorical("attention_head_size", [1, 2, 4])
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    hidden_cont = trial.suggest_categorical("hidden_continuous_size", [16, 32, 64, 128])

    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_cont,
        output_size=7,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=2,
    )

    train_loader = train_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    epoch_cap = int(trial.user_attrs.get("trial_epoch_cap", 20))
    max_epochs = trial.suggest_int("max_epochs", 20, max(epoch_cap, 20))
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
        callbacks=[OptunaPruningCallback(trial, monitor="val_loss")],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_loss = trainer.callback_metrics.get("val_loss")
    score = (
        float(val_loss.detach().cpu().item()) if hasattr(val_loss, "detach") else float(val_loss)
    )
    return score


def run_tft_hpo(
    train_dataset,
    val_dataset,
    n_trials: int = 20,
    trial_max_epochs: int = 20,
) -> optuna.Study:
    """Run TFT Optuna study and log best parameters.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        n_trials: Number of optimization trials.

    Returns:
        Completed Optuna study.
    """
    storage_uri = _ensure_optuna_study_dir()
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5),
        study_name="tft_hpo",
        storage=storage_uri,
        load_if_exists=True,
    )

    mlflow.set_experiment("retail_tft_hpo")
    active = mlflow.active_run()
    run_context = (
        nullcontext() if active is not None else mlflow.start_run(run_name="tft_hpo_master")
    )
    with run_context:
        mlflow.log_param("trial_max_epochs", int(max(trial_max_epochs, 1)))
        study.optimize(
            lambda trial: tft_objective_with_epochs(
                trial, train_dataset, val_dataset, int(max(trial_max_epochs, 1))
            ),
            n_trials=n_trials,
            callbacks=[_trial_logger],
        )
        mlflow.log_param("best_trial", study.best_trial.number)
        for key, value in study.best_trial.params.items():
            mlflow.log_param(f"best_{key}", value)
        mlflow.log_metric("best_objective", float(study.best_value))
    return study


def tft_objective_with_epochs(
    trial: optuna.Trial, train_dataset, val_dataset, trial_max_epochs: int
) -> float:
    """Proxy objective with externally configurable max-epoch ceiling."""
    upper = max(int(trial_max_epochs), 20)
    trial.set_user_attr("trial_epoch_cap", int(upper))
    return tft_objective(trial, train_dataset, val_dataset)
