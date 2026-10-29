"""Optuna hyperparameter optimization for RL inventory agents."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import mlflow
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO

from src.rl.inventory_env import RetailInventoryEnv


def _ensure_optuna_study_dir() -> str:
    """Ensure local Optuna study directory exists at runtime.

    Returns:
        SQLite storage URL for the RL study.
    """
    study_dir = Path("optuna_studies")
    study_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(study_dir / 'rl_study.db').as_posix()}"


def rl_objective(trial: optuna.Trial, demand_series: np.ndarray) -> float:
    """Objective function for RL hyperparameter tuning.

    Args:
        trial: Optuna trial object.
        demand_series: Store demand series.

    Returns:
        Negative average reward proxy (minimize).
    """
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.05, log=True)

    env = RetailInventoryEnv(config_path="configs/rl_config.yaml", demand_series=demand_series)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        verbose=0,
    )
    model.learn(total_timesteps=20_000, progress_bar=False)

    obs, _ = env.reset()
    rewards: list[float] = []
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            obs, _ = env.reset()

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    objective = -mean_reward
    trial.report(objective, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return objective


def run_rl_hpo(demand_series: np.ndarray, n_trials: int = 20) -> optuna.Study:
    """Run RL Optuna study and log best parameters.

    Args:
        demand_series: Demand series used to build training env.
        n_trials: Number of Optuna trials.

    Returns:
        Completed Optuna study.
    """
    storage_uri = _ensure_optuna_study_dir()

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5),
        study_name="rl_hpo",
        storage=storage_uri,
        load_if_exists=True,
    )

    mlflow.set_experiment("retail_rl_hpo")
    active = mlflow.active_run()
    run_context = (
        nullcontext() if active is not None else mlflow.start_run(run_name="rl_hpo_master")
    )
    with run_context:
        study.optimize(lambda trial: rl_objective(trial, demand_series), n_trials=n_trials)
        mlflow.log_param("best_trial", study.best_trial.number)
        for key, value in study.best_trial.params.items():
            mlflow.log_param(f"best_{key}", value)
        mlflow.log_metric("best_objective", float(study.best_value))
    return study
