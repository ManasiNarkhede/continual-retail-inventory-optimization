"""Stable Baselines3 wrapper for inventory optimization agents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from configs.validator import load_yaml_config, validate_config_schema
from src.rl.callbacks import InventoryMetricsCallback
from src.rl.curriculum import CurriculumScheduler
from src.rl.inventory_env import RetailInventoryEnv
from src.rl.multi_store_env import VectorizedMultiStoreEnv

ALGORITHM_MAP = {"PPO": PPO, "SAC": SAC}
logger = logging.getLogger(__name__)


class RLInventoryAgent:
    """Inventory optimization agent using PPO or SAC.

    Args:
        config_path: Path to rl_config.yaml.
        env: Environment instance.
    """

    def __init__(self, config_path: str, env: RetailInventoryEnv | VectorizedMultiStoreEnv) -> None:
        """Load configuration and wrap environment with monitor.

        Args:
            config_path: Path to RL configuration YAML.
            env: Initialized retail inventory environment.
        """
        self.config_path = config_path
        validate_config_schema(config_path, "configs/schemas/rl_schema.json")
        self.config = load_yaml_config(config_path)
        self.agent_cfg = self.config["agent"]
        self.mlflow_cfg = self.config.get("mlflow", {})
        self.env = Monitor(env) if isinstance(env, RetailInventoryEnv) else env
        self.model = None

    def _env_demand_series(self) -> np.ndarray:
        """Retrieve demand series from VecEnv using public get_attr API."""
        if self.model is None:
            raise RuntimeError("Model is unavailable.")
        env = self.model.get_env()
        try:
            values = env.get_attr("demand_series")
        except Exception:
            return self._primary_training_env().demand_series
        if not values:
            return self._primary_training_env().demand_series
        first = values[0]
        return np.asarray(first, dtype=float)

    def _flatten_params(self, payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested dictionaries for MLflow parameter logging.

        Args:
            payload: Nested dictionary payload.
            prefix: Key prefix for recursive flattening.

        Returns:
            Flattened dictionary with scalar-compatible values.
        """
        flattened: dict[str, Any] = {}
        for key, value in payload.items():
            compound_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flattened.update(self._flatten_params(value, prefix=compound_key))
            else:
                flattened[compound_key] = value
        return flattened

    def build(self) -> None:
        """Instantiate the configured SB3 model.

        Raises:
            ValueError: If the configured algorithm is unsupported.
        """
        algorithm_name = self.agent_cfg["algorithm"]
        algorithm_cls = ALGORITHM_MAP.get(algorithm_name)
        if algorithm_cls is None:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        if isinstance(self.env, VectorizedMultiStoreEnv):

            def _make_env(single_env: RetailInventoryEnv):
                return lambda: Monitor(single_env)

            vec_env = DummyVecEnv([_make_env(single_env) for single_env in self.env.envs])
        else:
            vec_env = DummyVecEnv([lambda: self.env])
        if self.agent_cfg.get("normalize_reward", False):
            # Inventory costs reach ~1e6 per episode; unscaled, the value loss dominates the
            # clipped gradient and the policy barely learns. Observations stay raw, so
            # inference (API, evaluation) needs no saved normalization statistics.
            vec_env = VecNormalize(
                vec_env, norm_obs=False, norm_reward=True, gamma=self.agent_cfg["gamma"]
            )
        model_kwargs = {
            "policy": self.agent_cfg["policy"],
            "env": vec_env,
            "learning_rate": self.agent_cfg["learning_rate"],
            "gamma": self.agent_cfg["gamma"],
            "verbose": self.agent_cfg["verbose"],
            "tensorboard_log": self.agent_cfg["tensorboard_log"],
        }
        if algorithm_name == "PPO":
            model_kwargs.update(
                {
                    "n_steps": self.agent_cfg["n_steps"],
                    "batch_size": self.agent_cfg["batch_size"],
                    "n_epochs": self.agent_cfg["n_epochs"],
                    "gae_lambda": self.agent_cfg["gae_lambda"],
                    "clip_range": self.agent_cfg["clip_range"],
                    "ent_coef": self.agent_cfg["ent_coef"],
                    "vf_coef": self.agent_cfg["vf_coef"],
                    "max_grad_norm": self.agent_cfg["max_grad_norm"],
                }
            )
        else:
            sac_cfg = self.agent_cfg.get("sac", {})
            model_kwargs.update(
                {
                    "batch_size": self.agent_cfg["batch_size"],
                    "buffer_size": sac_cfg.get("buffer_size", 1_000_000),
                    "learning_starts": sac_cfg.get("learning_starts", 10_000),
                    "tau": sac_cfg.get("tau", 0.005),
                    "ent_coef": sac_cfg.get("ent_coef", "auto"),
                }
            )

        self.model = algorithm_cls(**model_kwargs)

    def _primary_training_env(self) -> RetailInventoryEnv:
        """Return a representative single-store env for eval callbacks."""
        if isinstance(self.env, VectorizedMultiStoreEnv):
            return self.env.envs[0]
        if isinstance(self.env, Monitor):
            return self.env.env
        return self.env

    def train(self) -> None:
        """Train agent and log metrics to MLflow.

        Raises:
            RuntimeError: If build() was not called.
        """
        if self.model is None:
            raise RuntimeError("Call build() before train().")

        experiment_name = self.mlflow_cfg.get("experiment_name", "retail_rl_inventory")
        tracking_uri = self.mlflow_cfg.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        try:
            mlflow.set_experiment(experiment_name)
        except Exception:
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment(experiment_name)

        demand_series = self._env_demand_series()
        eval_env = RetailInventoryEnv(config_path=self.config_path, demand_series=demand_series)
        callback = InventoryMetricsCallback(eval_env=eval_env)

        store_df = None
        for candidate in ("data/processed/tft_reference.parquet", "data/raw/train.csv"):
            path = Path(candidate)
            if path.exists():
                try:
                    import pandas as pd

                    store_df = (
                        pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                    )
                    if "Store" in store_df.columns and (
                        "Sales" in store_df.columns or "sales" in store_df.columns
                    ):
                        break
                except Exception:
                    store_df = None
        enable_curriculum = bool(self.agent_cfg.get("enable_curriculum", False))
        curriculum_levels = int(self.agent_cfg.get("curriculum_levels", 3))
        if enable_curriculum and store_df is not None:
            scheduler = CurriculumScheduler(store_df, n_levels=max(curriculum_levels, 1))
        else:
            scheduler = None
            if enable_curriculum:
                logger.warning(
                    "Curriculum enabled but no store dataframe found; "
                    "falling back to standard RL training."
                )

        with mlflow.start_run(run_name=f"rl_{self.agent_cfg['algorithm'].lower()}"):
            mlflow.log_params(self._flatten_params(self.agent_cfg))
            for key, value in self.mlflow_cfg.get("run_tags", {}).items():
                mlflow.set_tag(str(key), str(value))
            total_timesteps = int(self.agent_cfg["total_timesteps"])
            if scheduler is None:
                self.model.learn(
                    total_timesteps=total_timesteps, callback=callback, progress_bar=False
                )
            else:
                per_level = max(total_timesteps // max(scheduler.n_levels, 1), 1)
                for level in range(scheduler.n_levels):
                    active_level = scheduler.current_level()
                    mlflow.log_metric("curriculum_level", float(active_level.level_id), step=level)
                    mlflow.log_metric(
                        "curriculum_store_count", float(len(active_level.store_ids)), step=level
                    )
                    self.model.learn(
                        total_timesteps=per_level, callback=callback, progress_bar=False
                    )
                    service_level = float(eval_env.service_level())
                    advanced = scheduler.record_episode(service_level=service_level)
                    mlflow.log_metric("curriculum_service_level", service_level, step=level)
                    mlflow.log_metric("curriculum_advanced", float(advanced), step=level)
                    if scheduler.is_complete():
                        break

            checkpoint_path = Path("checkpoints/rl/inventory_agent")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(checkpoint_path))
            artifact_path = f"{checkpoint_path}.zip"
            mlflow.log_artifact(artifact_path)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict an order action from observation.

        Args:
            obs: Environment observation.
            deterministic: Whether to use deterministic policy.

        Returns:
            Action vector in [-1, 1].

        Raises:
            RuntimeError: If model is unavailable.
        """
        if self.model is None:
            raise RuntimeError("Model not built or loaded.")
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action
