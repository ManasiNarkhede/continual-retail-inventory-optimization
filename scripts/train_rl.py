"""RL agent training script.

Trains one PPO/SAC agent across several stores at once and saves it to
``checkpoints/rl/inventory_agent.zip``.

Usage:
    python scripts/train_rl.py --config configs/rl_config.yaml [--n-stores 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.validator import validate_configs
from src.data.feature_engineer import FeatureEngineer
from src.data.loader import RossmannDataLoader
from src.data.preprocessor import RossmannPreprocessor
from src.rl.agent import RLInventoryAgent
from src.rl.multi_store_env import build_vectorized_multi_store_env


def _prepare_runtime_config(
    config_path: str, total_timesteps: int | None
) -> tuple[str, Path | None]:
    """Write runtime RL config with optional lightweight timestep override."""
    if total_timesteps is None:
        return config_path, None

    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    cfg["agent"]["total_timesteps"] = int(total_timesteps)
    runtime_cfg = Path("data/processed/rl_runtime_config.work.yaml")
    runtime_cfg.parent.mkdir(parents=True, exist_ok=True)
    runtime_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(runtime_cfg), runtime_cfg


def select_training_stores(train_df: pd.DataFrame, sales_col: str, n_stores: int) -> list[int]:
    """Pick ``n_stores`` stores spread evenly from the smallest to the largest by mean sales."""
    means = train_df.groupby("Store")[sales_col].mean().sort_values()
    if n_stores >= len(means):
        return [int(s) for s in means.index]
    picks = np.linspace(0, len(means) - 1, n_stores).round().astype(int)
    return [int(s) for s in means.index[picks]]


def main(config_path: str, total_timesteps: int | None = None, n_stores: int = 8) -> None:
    """Run RL inventory agent training pipeline.

    Args:
        config_path: Path to RL YAML configuration.
        total_timesteps: Optional override of ``agent.total_timesteps``.
        n_stores: Number of stores trained on in parallel.
    """
    runtime_config_path, runtime_tmp = _prepare_runtime_config(config_path, total_timesteps)
    validate_configs(tft_config_path="configs/tft_config.yaml", rl_config_path=runtime_config_path)
    loader = RossmannDataLoader("data/raw")
    # Keep closed days: real store calendars (and the API simulation) contain them.
    train_df, _ = loader.load(drop_closed=False)
    train_df = RossmannPreprocessor().fit_transform(train_df)
    train_df = FeatureEngineer().transform(train_df)

    sales_col = "Sales" if "Sales" in train_df.columns else "sales"
    if sales_col not in train_df.columns:
        raise ValueError("Training dataframe must contain a sales column ('Sales' or 'sales').")

    store_ids = select_training_stores(train_df, sales_col, n_stores)
    env = build_vectorized_multi_store_env(train_df, store_ids, config_path=runtime_config_path)
    agent = RLInventoryAgent(config_path=runtime_config_path, env=env)
    agent.build()
    agent.train()
    print(f"Trained on stores {store_ids}; saved checkpoints/rl/inventory_agent.zip")

    if runtime_tmp is not None and runtime_tmp.exists():
        runtime_final = Path("data/processed/rl_runtime_config.yaml")
        runtime_final.parent.mkdir(parents=True, exist_ok=True)
        runtime_final.write_text(runtime_tmp.read_text(encoding="utf-8"), encoding="utf-8")
        runtime_tmp.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_config.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-stores", type=int, default=8)
    args = parser.parse_args()
    main(config_path=args.config, total_timesteps=args.total_timesteps, n_stores=args.n_stores)
