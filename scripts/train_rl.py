"""RL agent training script.

Usage:
    python scripts/train_rl.py --config configs/rl_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.validator import validate_configs
from src.data.feature_engineer import FeatureEngineer
from src.data.loader import RossmannDataLoader
from src.data.preprocessor import RossmannPreprocessor
from src.rl.agent import RLInventoryAgent
from src.rl.curriculum import CurriculumScheduler
from src.rl.inventory_env import RetailInventoryEnv
from src.rl.reward import TripleTierReward


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


def main(
    config_path: str, total_timesteps: int | None = None, max_levels: int | None = None
) -> None:
    """Run RL inventory agent training pipeline.

    Args:
        config_path: Path to RL YAML configuration.
    """
    runtime_config_path, runtime_tmp = _prepare_runtime_config(config_path, total_timesteps)
    validate_configs(tft_config_path="configs/tft_config.yaml", rl_config_path=runtime_config_path)
    loader = RossmannDataLoader("data/raw")
    train_df, _ = loader.load()

    preprocessor = RossmannPreprocessor()
    train_df = preprocessor.fit_transform(train_df)

    feature_engineer = FeatureEngineer()
    train_df = feature_engineer.transform(train_df)

    with open(runtime_config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    scheduler = CurriculumScheduler(train_df, n_levels=4)
    reward_fn = TripleTierReward(
        holding_cost=float(cfg["environment"]["holding_cost"]),
        order_cost=float(cfg["environment"]["order_cost"]),
        stockout_cost=float(cfg["environment"]["stockout_cost"]),
        stockout_severity=float(cfg["environment"].get("stockout_severity", 0.05)),
    )

    levels = (
        scheduler.n_levels
        if max_levels is None
        else min(scheduler.n_levels, max(int(max_levels), 1))
    )
    sales_col = "Sales" if "Sales" in train_df.columns else "sales"
    if sales_col not in train_df.columns:
        raise ValueError("Training dataframe must contain a sales column ('Sales' or 'sales').")

    for _ in range(levels):
        store_ids = scheduler.current_store_ids()
        if not store_ids:
            break

        store_id = int(store_ids[0])
        store_sales = (
            train_df[train_df["Store"] == store_id]
            .sort_values("Date")[sales_col]
            .values.astype(float)
        )
        env = RetailInventoryEnv(
            config_path=runtime_config_path, demand_series=store_sales, reward_fn=reward_fn
        )
        agent = RLInventoryAgent(config_path=runtime_config_path, env=env)
        agent.build()
        agent.train()
        scheduler.record_episode(service_level=0.93)
        if scheduler.is_complete():
            break

    if runtime_tmp is not None and runtime_tmp.exists():
        runtime_final = Path("data/processed/rl_runtime_config.yaml")
        runtime_final.parent.mkdir(parents=True, exist_ok=True)
        runtime_final.write_text(runtime_tmp.read_text(encoding="utf-8"), encoding="utf-8")
        runtime_tmp.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_config.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--max-levels", type=int, default=None)
    args = parser.parse_args()
    main(config_path=args.config, total_timesteps=args.total_timesteps, max_levels=args.max_levels)
