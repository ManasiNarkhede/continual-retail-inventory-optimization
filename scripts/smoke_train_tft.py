"""Run a lightweight TFT smoke-training pass on real Rossmann data.

This script uses the existing data pipeline and model modules but limits data
volume and model/training size for fast runtime verification.

Usage:
    .venv39\Scripts\python scripts/smoke_train_tft.py
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from src.data.feature_engineer import FeatureEngineer
from src.data.loader import RossmannDataLoader
from src.data.preprocessor import RossmannPreprocessor
from src.models.tft_dataset import build_tft_dataset
from src.models.tft_model import TFTForecaster
from src.models.tft_trainer import TFTTrainer


def _build_smoke_config(base_config_path: str) -> str:
    """Create a temporary TFT config with smoke-friendly settings.

    Args:
        base_config_path: Path to base TFT config.

    Returns:
        Path to temporary YAML config file.
    """
    with Path(base_config_path).open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    tft_cfg = cfg["tft"]
    tft_cfg["max_epochs"] = 1
    tft_cfg["batch_size"] = 32
    tft_cfg["max_encoder_length"] = 30
    tft_cfg["max_prediction_length"] = 7
    tft_cfg["hidden_size"] = 16
    tft_cfg["hidden_continuous_size"] = 8
    tft_cfg["lstm_layers"] = 1
    tft_cfg["num_attention_heads"] = 1
    tft_cfg["dropout"] = 0.1
    tft_cfg["log_interval"] = -1
    tft_cfg["log_val_interval"] = -1

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.safe_dump(cfg, tmp)
        return tmp.name


def main(config_path: str, store_id: int, max_rows: int) -> None:
    """Execute a full smoke training pass.

    Args:
        config_path: Base TFT config path.
        store_id: Store id used for quick smoke subset.
        max_rows: Maximum rows after filtering/sorting.
    """
    smoke_config = _build_smoke_config(config_path)

    loader = RossmannDataLoader("data/raw")
    train_df, _ = loader.load()
    train_df = train_df[train_df["Store"] == store_id].sort_values("Date").tail(max_rows)

    preprocessor = RossmannPreprocessor()
    train_df = preprocessor.fit_transform(train_df)

    feature_engineer = FeatureEngineer()
    train_df = feature_engineer.transform(train_df)

    with Path(smoke_config).open("r", encoding="utf-8") as file:
        tft_cfg = yaml.safe_load(file)["tft"]

    required_cols = [
        "time_idx",
        "Store",
        tft_cfg["target"],
        *tft_cfg.get("static_categoricals", []),
        *tft_cfg.get("static_reals", []),
        *tft_cfg.get("time_varying_known_reals", []),
        *tft_cfg.get("time_varying_unknown_reals", []),
    ]
    existing_required = [column for column in required_cols if column in train_df.columns]
    train_df = train_df.dropna(subset=existing_required)
    if train_df.empty:
        raise RuntimeError("Smoke dataset is empty after dropping NA values.")

    if tft_cfg["target"] in train_df.columns:
        train_df[tft_cfg["target"]] = pd.to_numeric(train_df[tft_cfg["target"]], errors="coerce")
        train_df = train_df.dropna(subset=[tft_cfg["target"]])

    mlflow.set_experiment("retail_tft_smoke")
    with mlflow.start_run(run_name=f"smoke_store_{store_id}"):
        train_dataset = build_tft_dataset(train_df, config_path=smoke_config)
        val_dataset = build_tft_dataset(
            train_df,
            config_path=smoke_config,
            training=False,
            training_dataset=train_dataset,
        )

        forecaster = TFTForecaster(smoke_config, train_dataset)
        model = forecaster.build()

        trainer = TFTTrainer(config_path=smoke_config, mlflow_experiment="retail_tft_smoke")
        trainer.fit(model, train_dataset, val_dataset)

        val_loader = val_dataset.to_dataloader(train=False, batch_size=32, num_workers=0)
        predictions, _ = forecaster.predict(val_loader, mode="quantiles")
        print(f"SMOKE_OK predictions_shape={tuple(predictions.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tft_config.yaml")
    parser.add_argument("--store-id", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=400)
    args = parser.parse_args()
    main(config_path=args.config, store_id=args.store_id, max_rows=args.max_rows)
