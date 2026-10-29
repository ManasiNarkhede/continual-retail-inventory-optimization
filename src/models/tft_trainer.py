"""PyTorch Lightning trainer for TFT with MLflow integration."""

from __future__ import annotations

import logging
from pathlib import Path

import lightning.pytorch as pl
import mlflow
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from configs.validator import load_yaml_config, validate_config_schema

logging.getLogger("lightning.pytorch.trainer.connectors.callback_connector").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.trainer.connectors.data_connector").setLevel(logging.ERROR)


class TFTTrainer:
    """Manage the full TFT training lifecycle.

    Args:
        config_path: Path to tft_config.yaml.
        mlflow_experiment: MLflow experiment name.
    """

    def __init__(
        self,
        config_path: str = "configs/tft_config.yaml",
        mlflow_experiment: str = "retail_tft_forecasting",
    ) -> None:
        """Initialize trainer configuration.

        Args:
            config_path: Path to TFT configuration YAML.
            mlflow_experiment: MLflow experiment name for logging.
        """
        validate_config_schema(config_path, "configs/schemas/tft_schema.json")
        self.config = load_yaml_config(config_path).get("tft", {})
        self.mlflow_experiment = mlflow_experiment

    def fit(
        self,
        model: TemporalFusionTransformer,
        train_dataset: TimeSeriesDataSet,
        val_dataset: TimeSeriesDataSet,
    ) -> pl.Trainer:
        """Train TFT with early stopping and checkpointing.

        Args:
            model: Built TemporalFusionTransformer.
            train_dataset: Training TimeSeriesDataSet.
            val_dataset: Validation TimeSeriesDataSet.

        Returns:
            Fitted pl.Trainer instance.
        """
        cfg = self.config

        train_loader = train_dataset.to_dataloader(
            train=True, batch_size=cfg["batch_size"], num_workers=4
        )
        val_loader = val_dataset.to_dataloader(
            train=False,
            batch_size=cfg["batch_size"] * 2,
            num_workers=4,
        )

        tb_logger = TensorBoardLogger(
            save_dir="logs",
            name="tft_tensorboard",
        )

        checkpoint_dir = Path("checkpoints/tft")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=cfg["reduce_on_plateau_patience"] + 2),
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=str(checkpoint_dir),
                filename="tft-{epoch:02d}-{val_loss:.4f}",
                save_top_k=1,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        if hasattr(model, "hparams") and isinstance(getattr(model, "hparams"), dict):
            model.hparams["log_interval"] = -1
            model.hparams["log_val_interval"] = -1

        trainer = pl.Trainer(
            max_epochs=cfg["max_epochs"],
            gradient_clip_val=cfg["gradient_clip_val"],
            callbacks=callbacks,
            logger=tb_logger,
            accelerator="auto",
            devices=1,
            enable_model_summary=True,
            enable_progress_bar=False,
            log_every_n_steps=5,
        )

        if mlflow.active_run() is not None:
            mlflow.log_params(cfg)

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_path = trainer.checkpoint_callback.best_model_path
        if best_path and mlflow.active_run() is not None:
            mlflow.log_artifact(best_path)

        return trainer
