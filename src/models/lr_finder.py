"""Learning-rate finder helper for TFT training.

The helper wraps Lightning's LR range test and returns a suggested
learning-rate value that can be logged and optionally reused.
"""

from __future__ import annotations

import inspect
import logging

import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def find_optimal_lr(
    model: TemporalFusionTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    min_lr: float = 1e-5,
    max_lr: float = 1.0,
    num_training_steps: int = 100,
) -> float:
    """Run LR finder and return a suggested learning rate.

    Args:
        model: Untrained TFT model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        min_lr: Sweep lower bound.
        max_lr: Sweep upper bound.
        num_training_steps: Number of sweep optimization steps.

    Returns:
        Suggested learning rate.
    """
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    tuner = pl.tuner.Tuner(trainer)
    lr_find_signature = inspect.signature(tuner.lr_find)
    kwargs = {
        "min_lr": min_lr,
        "max_lr": max_lr,
        "mode": "exponential",
        "early_stop_threshold": 4.0,
    }
    if "num_training_steps" in lr_find_signature.parameters:
        kwargs["num_training_steps"] = num_training_steps
    elif "num_training" in lr_find_signature.parameters:
        kwargs["num_training"] = num_training_steps

    result = tuner.lr_find(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        **kwargs,
    )

    suggestion = result.suggestion()
    if suggestion is None:
        suggestion = min_lr
    logger.info("LR finder suggestion: %.3e", float(suggestion))
    return float(suggestion)
