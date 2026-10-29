"""Forecasting evaluation metrics implemented from first principles."""

from __future__ import annotations

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Symmetric Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth array.
        y_pred: Predicted array.

    Returns:
        SMAPE value in percent.
    """
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator > 0
    if not np.any(mask):
        return 0.0
    result = float(100 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]))
    if not np.isfinite(result):
        return float("nan")
    return result


def weighted_smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    under_weight: float = 1.5,
    over_weight: float = 1.0,
) -> float:
    """Compute asymmetric SMAPE with stronger under-forecast penalty.

    Args:
        y_true: Ground truth array.
        y_pred: Predicted array.
        under_weight: Multiplier when ``y_pred < y_true``.
        over_weight: Multiplier when ``y_pred >= y_true``.

    Returns:
        Weighted SMAPE value in percent.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator > 0
    if not np.any(mask):
        return 0.0

    true = y_true[mask]
    pred = y_pred[mask]
    weights = np.where(pred < true, float(under_weight), float(over_weight))
    errors = np.abs(true - pred) / denominator[mask]
    return float(100 * np.mean(weights * errors))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Ground truth array.
        y_pred: Predicted array.

    Returns:
        RMSE scalar.
    """
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    result = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if not np.isfinite(result):
        return float("nan")
    return result


def quantile_coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Compute empirical prediction interval coverage.

    Args:
        y_true: Ground truth values.
        y_lower: Lower quantile prediction.
        y_upper: Upper quantile prediction.

    Returns:
        Coverage fraction.
    """
    if y_true.size == 0 or y_lower.size == 0 or y_upper.size == 0:
        return float("nan")
    covered = (y_true >= y_lower) & (y_true <= y_upper)
    result = float(np.mean(covered))
    if not np.isfinite(result):
        return float("nan")
    return result


def mase(y_true: np.ndarray, y_pred: np.ndarray, m: int = 7) -> float:
    """Compute Mean Absolute Scaled Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        m: Seasonal period.

    Returns:
        MASE value.
    """
    model_mae = np.mean(np.abs(y_true - y_pred))
    if len(y_true) <= m:
        return float("inf")
    naive_diff = y_true[m:] - y_true[:-m]
    naive_mae = np.mean(np.abs(naive_diff))
    return float(model_mae / naive_mae) if naive_mae > 0 else float("inf")


def underestimation_rate(y_true: np.ndarray, y_pred: np.ndarray, atol: float = 1e-8) -> float:
    """Compute fraction of points where forecast underestimates demand.

    Args:
        y_true: Ground truth array.
        y_pred: Predicted array.
        atol: Small tolerance around equality.

    Returns:
        Fraction in ``[0, 1]`` where ``y_pred < y_true - atol``.
    """
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_pred < (y_true - atol)))
