"""Unit tests for forecast metric functions."""

from __future__ import annotations

import numpy as np

from src.eval.forecast_metrics import (
    mase,
    quantile_coverage,
    rmse,
    smape,
    underestimation_rate,
    weighted_smape,
)


def test_smape_zero_when_exact_match() -> None:
    """Ensure SMAPE is zero for perfect predictions."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert smape(y_true, y_pred) == 0.0


def test_rmse_positive_for_error() -> None:
    """Ensure RMSE is positive for imperfect predictions."""
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([2.0, 2.0])
    assert rmse(y_true, y_pred) > 0


def test_quantile_coverage_range() -> None:
    """Ensure quantile coverage is bounded in [0, 1]."""
    y_true = np.array([5.0, 6.0, 7.0])
    y_lower = np.array([4.0, 5.0, 6.0])
    y_upper = np.array([6.0, 7.0, 8.0])
    value = quantile_coverage(y_true, y_lower, y_upper)
    assert 0.0 <= value <= 1.0


def test_mase_finite_with_enough_samples() -> None:
    """Ensure MASE returns finite value for valid inputs."""
    y_true = np.arange(1, 20, dtype=float)
    y_pred = y_true + 0.5
    value = mase(y_true, y_pred, m=7)
    assert np.isfinite(value)


def test_weighted_smape_penalizes_underforecast_more() -> None:
    """Weighted SMAPE should be larger when under-forecasting dominates."""
    y_true = np.array([100.0, 120.0, 140.0])
    y_pred_under = np.array([80.0, 100.0, 120.0])
    y_pred_over = np.array([120.0, 140.0, 160.0])
    under_value = weighted_smape(y_true, y_pred_under, under_weight=1.5, over_weight=1.0)
    over_value = weighted_smape(y_true, y_pred_over, under_weight=1.5, over_weight=1.0)
    assert under_value > over_value


def test_underestimation_rate_bounds() -> None:
    """Underestimation rate should be in [0, 1]."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([9.0, 21.0, 29.0, 41.0])
    value = underestimation_rate(y_true, y_pred)
    assert 0.0 <= value <= 1.0
