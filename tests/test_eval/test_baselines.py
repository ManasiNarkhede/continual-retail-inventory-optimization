"""Unit tests for evaluation baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.baselines import BaseStockBaseline, EOQBaseline, NaiveSeasonalBaseline


def test_naive_baseline_predict_shape() -> None:
    """Naive predictions should preserve original shape."""
    baseline = NaiveSeasonalBaseline(seasonality=7)
    values = np.arange(20, dtype=float)
    preds = baseline.predict(values)
    assert preds.shape == values.shape
    assert np.isnan(preds[:7]).all()


def test_eoq_compute_positive_quantity() -> None:
    """EOQ quantity must be positive for positive annual demand."""
    baseline = EOQBaseline()
    assert baseline.compute_eoq(annual_demand=10000.0) > 0.0


def test_naive_evaluate_returns_result() -> None:
    """Evaluate should return a result object with named fields."""
    baseline = NaiveSeasonalBaseline(seasonality=7)
    df = pd.DataFrame({"Sales": np.linspace(100.0, 200.0, num=30)})
    result = baseline.evaluate(df)
    assert result.model_name.startswith("Naive")


def test_eoq_returns_inventory_metrics() -> None:
    """EOQ evaluation should provide service/fill/cost outputs."""
    baseline = EOQBaseline()
    df = pd.DataFrame({"Sales": np.linspace(80.0, 180.0, num=60)})
    result = baseline.evaluate(df)
    assert 0.0 <= result.service_level <= 1.0
    assert 0.0 <= result.fill_rate <= 1.0
    assert result.total_cost >= 0.0


def test_base_stock_baseline_evaluate() -> None:
    """Base-stock baseline should return finite inventory KPIs."""
    baseline = BaseStockBaseline()
    df = pd.DataFrame({"Sales": np.linspace(50.0, 150.0, num=50)})
    result = baseline.evaluate(df)
    assert result.model_name.startswith("Base-stock")
    assert 0.0 <= result.service_level <= 1.0
    assert 0.0 <= result.fill_rate <= 1.0
