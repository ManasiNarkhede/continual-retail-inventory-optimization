"""Tests for canonical metric name alias resolution."""

from __future__ import annotations

from src.eval.metric_names import MetricNames, read_metric


def test_read_metric_prefers_canonical_name() -> None:
    payload = {"smape": 12.3, "val_smape": 55.0}
    assert read_metric(payload, MetricNames.SMAPE) == 12.3


def test_read_metric_uses_alias_when_canonical_missing() -> None:
    payload = {"val_rmse": 4.4}
    assert read_metric(payload, MetricNames.RMSE) == 4.4
