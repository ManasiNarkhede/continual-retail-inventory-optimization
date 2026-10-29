"""API metric alias handling tests."""

from __future__ import annotations

from src.eval.metric_names import MetricNames, read_metric


def test_val_aliases_resolve_to_canonical_keys() -> None:
    payload = {"val_smape": 12.0, "val_rmse": 5.5, "val_coverage_80pct": 0.81}
    assert read_metric(payload, MetricNames.SMAPE) == 12.0
    assert read_metric(payload, MetricNames.RMSE) == 5.5
    assert read_metric(payload, MetricNames.COVERAGE_80) == 0.81
