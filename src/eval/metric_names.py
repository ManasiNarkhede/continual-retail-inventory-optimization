"""Central metric naming constants for logs and artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum


class MetricNames(str, Enum):
    """Canonical metric names used across training and serving."""

    SMAPE = "smape"
    BASELINE_SMAPE = "baseline_smape"
    SMAPE_UPLIFT = "smape_uplift_vs_baseline"
    RMSE = "rmse"
    BASELINE_RMSE = "baseline_rmse"
    COVERAGE_80 = "coverage_80pct"
    SERVICE_LEVEL = "service_level"
    FILL_RATE = "fill_rate"
    TOTAL_COST = "total_cost"


METRIC_ALIASES: dict[MetricNames, tuple[str, ...]] = {
    MetricNames.SMAPE: ("smape", "val_smape"),
    MetricNames.BASELINE_SMAPE: ("baseline_smape",),
    MetricNames.SMAPE_UPLIFT: ("smape_uplift_vs_baseline",),
    MetricNames.RMSE: ("rmse", "val_rmse"),
    MetricNames.BASELINE_RMSE: ("baseline_rmse",),
    MetricNames.COVERAGE_80: ("coverage_80pct", "val_coverage_80pct"),
    MetricNames.SERVICE_LEVEL: ("service_level", "mean_service_level"),
    MetricNames.FILL_RATE: ("fill_rate", "mean_fill_rate"),
    MetricNames.TOTAL_COST: ("total_cost", "mean_total_cost"),
}


def read_metric(payload: Mapping[str, object], metric: MetricNames, default: float = 0.0) -> float:
    """Read a metric value from payload using canonical key aliases."""
    aliases = METRIC_ALIASES.get(metric, (metric.value,))
    for key in aliases:
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, (int, float, str)):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)
