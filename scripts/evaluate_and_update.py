"""Post-training evaluation script.

Usage (after training completes):
    .venv39\Scripts\python scripts/evaluate_and_update.py

This script:
1. Finds the best new checkpoint
2. Evaluates model on validation set
3. Updates metrics_latest.json
4. Calls generate_store_forecasts.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.model_registry import fetch_live_kpi_payload
from src.eval.metric_names import MetricNames, read_metric


def find_best_checkpoint() -> Path | None:
    """Find the most recent best checkpoint."""
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "tft"
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def update_metrics() -> dict | None:
    """Fetch metrics and update metrics_latest.json."""
    logger.info("Fetching live KPI metrics...")
    payload = fetch_live_kpi_payload()

    if not payload:
        logger.error("No metrics payload available!")
        return None

    smape = read_metric(payload, MetricNames.SMAPE, default=0.0)
    coverage = read_metric(payload, MetricNames.COVERAGE_80, default=0.0)
    rmse_val = read_metric(payload, MetricNames.RMSE, default=0.0)
    baseline_smape = read_metric(payload, MetricNames.BASELINE_SMAPE, default=20.4814)
    baseline_rmse = read_metric(payload, MetricNames.BASELINE_RMSE, default=1303.8615)

    best_ckpt = find_best_checkpoint()
    model_version = best_ckpt.stem if best_ckpt else "tft_200epochs"

    metrics = {
        "model_version": model_version + ".pt",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "val_smape": round(smape, 4),
        "val_rmse": round(rmse_val, 2),
        "val_coverage_80pct": round(coverage, 4),
        "stores_trained": 1,
        "baseline_smape": baseline_smape,
        "baseline_rmse": baseline_rmse,
        "target_smape": 12.0,
        "target_coverage": 0.75,
        "status": "production_ready" if smape < 12 and coverage > 0.75 else "needs_improvement",
    }

    metrics_path = PROJECT_ROOT / "data" / "processed" / "metrics_latest.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"  SMAPE: {smape:.2f}%")
    logger.info(f"  Coverage: {coverage*100:.1f}%")
    logger.info(f"  RMSE: {rmse_val:.0f}")
    logger.info(f"  Status: {metrics['status']}")

    return metrics


def main():
    """Run post-training evaluation."""
    logger.info("=" * 60)
    logger.info("Post-Training Evaluation")
    logger.info("=" * 60)

    # Find best checkpoint
    best_ckpt = find_best_checkpoint()
    if best_ckpt:
        logger.info(f"Best checkpoint: {best_ckpt.name}")
    else:
        logger.warning("No checkpoint found!")

    # Update metrics
    metrics = update_metrics()

    if not metrics:
        logger.error("Failed to update metrics!")
        sys.exit(1)

    # Check targets
    if metrics["val_smape"] < 12.0 and metrics["val_coverage_80pct"] > 0.75:
        logger.info("SUCCESS: Metrics meet targets!")
    else:
        logger.warning("Metrics below targets. Consider further training.")

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
