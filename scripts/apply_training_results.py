"""Post-training automation script.

Usage (after user completes Colab training):
    python scripts/apply_training_results.py --checkpoint-path /path/to/best.ckpt
        --metrics-path /path/to/metrics_latest.json
        --forecast-path /path/to/tft_reference.parquet

Or simply:
    python scripts/apply_training_results.py

This will:
1. Copy new checkpoint to checkpoints/tft/
2. Update metrics_latest.json
3. Update tft_reference.parquet
4. Build the frontend
5. Run tests
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Apply training results from Colab")
    parser.add_argument("--checkpoint-path", type=str, help="Path to best .ckpt file")
    parser.add_argument("--metrics-path", type=str, help="Path to metrics_latest.json")
    parser.add_argument("--forecast-path", type=str, help="Path to tft_reference.parquet")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Applying Training Results")
    logger.info("=" * 60)

    # Step 1: Copy checkpoint
    if args.checkpoint_path:
        ckpt_src = Path(args.checkpoint_path)
        if ckpt_src.exists():
            ckpt_dst = PROJECT_ROOT / "checkpoints" / "tft" / ckpt_src.name
            shutil.copy2(ckpt_src, ckpt_dst)
            logger.info(f"Checkpoint copied: {ckpt_dst}")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_src}")

    # Step 2: Copy metrics
    if args.metrics_path:
        metrics_src = Path(args.metrics_path)
        if metrics_src.exists():
            metrics_dst = PROJECT_ROOT / "data" / "processed" / "metrics_latest.json"
            shutil.copy2(metrics_src, metrics_dst)
            logger.info(f"Metrics copied: {metrics_dst}")

            # Display metrics
            with open(metrics_dst) as f:
                metrics = json.load(f)
            logger.info(f"  SMAPE: {metrics.get('val_smape', 'N/A')}")
            logger.info(f"  Coverage: {metrics.get('val_coverage_80pct', 0) * 100:.1f}%")
            logger.info(f"  RMSE: {metrics.get('val_rmse', 'N/A')}")
        else:
            logger.warning(f"Metrics not found: {metrics_src}")

    # Step 3: Copy forecasts
    if args.forecast_path:
        forecast_src = Path(args.forecast_path)
        if forecast_src.exists():
            forecast_dst = PROJECT_ROOT / "data" / "processed" / "tft_reference.parquet"
            shutil.copy2(forecast_src, forecast_dst)
            logger.info(f"Forecasts copied: {forecast_dst}")
        else:
            logger.warning(f"Forecasts not found: {forecast_src}")

    # Step 4: Build frontend
    logger.info("Building frontend...")
    import subprocess

    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=PROJECT_ROOT / "frontend",
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("Frontend build: SUCCESS")
    else:
        logger.error("Frontend build: FAILED")
        logger.error(result.stderr)

    # Step 5: Run tests
    logger.info("Running tests...")
    result = subprocess.run(
        ["npm", "test"],
        cwd=PROJECT_ROOT / "frontend",
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("Tests: PASSED")
    else:
        logger.error("Tests: FAILED")
        logger.error(result.stderr[-500:])

    logger.info("=" * 60)
    logger.info("Done! Dashboard updated with new model results.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
