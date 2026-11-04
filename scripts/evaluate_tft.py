"""Evaluate the TFT on each store's held-out final window and write metrics_latest.json.

The training dataset excludes the last ``max_prediction_length`` rows of every store, so
those rows are never seen in training. The model forecasts them in one shot and is compared
with two seasonal-naive baselines built only from data before the window:

- last week: sales on the same weekday in the most recent week before the window
- 4-week mean: mean sales on the same weekday over the four most recent weeks

Usage:
    python scripts/evaluate_tft.py [--checkpoint checkpoints/tft/tft_latest] [--no-write]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.validator import load_yaml_config
from src.data.feature_engineer import FeatureEngineer
from src.data.loader import RossmannDataLoader
from src.data.preprocessor import RossmannPreprocessor
from src.models.tft_dataset import build_tft_dataset
from src.models.tft_model import TFTForecaster


def _smape(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.mean(2 * np.abs(y - f) / (np.abs(y) + np.abs(f) + 1e-9)) * 100)


def _rmse(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - f) ** 2)))


def _weekday_history(history: pd.Series, date: pd.Timestamp, weeks: int) -> list[float]:
    """Sales on ``date``'s weekday in the ``weeks`` most recent weeks present in ``history``."""
    last = history.index.max()
    start = date - pd.Timedelta(days=7 * int(np.ceil((date - last).days / 7)))
    values = [history.get(start - pd.Timedelta(days=7 * k)) for k in range(weeks + 4)]
    return [float(v) for v in values if v is not None and np.isfinite(v)][:weeks]


def main(checkpoint: str, config_path: str, write: bool) -> dict:
    """Run the held-out evaluation and optionally persist it as metrics_latest.json."""
    cfg = load_yaml_config(config_path)["tft"]
    df, _ = RossmannDataLoader("data/raw").load()
    df = FeatureEngineer().transform(RossmannPreprocessor().fit_transform(df))
    required = ["time_idx", "Store", cfg["target"]] + [
        col
        for key in (
            "static_categoricals",
            "static_reals",
            "time_varying_known_reals",
            "time_varying_unknown_reals",
        )
        for col in cfg[key]
    ]
    df = df.dropna(subset=[c for c in required if c in df.columns])  # same as train_tft.py

    training = build_tft_dataset(df, config_path=config_path)
    holdout = build_tft_dataset(
        df, config_path=config_path, training=False, training_dataset=training, predict=True
    )
    forecaster = TFTForecaster.load_any(checkpoint, config_path, training)
    preds, index = forecaster.predict(
        holdout.to_dataloader(train=False, batch_size=128, num_workers=0), mode="quantiles"
    )
    preds = preds.detach().cpu().numpy() if hasattr(preds, "detach") else np.asarray(preds)
    quantiles = list(cfg["quantiles"])
    i10, i50, i90 = (quantiles.index(q) for q in (0.1, 0.5, 0.9))

    rows = df.set_index(["Store", "time_idx"])
    actual, p10, p50, p90, last_week, four_week = [], [], [], [], [], []
    for row, (_, item) in enumerate(index.iterrows()):
        store, first = int(item["Store"]), int(item["time_idx"])
        store_rows = rows.loc[store]
        history = store_rows.loc[store_rows.index < first].set_index("Date")[cfg["target"]]
        for step in range(preds.shape[1]):
            if first + step not in store_rows.index:
                continue
            target = store_rows.loc[first + step]
            past = _weekday_history(history, target["Date"], weeks=4)
            if not past:
                continue
            actual.append(float(target[cfg["target"]]))
            p10.append(preds[row, step, i10])
            p50.append(preds[row, step, i50])
            p90.append(preds[row, step, i90])
            last_week.append(past[0])
            four_week.append(float(np.mean(past)))
    y, lo, mid, hi, naive, naive4 = map(np.asarray, (actual, p10, p50, p90, last_week, four_week))

    result = {
        "model_version": Path(checkpoint).stem + ".pt",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": f"held-out final {preds.shape[1]} open days per store, unseen in training",
        "stores_trained": int(df["Store"].nunique()),
        "holdout_days_scored": int(len(y)),
        "val_smape": round(_smape(y, mid), 2),
        "val_rmse": round(_rmse(y, mid), 2),
        "val_coverage_80pct": round(float(np.mean((y >= lo) & (y <= hi))), 4),
        "baseline_smape": round(_smape(y, naive), 2),
        "baseline_rmse": round(_rmse(y, naive), 2),
        "baseline": "same weekday, last week before the window",
        "baseline_4wk_smape": round(_smape(y, naive4), 2),
        "baseline_4wk_rmse": round(_rmse(y, naive4), 2),
    }
    if write:
        path = Path("data/processed/metrics_latest.json")
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tft/tft_latest")
    parser.add_argument("--config", default="data/processed/tft_runtime_config.yaml")
    parser.add_argument("--no-write", action="store_true", help="print only")
    args = parser.parse_args()
    main(checkpoint=args.checkpoint, config_path=args.config, write=not args.no_write)
