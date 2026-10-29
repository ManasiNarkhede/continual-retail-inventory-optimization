"""Diagnostic script to investigate low p10-p90 coverage.

Runs a minimal TFT training (2 epochs, store 1, 5000 rows) and computes:
- Per-quantile empirical coverage
- Quantile monotonicity violations
- Summary statistics for diagnosis
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import numpy as np
import yaml

from src.data.feature_engineer import FeatureEngineer
from src.data.loader import RossmannDataLoader
from src.data.preprocessor import RossmannPreprocessor
from src.eval.forecast_metrics import quantile_coverage
from src.models.tft_dataset import build_tft_dataset
from src.models.tft_model import TFTForecaster
from src.models.tft_trainer import TFTTrainer


def _build_diagnostic_config(base_config_path: str) -> str:
    with Path(base_config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tft_cfg = cfg["tft"]
    tft_cfg["max_epochs"] = 2
    tft_cfg["batch_size"] = 128
    tft_cfg["max_encoder_length"] = 90
    tft_cfg["max_prediction_length"] = 28
    tft_cfg["hidden_size"] = 256
    tft_cfg["hidden_continuous_size"] = 64
    tft_cfg["lstm_layers"] = 2
    tft_cfg["num_attention_heads"] = 4
    tft_cfg["dropout"] = 0.1
    tft_cfg["log_interval"] = -1
    tft_cfg["log_val_interval"] = -1

    out_path = Path("data/processed/diagnostic_tft_config.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(out_path)


def main():
    config_path = "configs/tft_config.yaml"
    diag_config = _build_diagnostic_config(config_path)

    print("Loading data...")
    loader = RossmannDataLoader("data/raw")
    train_df, _ = loader.load()
    train_df = train_df[train_df["Store"] == 1].sort_values("Date").tail(5000)

    print("Preprocessing...")
    preprocessor = RossmannPreprocessor()
    train_df = preprocessor.fit_transform(train_df)

    feature_engineer = FeatureEngineer()
    train_df = feature_engineer.transform(train_df)

    with Path(diag_config).open("r", encoding="utf-8") as f:
        tft_cfg = yaml.safe_load(f)["tft"]

    required_cols = [
        "time_idx",
        "Store",
        tft_cfg["target"],
        *tft_cfg.get("static_categoricals", []),
        *tft_cfg.get("static_reals", []),
        *tft_cfg.get("time_varying_known_reals", []),
        *tft_cfg.get("time_varying_unknown_reals", []),
    ]
    existing_required = [c for c in required_cols if c in train_df.columns]
    train_df = train_df.dropna(subset=existing_required)
    if train_df.empty:
        raise RuntimeError("Diagnostic dataset is empty after dropping NA values.")

    print("Building datasets...")
    train_dataset = build_tft_dataset(train_df, config_path=diag_config)
    val_dataset = build_tft_dataset(
        train_df,
        config_path=diag_config,
        training=False,
        training_dataset=train_dataset,
    )

    print("Building model...")
    forecaster = TFTForecaster(diag_config, train_dataset)
    model = forecaster.build()

    print("Training for 2 epochs...")
    trainer = TFTTrainer(config_path=diag_config)
    trainer.fit(model, train_dataset, val_dataset)

    print("Generating predictions...")
    val_loader = val_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)
    predictions, _ = forecaster.predict(val_loader, mode="quantiles")

    pred_np = (
        predictions.detach().cpu().numpy()
        if hasattr(predictions, "detach")
        else np.asarray(predictions)
    )
    if pred_np.ndim == 3:
        pred_np = pred_np.reshape(-1, pred_np.shape[-1])

    y_true_batches = []
    for _x_batch, target_tuple in val_loader:
        target = target_tuple[0]
        target_np = (
            target.detach().cpu().numpy() if hasattr(target, "detach") else np.asarray(target)
        )
        y_true_batches.append(target_np.reshape(-1).astype(float))
    y_true = np.concatenate(y_true_batches) if y_true_batches else np.array([], dtype=float)

    aligned = min(len(y_true), len(pred_np))
    y_true = y_true[:aligned]
    pred_np = pred_np[:aligned]

    quantiles = tft_cfg["quantiles"]
    n_quantiles = len(quantiles)

    # Per-quantile coverage: fraction where y_true <= y_pred_q
    per_quantile_coverage = {}
    for i, q in enumerate(quantiles):
        q_pred = pred_np[:, i]
        finite = np.isfinite(y_true) & np.isfinite(q_pred)
        if finite.sum() == 0:
            per_quantile_coverage[q] = float("nan")
        else:
            covered = y_true[finite] <= q_pred[finite]
            per_quantile_coverage[q] = float(np.mean(covered))

    # p10-p90 coverage
    p10_idx = quantiles.index(0.1)
    p90_idx = quantiles.index(0.9)
    coverage_p10_p90 = quantile_coverage(y_true, pred_np[:, p10_idx], pred_np[:, p90_idx])

    # Monotonicity check: p02 <= p10 <= p25 <= p50 <= p75 <= p90 <= p98
    monotonicity_violations = 0
    monotonicity_details = []
    for row in pred_np:
        for i in range(n_quantiles - 1):
            if row[i] > row[i + 1]:
                monotonicity_violations += 1
                monotonicity_details.append(
                    {
                        "row": int(len(monotonicity_details)),
                        "quantile_lower": quantiles[i],
                        "quantile_upper": quantiles[i + 1],
                        "value_lower": float(row[i]),
                        "value_upper": float(row[i + 1]),
                        "diff": float(row[i] - row[i + 1]),
                    }
                )
                break  # count one violation per row

    # Quantile spread statistics
    spreads = {}
    for q in quantiles:
        idx = quantiles.index(q)
        q_pred = pred_np[:, idx]
        finite = np.isfinite(q_pred)
        spreads[q] = {
            "mean": float(np.mean(q_pred[finite])),
            "std": float(np.std(q_pred[finite])),
            "min": float(np.min(q_pred[finite])),
            "max": float(np.max(q_pred[finite])),
            "median": float(np.median(q_pred[finite])),
        }

    # Target statistics
    finite_y = np.isfinite(y_true)
    target_stats = {
        "mean": float(np.mean(y_true[finite_y])),
        "std": float(np.std(y_true[finite_y])),
        "min": float(np.min(y_true[finite_y])),
        "max": float(np.max(y_true[finite_y])),
        "median": float(np.median(y_true[finite_y])),
    }

    # Prediction vs target alignment: median prediction vs target
    p50_idx = quantiles.index(0.5)
    p50_pred = pred_np[:, p50_idx]
    finite = np.isfinite(y_true) & np.isfinite(p50_pred)
    median_bias = float(np.mean(p50_pred[finite] - y_true[finite]))

    report = {
        "config": {
            "quantiles": quantiles,
            "under_penalty_factor": tft_cfg.get("under_penalty_factor", 1.5),
            "max_epochs": tft_cfg["max_epochs"],
            "hidden_size": tft_cfg["hidden_size"],
            "batch_size": tft_cfg["batch_size"],
        },
        "data": {
            "n_samples": int(len(y_true)),
            "n_finite": int(finite_y.sum()),
            "target_stats": target_stats,
        },
        "coverage": {
            "p10_p90": float(coverage_p10_p90),
            "per_quantile": per_quantile_coverage,
        },
        "monotonicity": {
            "total_rows": int(len(pred_np)),
            "violations": int(monotonicity_violations),
            "violation_rate": (
                float(monotonicity_violations / len(pred_np)) if len(pred_np) > 0 else 0.0
            ),
            "sample_violations": monotonicity_details[:10],
        },
        "spreads": spreads,
        "median_bias": median_bias,
    }

    out_path = Path(".sisyphus/evidence/t0b_coverage_diagnosis_raw.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Raw diagnosis saved to {out_path}")

    # Print summary
    print("\n=== COVERAGE DIAGNOSIS SUMMARY ===")
    print(f"Samples: {len(y_true)}")
    print(f"p10-p90 coverage: {coverage_p10_p90:.4%} (target ~80%)")
    print("\nPer-quantile empirical coverage (y_true <= y_pred_q):")
    for q, cov in per_quantile_coverage.items():
        print(f"  q={q:4.2f}: {cov:.4%} (target ~{q:.0%})")
    print(
        f"\nMonotonicity violations: {monotonicity_violations}/{len(pred_np)} "
        f"({monotonicity_violations/len(pred_np):.2%})"
    )
    print(f"Median bias (p50_pred - y_true): {median_bias:.4f}")
    print("===================================\n")


if __name__ == "__main__":
    main()
