"""Calibrate asymmetric forecast loss against inventory risk proxies.

This script evaluates weighted SMAPE and underestimation rate across
quantile columns and under-forecast penalty factors. It writes a thesis-
ready calibration table and recommends a practical penalty factor.

Usage:
    python -m src.eval.calibrate_loss --checkpoint checkpoints/tft/tft_latest.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.eval.forecast_metrics import underestimation_rate, weighted_smape
from src.models.tft_dataset import build_tft_dataset
from src.models.tft_model import TFTForecaster


def _collect_targets(loader) -> np.ndarray:
    batches: list[np.ndarray] = []
    for _x_batch, target_tuple in loader:
        target = target_tuple[0]
        target_np = (
            target.detach().cpu().numpy() if hasattr(target, "detach") else np.asarray(target)
        )
        batches.append(target_np.reshape(-1).astype(float))
    return np.concatenate(batches) if batches else np.array([], dtype=float)


def _parse_under_weights(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    return sorted(set(values)) if values else [1.0, 1.25, 1.5, 1.75, 2.0]


def _recommend_under_weight(
    summary_df: pd.DataFrame, preferred: float = 1.5, tolerance: float = 0.02
) -> float:
    if summary_df.empty:
        return preferred

    best_value = float(summary_df["best_inventory_cost_proxy"].min())
    preferred_row = summary_df[np.isclose(summary_df["under_weight"], preferred)]
    if not preferred_row.empty:
        preferred_value = float(preferred_row.iloc[0]["best_inventory_cost_proxy"])
        if preferred_value <= (1.0 + tolerance) * best_value:
            return preferred

    best_row = summary_df.sort_values("best_inventory_cost_proxy", ascending=True).iloc[0]
    return float(best_row["under_weight"])


def run_calibration(
    checkpoint_path: str,
    config_path: str,
    reference_path: str,
    output_dir: str,
    under_weights: list[float],
) -> tuple[Path, Path, float]:
    """Run calibration sweep and write CSV/markdown outputs."""
    runtime_cfg = Path(config_path)
    if not runtime_cfg.exists():
        runtime_cfg = Path("configs/tft_config.yaml")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    ref_df = pd.read_parquet(reference_path)
    train_ds = build_tft_dataset(ref_df, config_path=str(runtime_cfg))
    val_ds = build_tft_dataset(
        ref_df, config_path=str(runtime_cfg), training=False, training_dataset=train_ds
    )
    val_loader = val_ds.to_dataloader(train=False, batch_size=128, num_workers=0)

    forecaster = TFTForecaster.load_any(
        path_or_base=checkpoint_path, config_path=str(runtime_cfg), dataset=train_ds
    )
    predictions, _ = forecaster.predict(val_loader, mode="quantiles")

    pred_np = (
        predictions.detach().cpu().numpy()
        if hasattr(predictions, "detach")
        else np.asarray(predictions)
    )
    if pred_np.ndim == 3:
        pred_np = pred_np.reshape(-1, pred_np.shape[-1])

    y_true = _collect_targets(val_loader)
    aligned = min(len(y_true), len(pred_np))
    if aligned == 0:
        raise RuntimeError("No aligned rows available for calibration.")

    y_true = y_true[:aligned]
    pred_np = pred_np[:aligned]

    rows: list[dict[str, float | int]] = []
    quantiles = forecaster.config.get("quantiles", list(range(pred_np.shape[1])))
    for quantile_idx in range(pred_np.shape[1]):
        quantile_value = (
            float(quantiles[quantile_idx]) if quantile_idx < len(quantiles) else float(quantile_idx)
        )
        y_pred_q = pred_np[:, quantile_idx].astype(float)

        finite = np.isfinite(y_true) & np.isfinite(y_pred_q)
        yt = y_true[finite]
        yp = y_pred_q[finite]
        if yt.size == 0:
            continue

        under_rate = float(underestimation_rate(yt, yp))
        for under_weight in under_weights:
            wsmape = float(weighted_smape(yt, yp, under_weight=under_weight, over_weight=1.0))
            inventory_cost_proxy = float(wsmape * (1.0 + under_rate))
            rows.append(
                {
                    "quantile_idx": int(quantile_idx),
                    "quantile": quantile_value,
                    "under_weight": float(under_weight),
                    "weighted_smape": wsmape,
                    "underestimation_rate": under_rate,
                    "inventory_cost_proxy": inventory_cost_proxy,
                }
            )

    if not rows:
        raise RuntimeError("Calibration produced no finite rows.")

    detail_df = pd.DataFrame(rows).sort_values(
        ["under_weight", "inventory_cost_proxy", "quantile_idx"]
    )

    summary_rows = []
    for weight, group in detail_df.groupby("under_weight", sort=True):
        best = group.sort_values("inventory_cost_proxy", ascending=True).iloc[0]
        summary_rows.append(
            {
                "under_weight": float(weight),
                "best_quantile_idx": int(best["quantile_idx"]),
                "best_quantile": float(best["quantile"]),
                "best_weighted_smape": float(best["weighted_smape"]),
                "best_underestimation_rate": float(best["underestimation_rate"]),
                "best_inventory_cost_proxy": float(best["inventory_cost_proxy"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("best_inventory_cost_proxy", ascending=True)
    recommended_weight = _recommend_under_weight(summary_df=summary_df, preferred=1.5)

    detail_path = output_root / "loss_calibration_detail.csv"
    summary_path = output_root / "loss_calibration_summary.md"

    detail_df.to_csv(detail_path, index=False)

    summary_for_md = summary_df.copy()
    summary_for_md["recommended"] = summary_for_md["under_weight"].apply(
        lambda value: "yes" if np.isclose(float(value), float(recommended_weight)) else ""
    )
    summary_markdown = (
        "# Asymmetric Loss Calibration\n\n"
        "Best quantile per under-estimation penalty factor.\n\n"
        + summary_for_md.to_markdown(index=False)
        + f"\n\nRecommended under_penalty_factor: {recommended_weight:.2f}\n"
    )
    summary_path.write_text(summary_markdown, encoding="utf-8")

    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(detail_path))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_metric("calibration_recommended_under_penalty", float(recommended_weight))

    return detail_path, summary_path, recommended_weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tft/tft_latest.pt")
    parser.add_argument("--config", default="data/processed/tft_runtime_config.yaml")
    parser.add_argument("--reference", default="data/processed/tft_reference.parquet")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--under-weights", default="1.0,1.25,1.5,1.75,2.0")
    args = parser.parse_args()

    detail_path, summary_path, recommended = run_calibration(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        reference_path=args.reference,
        output_dir=args.output_dir,
        under_weights=_parse_under_weights(args.under_weights),
    )

    print(f"Calibration detail CSV: {detail_path}")
    print(f"Calibration summary MD: {summary_path}")
    print(f"Recommended under_penalty_factor: {recommended:.2f}")


if __name__ == "__main__":
    main()
