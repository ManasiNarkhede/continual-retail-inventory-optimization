"""End-to-end TFT training script.

Usage:
    python scripts/train_tft.py --config configs/tft_config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.validator import validate_configs
from src.data.feature_engineer import FeatureEngineer
from src.data.loader import RossmannDataLoader
from src.data.preprocessor import RossmannPreprocessor
from src.eval.forecast_metrics import quantile_coverage, rmse, smape
from src.models.lr_finder import find_optimal_lr
from src.models.tft_dataset import build_tft_dataset
from src.models.tft_model import TFTForecaster
from src.models.tft_trainer import TFTTrainer

logger = logging.getLogger(__name__)


def _fingerprint_frame(df: pd.DataFrame, key_columns: list[str]) -> str:
    """Build deterministic fingerprint from selected columns.

    Args:
        df: DataFrame to fingerprint.
        key_columns: Columns used to generate deterministic hash.

    Returns:
        Hex digest string.
    """
    available = [column for column in key_columns if column in df.columns]
    if not available:
        return ""
    ordered = df[available].copy()
    csv_blob = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_blob).hexdigest()


def _fingerprint_index_frame(df: pd.DataFrame) -> str:
    """Build deterministic hash for a dataset index frame.

    Args:
        df: Dataset index frame.

    Returns:
        Hex digest string.
    """
    ordered = df.sort_values(list(df.columns)).reset_index(drop=True)
    csv_blob = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_blob).hexdigest()


def _prepare_runtime_config(
    base_config_path: str,
    output_config_path: Path,
    override_max_epochs: int | None,
    override_batch_size: int | None,
    override_hidden_size: int | None,
    override_hidden_cont_size: int | None,
    override_lstm_layers: int | None,
    override_attention_heads: int | None,
    override_max_encoder_length: int | None,
    override_max_prediction_length: int | None,
) -> dict:
    """Load base TFT config, apply runtime overrides, and persist it.

    Args:
        base_config_path: Source YAML config path.
        output_config_path: Destination path for resolved runtime config.
        override_*: Optional runtime overrides.

    Returns:
        Resolved TFT config dictionary.
    """
    with Path(base_config_path).open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    tft_cfg = payload["tft"]
    if override_max_epochs is not None:
        tft_cfg["max_epochs"] = int(override_max_epochs)
    if override_batch_size is not None:
        tft_cfg["batch_size"] = int(override_batch_size)
    if override_hidden_size is not None:
        tft_cfg["hidden_size"] = int(override_hidden_size)
    if override_hidden_cont_size is not None:
        tft_cfg["hidden_continuous_size"] = int(override_hidden_cont_size)
    if override_lstm_layers is not None:
        tft_cfg["lstm_layers"] = int(override_lstm_layers)
    if override_attention_heads is not None:
        tft_cfg["num_attention_heads"] = int(override_attention_heads)
    if override_max_encoder_length is not None:
        tft_cfg["max_encoder_length"] = int(override_max_encoder_length)
    if override_max_prediction_length is not None:
        tft_cfg["max_prediction_length"] = int(override_max_prediction_length)

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return tft_cfg


def main(
    config_path: str,
    store_id: int | None = None,
    max_rows: int | None = None,
    max_rows_per_store: int | None = None,
    override_max_epochs: int | None = None,
    override_batch_size: int | None = None,
    override_hidden_size: int | None = None,
    override_hidden_cont_size: int | None = None,
    override_lstm_layers: int | None = None,
    override_attention_heads: int | None = None,
    override_max_encoder_length: int | None = None,
    override_max_prediction_length: int | None = None,
) -> None:
    """Run complete TFT training pipeline.

    Args:
        config_path: Path to TFT YAML configuration.
        store_id: Optional store id filter for faster execution.
        max_rows: Optional cap on rows after sorting by date.
        max_rows_per_store: Optional per-store cap preserving multi-store coverage.
        override_max_epochs: Optional max epoch override.
        override_batch_size: Optional batch size override.
        override_hidden_size: Optional hidden size override.
        override_hidden_cont_size: Optional hidden continuous size override.
        override_lstm_layers: Optional LSTM layer count override.
        override_attention_heads: Optional attention head override.
        override_max_encoder_length: Optional encoder length override.
        override_max_prediction_length: Optional prediction length override.
    """
    validate_configs(tft_config_path=config_path, rl_config_path="configs/rl_config.yaml")
    mlflow.set_experiment("retail_tft_forecasting")
    with mlflow.start_run(run_name="tft_baseline"):
        processed_dir = Path("data/processed")
        runtime_config_path = processed_dir / "tft_runtime_config.yaml"
        runtime_config_work_path = processed_dir / "tft_runtime_config.work.yaml"
        tft_cfg = _prepare_runtime_config(
            base_config_path=config_path,
            output_config_path=runtime_config_work_path,
            override_max_epochs=override_max_epochs,
            override_batch_size=override_batch_size,
            override_hidden_size=override_hidden_size,
            override_hidden_cont_size=override_hidden_cont_size,
            override_lstm_layers=override_lstm_layers,
            override_attention_heads=override_attention_heads,
            override_max_encoder_length=override_max_encoder_length,
            override_max_prediction_length=override_max_prediction_length,
        )

        loader = RossmannDataLoader("data/raw")
        train_df, _ = loader.load()

        if store_id is not None:

            # Store range like 1-50
            if isinstance(store_id, str) and "-" in store_id:

                start_store, end_store = map(int, store_id.split("-"))

                train_df = train_df[
                    (train_df["Store"] >= start_store) & (train_df["Store"] <= end_store)
                ]

                print(f"Training stores from {start_store} to {end_store}")

            else:

                train_df = train_df[train_df["Store"] == int(store_id)]

        print(f"Training store {store_id}")
        if max_rows is not None and max_rows > 0:
            train_df = train_df.sort_values("Date").tail(max_rows)
        if max_rows_per_store is not None and max_rows_per_store > 0:
            train_df = (
                train_df.sort_values(["Store", "Date"])
                .groupby("Store", group_keys=False)
                .tail(max_rows_per_store)
            )

        preprocessor = RossmannPreprocessor()
        train_df = preprocessor.fit_transform(train_df)

        feature_engineer = FeatureEngineer()
        train_df = feature_engineer.transform(train_df)

        required_cols = [
            "time_idx",
            "Store",
            tft_cfg["target"],
            *tft_cfg.get("static_categoricals", []),
            *tft_cfg.get("static_reals", []),
            *tft_cfg.get("time_varying_known_reals", []),
            *tft_cfg.get("time_varying_unknown_reals", []),
        ]
        existing_required = [column for column in required_cols if column in train_df.columns]
        train_df = train_df.dropna(subset=existing_required)

        processed_dir.mkdir(parents=True, exist_ok=True)
        reference_df = train_df.sort_values(["Store", "time_idx"]).reset_index(drop=True)

        train_dataset = build_tft_dataset(train_df, config_path=str(runtime_config_work_path))
        val_dataset = build_tft_dataset(
            train_df,
            config_path=str(runtime_config_work_path),
            training=False,
            training_dataset=train_dataset,
        )

        forecaster = TFTForecaster(str(runtime_config_work_path), train_dataset)
        model = forecaster.build()

        train_loader_for_lr = train_dataset.to_dataloader(
            train=True,
            batch_size=int(tft_cfg["batch_size"]),
            num_workers=0,
        )
        val_loader_for_lr = val_dataset.to_dataloader(
            train=False,
            batch_size=int(tft_cfg["batch_size"]),
            num_workers=0,
        )
        try:
            optimal_lr = find_optimal_lr(
                model,
                train_loader_for_lr,
                val_loader_for_lr,
                num_training_steps=50,
            )
            mlflow.log_param("lr_finder_suggestion", float(optimal_lr))
            logger.info("Using LR finder suggestion: %.3e", optimal_lr)
        except Exception as e:
            optimal_lr = tft_cfg.get("learning_rate", 0.001)
            logger.warning("LR finder failed (%s). Using config LR: %.3e", e, optimal_lr)
            mlflow.log_param("lr_finder_failed", True)
            mlflow.log_param("lr_finder_error", str(e))

        trainer = TFTTrainer(str(runtime_config_work_path))
        trainer.fit(model, train_dataset, val_dataset)

        checkpoint_dir = Path("checkpoints/tft")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_base_path = checkpoint_dir / "tft_latest"
        forecaster.save_model(str(model_base_path))

        val_loader = val_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)
        predictions, _ = forecaster.predict(val_loader, mode="quantiles")
        prediction_array = (
            predictions.detach().cpu().numpy()
            if hasattr(predictions, "detach")
            else np.asarray(predictions)
        )
        if prediction_array.ndim == 3:
            prediction_array = prediction_array.reshape(-1, prediction_array.shape[-1])

        # DEBUG: Log prediction array stats
        print(
            f"DEBUG: prediction_array shape={prediction_array.shape}, ndim={prediction_array.ndim}"
        )
        print(
            f"DEBUG: prediction_array NaN count={int(np.isnan(prediction_array).sum())}, "
            f"Inf count={int(np.isinf(prediction_array).sum())}"
        )
        if prediction_array.ndim > 1 and prediction_array.shape[1] > 3:
            print(
                "DEBUG: prediction_array q50 NaN count="
                f"{int(np.isnan(prediction_array[:, 3]).sum())}"
            )

        y_true_batches: list[np.ndarray] = []
        baseline_batches: list[np.ndarray] = []
        for x_batch, target_tuple in val_loader:
            target = target_tuple[0]
            target_np = (
                target.detach().cpu().numpy() if hasattr(target, "detach") else np.asarray(target)
            )
            y_true_batches.append(target_np.reshape(-1).astype(float))

            encoder_target = x_batch.get("encoder_target")
            if encoder_target is None:
                continue
            enc_np = (
                encoder_target.detach().cpu().numpy()
                if hasattr(encoder_target, "detach")
                else np.asarray(encoder_target)
            )
            last_value = enc_np[:, -1].reshape(-1, 1)
            repeated = np.repeat(last_value, target_np.shape[1], axis=1)
            baseline_batches.append(repeated.reshape(-1).astype(float))

        y_true = np.concatenate(y_true_batches) if y_true_batches else np.array([], dtype=float)
        baseline_pred = (
            np.concatenate(baseline_batches) if baseline_batches else np.array([], dtype=float)
        )

        # DEBUG: Log raw alignment stats
        print(
            f"DEBUG: y_true shape={y_true.shape}, NaN count={int(np.isnan(y_true).sum())}, "
            f"Inf count={int(np.isinf(y_true).sum())}"
        )
        print(
            f"DEBUG: baseline_pred shape={baseline_pred.shape}, "
            f"NaN count={int(np.isnan(baseline_pred).sum())}, "
            f"Inf count={int(np.isinf(baseline_pred).sum())}"
        )

        aligned_len = min(len(y_true), len(prediction_array))
        y_true = y_true[:aligned_len]
        prediction_array = prediction_array[:aligned_len]
        baseline_pred = (
            baseline_pred[:aligned_len]
            if len(baseline_pred) >= aligned_len
            else np.full(aligned_len, np.nan)
        )

        print(f"DEBUG: aligned_len={aligned_len}")

        if aligned_len == 0:
            raise RuntimeError("Validation alignment produced zero rows; unable to log metrics.")

        finite_mask = np.isfinite(y_true) & np.isfinite(prediction_array[:, 3])
        if prediction_array.shape[1] >= 6:
            finite_mask &= np.isfinite(prediction_array[:, 1]) & np.isfinite(prediction_array[:, 5])
        if baseline_pred.shape[0] == finite_mask.shape[0]:
            finite_mask &= np.isfinite(baseline_pred)

        print(f"DEBUG: finite_mask sum={int(finite_mask.sum())}, total={len(finite_mask)}")

        if not np.any(finite_mask):
            raise RuntimeError("No finite rows available for validation metrics after filtering.")

        dropped = int((~finite_mask).sum())
        if dropped > 0:
            logger.warning("Dropping %d non-finite rows before metric computation.", dropped)
            print(f"DEBUG: Dropping {dropped} non-finite rows before metric computation.")

        y_true = y_true[finite_mask]
        prediction_array = prediction_array[finite_mask]
        if baseline_pred.shape[0] == finite_mask.shape[0]:
            baseline_pred = baseline_pred[finite_mask]

        # DEBUG: Log post-filter stats
        print(
            f"DEBUG: post-filter y_true shape={y_true.shape}, "
            f"min={float(y_true.min()):.4f}, max={float(y_true.max()):.4f}"
        )
        print(
            f"DEBUG: post-filter prediction_array q50 shape={prediction_array[:, 3].shape}, "
            f"min={float(prediction_array[:, 3].min()):.4f}, "
            f"max={float(prediction_array[:, 3].max()):.4f}"
        )

        val_smape_computed = smape(y_true, prediction_array[:, 3])
        val_rmse_computed = rmse(y_true, prediction_array[:, 3])
        val_coverage_computed = quantile_coverage(
            y_true, prediction_array[:, 1], prediction_array[:, 5]
        )
        print(
            f"DEBUG: val_smape={val_smape_computed}, val_rmse={val_rmse_computed}, "
            f"val_coverage={val_coverage_computed}"
        )

        if not np.isfinite(val_smape_computed):
            raise RuntimeError(
                f"Validation SMAPE is non-finite ({val_smape_computed}); "
                "model predictions may be NaN/Inf. Check for undertraining or data issues."
            )
        if not np.isfinite(val_rmse_computed):
            raise RuntimeError(
                f"Validation RMSE is non-finite ({val_rmse_computed}); "
                "model predictions may be NaN/Inf. Check for undertraining or data issues."
            )
        if not np.isfinite(val_coverage_computed):
            raise RuntimeError(
                f"Validation coverage is non-finite ({val_coverage_computed}); "
                "quantile predictions may be NaN/Inf."
            )

        mlflow.log_metric("val_smape", val_smape_computed)
        mlflow.log_metric("val_rmse", val_rmse_computed)
        mlflow.log_metric("val_coverage_80pct", val_coverage_computed)

        if np.isfinite(baseline_pred).all() and len(baseline_pred) == aligned_len:
            baseline_smape = smape(y_true, baseline_pred)
            baseline_rmse = rmse(y_true, baseline_pred)
            model_smape = smape(y_true, prediction_array[:, 3])
            model_rmse = rmse(y_true, prediction_array[:, 3])
            if np.isfinite(baseline_smape) and np.isfinite(model_smape):
                mlflow.log_metric("baseline_smape", baseline_smape)
                mlflow.log_metric("smape_uplift_vs_baseline", baseline_smape - model_smape)
            if np.isfinite(baseline_rmse) and np.isfinite(model_rmse):
                mlflow.log_metric("baseline_rmse", baseline_rmse)
                mlflow.log_metric("rmse_uplift_vs_baseline", baseline_rmse - model_rmse)

        val_smape_value = float(val_smape_computed)
        val_rmse_value = float(val_rmse_computed)

        metrics_payload = {
            "model_version": model_base_path.with_suffix(".pt").name,
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "val_smape": val_smape_value,
            "val_rmse": val_rmse_value,
            "val_coverage_80pct": float(val_coverage_computed),
            "stores_trained": (
                int(reference_df["Store"].nunique()) if "Store" in reference_df.columns else 0
            ),
        }
        if np.isfinite(baseline_pred).all() and len(baseline_pred) == aligned_len:
            baseline_smape_val = float(smape(y_true, baseline_pred))
            baseline_rmse_val = float(rmse(y_true, baseline_pred))
            if np.isfinite(baseline_smape_val):
                metrics_payload["baseline_smape"] = baseline_smape_val
            if np.isfinite(baseline_rmse_val):
                metrics_payload["baseline_rmse"] = baseline_rmse_val

        reference_path = processed_dir / "tft_reference.parquet"
        reference_df.to_parquet(reference_path, index=False)
        mlflow.log_artifact(str(reference_path))

        split_payload = {
            "train_index_rows": int(len(train_dataset.index)),
            "val_index_rows": int(len(val_dataset.index)),
            "train_index_sha256": _fingerprint_index_frame(train_dataset.index),
            "val_index_sha256": _fingerprint_index_frame(val_dataset.index),
        }
        split_path = processed_dir / "tft_split_fingerprint.json"
        split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(split_path))

        metrics_path = processed_dir / "metrics_latest.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path))

        fingerprint_payload = {
            "rows": int(len(reference_df)),
            "stores": (
                int(reference_df["Store"].nunique()) if "Store" in reference_df.columns else 0
            ),
            "min_date": str(reference_df["Date"].min()) if "Date" in reference_df.columns else "",
            "max_date": str(reference_df["Date"].max()) if "Date" in reference_df.columns else "",
            "sha256": _fingerprint_frame(reference_df, ["Store", "Date", "time_idx", "sales"]),
            "config_path": str(runtime_config_path.resolve()),
        }
        fingerprint_path = processed_dir / "tft_fingerprint.json"
        fingerprint_path.write_text(json.dumps(fingerprint_payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(fingerprint_path))

        runtime_config_path.write_text(
            runtime_config_work_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

        if runtime_config_work_path.exists():
            runtime_config_work_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tft_config.yaml")
    parser.add_argument("--store-id", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-rows-per-store", type=int, default=None)
    parser.add_argument("--override-max-epochs", type=int, default=None)
    parser.add_argument("--override-batch-size", type=int, default=None)
    parser.add_argument("--override-hidden-size", type=int, default=None)
    parser.add_argument("--override-hidden-cont-size", type=int, default=None)
    parser.add_argument("--override-lstm-layers", type=int, default=None)
    parser.add_argument("--override-attention-heads", type=int, default=None)
    parser.add_argument("--override-max-encoder-length", type=int, default=None)
    parser.add_argument("--override-max-prediction-length", type=int, default=None)
    args = parser.parse_args()
    main(
        config_path=args.config,
        store_id=args.store_id,
        max_rows=args.max_rows,
        max_rows_per_store=args.max_rows_per_store,
        override_max_epochs=args.override_max_epochs,
        override_batch_size=args.override_batch_size,
        override_hidden_size=args.override_hidden_size,
        override_hidden_cont_size=args.override_hidden_cont_size,
        override_lstm_layers=args.override_lstm_layers,
        override_attention_heads=args.override_attention_heads,
        override_max_encoder_length=args.override_max_encoder_length,
        override_max_prediction_length=args.override_max_prediction_length,
    )
