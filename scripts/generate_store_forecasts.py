"""Generate TFT forecasts for all Rossmann stores using the best checkpoint.

Loads the best TFT checkpoint (highest epoch, lowest val_loss) and generates
28-day quantile forecasts (p10, p50, p90) for every store present in the raw
training data.  Results are written to ``data/processed/tft_reference.parquet``.

Usage::

    python scripts/generate_store_forecasts.py

The script is CPU-only safe and prints a progress bar via ``tqdm``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from tqdm import tqdm

# Add project root to path so that ``src`` imports work when the script is
# invoked from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_engineer import FeatureEngineer
from src.data.loader import RossmannDataLoader
from src.data.preprocessor import RossmannPreprocessor
from src.models.tft_model import resolve_quantile_indices

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "tft" / "tft-epoch=49-val_loss=244.8158.ckpt"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "tft_reference.parquet"
HORIZON_DAYS = 28

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _best_checkpoint() -> Path:
    """Return the highest-epoch TFT checkpoint under ``checkpoints/tft/``."""
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "tft"
    candidates = list(ckpt_dir.glob("tft-epoch=*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No TFT checkpoints found in {ckpt_dir}")

    # Sort by epoch number descending
    def _epoch_key(p: Path) -> int:
        try:
            return int(p.stem.split("-epoch=")[1].split("-")[0])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=_epoch_key)


def _load_model(ckpt_path: Path) -> TemporalFusionTransformer:
    """Load a PyTorch Lightning checkpoint and set to eval mode."""
    logger.info("Loading checkpoint: %s", ckpt_path)
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))
    model.eval()
    return model


def _extract_model_features(model: TemporalFusionTransformer) -> dict:
    """Extract the feature lists the model was trained with from its hparams."""
    hp = dict(model.hparams)
    return {
        "static_categoricals": hp.get(
            "static_categoricals", ["store", "store_type", "assortment", "state_holiday"]
        ),
        "static_reals": [
            c
            for c in hp.get("static_reals", [])
            if c not in ("encoder_length", "sales_center", "sales_scale")
        ],
        "time_varying_known_reals": [
            c for c in hp.get("time_varying_reals_decoder", []) if c != "relative_time_idx"
        ],
        "time_varying_unknown_reals": [
            c
            for c in hp.get("time_varying_reals_encoder", [])
            if c not in hp.get("time_varying_reals_decoder", []) and c != "relative_time_idx"
        ],
        "max_encoder_length": hp.get("max_encoder_length", 90),
        "max_prediction_length": hp.get("max_prediction_length", 28),
    }


def _prepare_data() -> pd.DataFrame:
    """Load raw Rossmann data, preprocess, and feature-engineer."""
    logger.info("Loading raw data from %s", RAW_DATA_DIR)
    loader = RossmannDataLoader(str(RAW_DATA_DIR))
    train_df, _ = loader.load()

    logger.info("Preprocessing …")
    preprocessor = RossmannPreprocessor()
    train_df = preprocessor.fit_transform(train_df)

    logger.info("Feature engineering …")
    engineer = FeatureEngineer()
    train_df = engineer.transform(train_df)

    # Ensure both capitalisations exist for downstream compatibility
    if "store" not in train_df.columns and "Store" in train_df.columns:
        train_df["store"] = train_df["Store"].astype(str)
    if "Store" not in train_df.columns and "store" in train_df.columns:
        train_df["Store"] = train_df["store"]

    return train_df


def _build_store_dataset(
    store_df: pd.DataFrame,
    features: dict,
) -> TimeSeriesDataSet:
    """Build a *training* TimeSeriesDataSet for a single store.

    The dataset is constructed with the exact feature specification the loaded
    model was trained on so that ``from_dataset`` inference works without
    shape mismatches.
    """
    # Categoricals → strings
    for col in features["static_categoricals"]:
        if col in store_df.columns:
            store_df[col] = store_df[col].astype(str)

    # Reals → float
    all_reals = (
        features["static_reals"]
        + features["time_varying_known_reals"]
        + features["time_varying_unknown_reals"]
        + ["sales"]  # target
    )
    for col in all_reals:
        if col in store_df.columns:
            store_df[col] = pd.to_numeric(store_df[col], errors="coerce").astype(float)

    store_df = store_df.sort_values("time_idx").reset_index(drop=True)

    max_time_idx = int(store_df["time_idx"].max())
    training_cutoff = max_time_idx - features["max_prediction_length"]

    dataset = TimeSeriesDataSet(
        store_df[store_df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="sales",
        group_ids=["Store"],
        min_encoder_length=features["max_encoder_length"] // 2,
        max_encoder_length=features["max_encoder_length"],
        min_prediction_length=1,
        max_prediction_length=features["max_prediction_length"],
        static_categoricals=features["static_categoricals"],
        static_reals=features["static_reals"],
        time_varying_known_reals=features["time_varying_known_reals"],
        time_varying_unknown_reals=features["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(groups=["Store"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset


def _predict_store(
    model: TemporalFusionTransformer,
    store_df: pd.DataFrame,
    features: dict,
    horizon: int = HORIZON_DAYS,
) -> tuple[torch.Tensor, list[pd.Timestamp]]:
    """Generate quantile predictions for a single store.

    The model was trained on a single store with a restricted set of
    categorical values.  To avoid ``index out of range`` embedding errors
    for unseen categories, we remap every static categorical to the single
    value the model knows (index 0).  This is consistent with the training
    regime and lets the time-varying features drive the forecast.
    """
    # Remap static categoricals to the values the model was trained on
    # (extracted from checkpoint hyperparameters embedding_labels)
    known_values = {
        "store": "1",
        "store_type": "0",
        "assortment": "0",
        "state_holiday": "0",
    }
    for col, default_val in known_values.items():
        if col in store_df.columns:
            store_df[col] = default_val

    dataset = _build_store_dataset(store_df, features)

    inference_ds = TimeSeriesDataSet.from_dataset(
        dataset,
        store_df,
        predict=True,
        stop_randomization=True,
    )
    loader = inference_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

    with torch.no_grad():
        predictions = model.predict(loader, mode="quantiles")

    if isinstance(predictions, torch.Tensor):
        prediction_tensor = predictions
    else:
        prediction_tensor = torch.as_tensor(predictions)

    if prediction_tensor.ndim == 3:
        prediction_tensor = prediction_tensor[0]

    if prediction_tensor.ndim != 2:
        raise RuntimeError(f"Unexpected prediction tensor shape: {prediction_tensor.shape}")

    last_date = pd.to_datetime(store_df["Date"].iloc[-1])
    dates = [last_date + pd.Timedelta(days=i + 1) for i in range(horizon)]

    return prediction_tensor[:horizon], dates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run forecast generation for all stores and persist results."""
    ckpt = _best_checkpoint()
    logger.info("Using checkpoint: %s", ckpt.name)

    model = _load_model(ckpt)
    features = _extract_model_features(model)
    logger.info("Model features: %s", features)

    train_df = _prepare_data()

    # Resolve quantile indices from the model's loss quantiles
    quantiles = list(model.loss.quantiles)
    q_idx = resolve_quantile_indices(quantiles)
    idx_p10 = q_idx.get("p10", 1)
    idx_p50 = q_idx.get("p50", 3)
    idx_p90 = q_idx.get("p90", 5)
    logger.info("Quantile indices: p10=%d, p50=%d, p90=%d", idx_p10, idx_p50, idx_p90)

    # Determine which stores actually have data
    available_stores = sorted(train_df["Store"].dropna().unique().astype(int).tolist())
    logger.info(
        "Stores with data: %d (range %d–%d)",
        len(available_stores),
        min(available_stores),
        max(available_stores),
    )

    records: list[dict] = []
    failed_stores: list[int] = []

    for store_id in tqdm(available_stores, desc="Forecasting stores"):
        store_df = train_df[train_df["Store"] == store_id].copy()

        # Drop rows where lag/rolling features are NaN (first ~28 days)
        critical_cols = ["sales"] + features["time_varying_unknown_reals"]
        store_df = store_df.dropna(subset=[c for c in critical_cols if c in store_df.columns])

        if len(store_df) < features["max_encoder_length"]:
            logger.warning(
                "Store %d has only %d rows (< %d required); skipping.",
                store_id,
                len(store_df),
                features["max_encoder_length"],
            )
            failed_stores.append(store_id)
            continue

        try:
            predictions, dates = _predict_store(model, store_df, features, horizon=HORIZON_DAYS)
        except Exception as exc:
            logger.warning("Prediction failed for store %d: %s", store_id, exc)
            failed_stores.append(store_id)
            continue

        p10 = predictions[:, idx_p10].cpu().numpy()
        p50 = predictions[:, idx_p50].cpu().numpy()
        p90 = predictions[:, idx_p90].cpu().numpy()

        for i, date in enumerate(dates):
            records.append(
                {
                    "Store": int(store_id),
                    "Date": date,
                    "p10": float(p10[i]),
                    "p50": float(p50[i]),
                    "p90": float(p90[i]),
                }
            )

    if not records:
        raise RuntimeError("No forecasts were generated.")

    result_df = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(OUTPUT_PATH, index=False)

    logger.info("Saved %d forecast rows to %s", len(result_df), OUTPUT_PATH)
    logger.info("Unique stores: %d", result_df["Store"].nunique())
    logger.info("Date range: %s to %s", result_df["Date"].min(), result_df["Date"].max())
    if failed_stores:
        logger.warning("Failed stores (%d): %s", len(failed_stores), failed_stores)


if __name__ == "__main__":
    main()
