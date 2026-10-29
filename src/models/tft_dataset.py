"""Build TimeSeriesDataSet objects for TFT workflows."""

from __future__ import annotations

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from configs.validator import load_yaml_config, validate_config_schema


def validate_tft_feature_schema(
    df: pd.DataFrame, config_path: str = "configs/tft_config.yaml"
) -> None:
    """Validate required TFT schema and NaN safety before dataset build.

    Args:
        df: Feature-engineered dataframe.
        config_path: TFT config path.

    Raises:
        ValueError: If required columns are missing or target/time/group contain NaNs.
    """
    validate_config_schema(config_path, "configs/schemas/tft_schema.json")
    cfg = load_yaml_config(config_path).get("tft", {})

    required = {
        "time_idx",
        "Store",
        cfg.get("target", "sales"),
        *cfg.get("static_categoricals", []),
        *cfg.get("static_reals", []),
        *cfg.get("time_varying_known_reals", []),
        *cfg.get("time_varying_unknown_reals", []),
    }
    required = {col for col in required if isinstance(col, str) and col}

    missing = sorted([col for col in required if col not in df.columns])
    if missing:
        raise ValueError(f"Missing required TFT columns: {missing}")

    critical = ["time_idx", "Store", cfg.get("target", "sales")]
    for col in critical:
        if df[col].isna().any():
            raise ValueError(
                f"Column '{col}' contains NaN values; cannot build leakage-safe dataset."
            )


def build_tft_dataset(
    df: pd.DataFrame,
    config_path: str = "configs/tft_config.yaml",
    training: bool = True,
    training_dataset: TimeSeriesDataSet | None = None,
) -> TimeSeriesDataSet:
    """Build a TimeSeriesDataSet for TFT training or validation.

    Args:
        df: Feature-engineered dataframe with time_idx column.
        config_path: Path to the TFT YAML config.
        training: Whether to create a training dataset.
        training_dataset: Required for validation dataset creation.

    Returns:
        Configured TimeSeriesDataSet instance.

    Raises:
        ValueError: If training is False and no training_dataset is provided.
    """
    data = df.copy()
    if "store" in data.columns and "Store" not in data.columns:
        data["Store"] = data["store"]
    if "store" not in data.columns and "Store" in data.columns:
        data["store"] = data["Store"]

    validate_tft_feature_schema(data, config_path=config_path)

    validate_config_schema(config_path, "configs/schemas/tft_schema.json")
    cfg = load_yaml_config(config_path).get("tft", {})

    for categorical_column in cfg.get("static_categoricals", []):
        if categorical_column in data.columns:
            data[categorical_column] = data[categorical_column].astype(str)

    real_columns = (
        cfg.get("static_reals", [])
        + cfg.get("time_varying_known_reals", [])
        + cfg.get("time_varying_unknown_reals", [])
        + [cfg.get("target", "")]
    )
    for real_column in [column for column in real_columns if column and column in data.columns]:
        data[real_column] = pd.to_numeric(data[real_column], errors="coerce").astype(float)

    if not training:
        if training_dataset is None:
            raise ValueError("training_dataset required for validation set creation.")
        return TimeSeriesDataSet.from_dataset(
            training_dataset,
            data,
            predict=False,
            stop_randomization=True,
        )

    max_time_idx = int(data["time_idx"].max())
    training_cutoff = max_time_idx - int(cfg["max_prediction_length"])

    dataset = TimeSeriesDataSet(
        data[data["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target=cfg["target"],
        group_ids=["Store"],
        min_encoder_length=cfg["max_encoder_length"] // 2,
        max_encoder_length=cfg["max_encoder_length"],
        min_prediction_length=1,
        max_prediction_length=cfg["max_prediction_length"],
        static_categoricals=cfg["static_categoricals"],
        static_reals=cfg["static_reals"],
        time_varying_known_reals=cfg["time_varying_known_reals"],
        time_varying_unknown_reals=cfg["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(groups=["Store"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset
