"""Unit tests for TFT dataset builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.models.tft_dataset import build_tft_dataset, validate_tft_feature_schema


@pytest.fixture
def config_path(tmp_path: Path) -> str:
    """Create temporary TFT config file.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        Config file path.
    """
    cfg = {
        "tft": {
            "max_prediction_length": 7,
            "max_encoder_length": 14,
            "target": "sales",
            "quantiles": [0.1, 0.5, 0.9],
            "static_categoricals": ["store"],
            "static_reals": ["competition_distance"],
            "time_varying_known_reals": ["day_of_week", "month"],
            "time_varying_unknown_reals": ["sales"],
        }
    }
    path = tmp_path / "tft_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create minimal dataframe compatible with TimeSeriesDataSet.

    Returns:
        Input dataframe fixture.
    """
    days = pd.date_range("2015-01-01", periods=30)
    return pd.DataFrame(
        {
            "Store": [1] * len(days),
            "Date": days,
            "time_idx": list(range(len(days))),
            "store": [1] * len(days),
            "competition_distance": [100.0] * len(days),
            "day_of_week": [d.dayofweek for d in days],
            "month": [d.month for d in days],
            "sales": [100.0 + i for i in range(len(days))],
        }
    )


def test_build_training_dataset(sample_df: pd.DataFrame, config_path: str) -> None:
    """Ensure training dataset can be built.

    Args:
        sample_df: Input dataframe fixture.
        config_path: Temporary TFT config path.
    """
    dataset = build_tft_dataset(sample_df, config_path=config_path)
    assert dataset is not None


def test_validation_requires_training_dataset(sample_df: pd.DataFrame, config_path: str) -> None:
    """Ensure validation mode requires training reference dataset.

    Args:
        sample_df: Input dataframe fixture.
        config_path: Temporary TFT config path.
    """
    with pytest.raises(ValueError):
        build_tft_dataset(sample_df, config_path=config_path, training=False)


def test_validate_tft_feature_schema_missing_column(
    sample_df: pd.DataFrame, config_path: str
) -> None:
    """Schema validator should fail when required column is missing."""
    broken = sample_df.drop(columns=["sales"])
    with pytest.raises(ValueError):
        validate_tft_feature_schema(broken, config_path=config_path)
