"""Rossmann Store Sales dataset loader.

Loads raw CSV files from disk, merges store metadata, and returns
clean DataFrames ready for feature engineering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pandas as pd

from src.data.schemas import LoaderOutputSchema
from src.data.validator import validate_pipeline_schema

logger = logging.getLogger(__name__)


class RossmannDataLoader:
    """Loads and merges Rossmann Store Sales CSV files.

    Args:
        data_dir: Path to the directory containing raw CSV files.

    Example:
        >>> loader = RossmannDataLoader("data/raw")
        >>> train_df, test_df = loader.load()
    """

    def __init__(self, data_dir: str | Path) -> None:
        """Initialize loader with raw data directory.

        Args:
            data_dir: Directory containing Rossmann CSV files.
        """
        self.data_dir = Path(data_dir)

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and merge train/test with store metadata.

        Returns:
            Tuple of (train_df, test_df) with store columns merged in.

        Raises:
            FileNotFoundError: If any required CSV is missing.
        """
        train = self._read_csv("train.csv")
        test = self._read_csv("test.csv")
        store = self._read_csv("store.csv")

        train = train.merge(store, on="Store", how="left")
        test = test.merge(store, on="Store", how="left")

        train["Date"] = pd.to_datetime(train["Date"])
        test["Date"] = pd.to_datetime(test["Date"])

        if "Open" in train.columns:
            train = train[train["Open"] == 1].copy()
        if "Sales" in train.columns:
            train = train[train["Sales"] > 0].copy()

        train = self.validate_train_schema(train)
        logger.info("Loaded train=%d rows, test=%d rows", len(train), len(test))
        return train, test

    @validate_pipeline_schema(
        required_columns=cast(list[str], LoaderOutputSchema().required_columns),
        non_nullable_columns=cast(list[str], LoaderOutputSchema().non_nullable_columns),
    )
    def validate_train_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate schema contract for train dataframe."""
        return df

    def _read_csv(self, filename: str) -> pd.DataFrame:
        """Read a CSV file from the configured raw data directory.

        Args:
            filename: CSV file name.

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.read_csv(path, low_memory=False)

    _schema = LoaderOutputSchema()
