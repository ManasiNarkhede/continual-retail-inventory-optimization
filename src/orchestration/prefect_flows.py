"""Optional Prefect-style orchestration entrypoints.

The module is intentionally dependency-light: it can be imported even when
Prefect is not installed, and falls back to plain Python execution.
"""

from __future__ import annotations

from typing import Any

from src.data.feature_engineer import FeatureEngineer
from src.data.loader import RossmannDataLoader
from src.data.preprocessor import RossmannPreprocessor


def load_data(data_dir: str = "data/raw"):
    """Load Rossmann train/test datasets."""
    loader = RossmannDataLoader(data_dir)
    return loader.load()


def preprocess_data(train_df):
    """Run preprocessing on train dataframe."""
    return RossmannPreprocessor().fit_transform(train_df)


def engineer_features(train_df):
    """Run feature engineering on preprocessed dataframe."""
    return FeatureEngineer().transform(train_df)


def run_retail_pipeline(data_dir: str = "data/raw") -> dict[str, Any]:
    """Run a lightweight orchestrated pipeline without external scheduler."""
    train_df, test_df = load_data(data_dir=data_dir)
    preprocessed = preprocess_data(train_df)
    features = engineer_features(preprocessed)
    return {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_rows": int(len(features)),
        "feature_columns": int(len(features.columns)),
    }


def build_prefect_flow() -> Any:
    """Create and return a Prefect flow when Prefect is installed.

    Returns:
        Prefect flow object if available, else ``None``.
    """
    try:
        from prefect import flow, task
    except Exception:
        return None

    @task
    def _load_data_task(data_dir: str = "data/raw"):
        return load_data(data_dir=data_dir)

    @task
    def _preprocess_task(train_df):
        return preprocess_data(train_df)

    @task
    def _engineer_task(train_df):
        return engineer_features(train_df)

    @flow(name="retail-inventory-pipeline")
    def retail_pipeline(data_dir: str = "data/raw"):
        train_df, test_df = _load_data_task(data_dir=data_dir)
        preprocessed = _preprocess_task(train_df)
        features = _engineer_task(preprocessed)
        return {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "feature_rows": int(len(features)),
        }

    return retail_pipeline
