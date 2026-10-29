"""Unit tests for pipeline schema validation decorator."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validator import validate_pipeline_schema


@validate_pipeline_schema(required_columns=["a", "b"], non_nullable_columns=["a"])
def _valid_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1], "b": [2]})


@validate_pipeline_schema(required_columns=["a", "b"], non_nullable_columns=["a"])
def _missing_column_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1]})


@validate_pipeline_schema(required_columns=["a", "b"], non_nullable_columns=["a"])
def _nan_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [None], "b": [2]})


def test_validate_pipeline_schema_accepts_valid_dataframe() -> None:
    """Decorator should pass valid schema and non-null constraints."""
    df = _valid_df()
    assert list(df.columns) == ["a", "b"]


def test_validate_pipeline_schema_rejects_missing_columns() -> None:
    """Decorator should reject dataframe with missing required columns."""
    with pytest.raises(ValueError):
        _missing_column_df()


def test_validate_pipeline_schema_rejects_nulls() -> None:
    """Decorator should reject NaN in non-nullable columns."""
    with pytest.raises(ValueError):
        _nan_df()
