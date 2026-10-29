"""Unit tests for FeatureEngineer."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a minimal Rossmann-like dataframe for tests.

    Returns:
        Sample DataFrame.
    """
    return pd.DataFrame(
        {
            "Store": [1, 1, 1, 1, 2, 2, 2, 2],
            "Date": pd.date_range("2015-01-01", periods=8),
            "Sales": [5000, 4800, 5200, 5100, 3000, 2900, 3100, 3050],
            "StateHoliday": [0, 0, 0, 0, 0, 0, 0, 0],
            "CompetitionDistance": [500, 500, 500, 500, 1000, 1000, 1000, 1000],
            "CompetitionOpenSinceMonth": [1, 1, 1, 1, 3, 3, 3, 3],
            "CompetitionOpenSinceYear": [2014, 2014, 2014, 2014, 2013, 2013, 2013, 2013],
        }
    )


def test_calendar_features_created(sample_df: pd.DataFrame) -> None:
    """Ensure calendar columns are produced.

    Args:
        sample_df: Test fixture dataframe.
    """
    result = FeatureEngineer().transform(sample_df)
    for column in ["year", "month", "week_of_year", "day_of_week", "is_weekend", "quarter"]:
        assert column in result.columns


def test_time_idx_monotonic_per_store(sample_df: pd.DataFrame) -> None:
    """Ensure time_idx increases within each store group.

    Args:
        sample_df: Test fixture dataframe.
    """
    result = FeatureEngineer().transform(sample_df)
    for _, group in result.groupby("Store"):
        assert group["time_idx"].is_monotonic_increasing


def test_rolling_mean_no_lookahead(sample_df: pd.DataFrame) -> None:
    """Ensure rolling mean uses past-only values.

    Args:
        sample_df: Test fixture dataframe.
    """
    result = FeatureEngineer().transform(sample_df)
    store1 = result[result["Store"] == 1].reset_index(drop=True)
    first_value = store1["rolling_mean_7d"].iloc[0]
    assert first_value != store1["sales"].iloc[0] or pd.isna(first_value)


def test_competition_feature_non_negative(sample_df: pd.DataFrame) -> None:
    """Ensure competition_open_since_months is non-negative.

    Args:
        sample_df: Test fixture dataframe.
    """
    result = FeatureEngineer().transform(sample_df)
    assert (result["competition_open_since_months"] >= 0).all()


def test_temporal_leakage_guard_shift_one() -> None:
    """Deliberately construct leakage-prone data and verify shift(1) protection."""
    df = pd.DataFrame(
        {
            "Store": [1, 1, 1, 1],
            "Date": pd.date_range("2015-01-01", periods=4),
            "Sales": [100.0, 9999.0, 100.0, 100.0],
            "StateHoliday": [0, 0, 0, 0],
            "CompetitionDistance": [200.0, 200.0, 200.0, 200.0],
            "CompetitionOpenSinceMonth": [1, 1, 1, 1],
            "CompetitionOpenSinceYear": [2014, 2014, 2014, 2014],
        }
    )
    result = FeatureEngineer().transform(df)

    # At t=1, rolling_mean_7d must only see t=0 due to shift(1).
    # If leakage existed, it would include the 9999.0 current value.
    t1 = result.iloc[1]
    assert float(t1["rolling_mean_7d"]) == pytest.approx(100.0)


def test_normalize_column_names_preserves_store_and_adds_sales_alias(
    sample_df: pd.DataFrame,
) -> None:
    """Feature engineering should keep Store and expose snake_case sales."""
    result = FeatureEngineer().transform(sample_df)
    assert "Store" in result.columns
    assert "sales" in result.columns
    assert "Sales" not in result.columns


def test_feature_engineer_drops_uppercase_aliases_when_lowercase_present(
    sample_df: pd.DataFrame,
) -> None:
    """Feature output should keep normalized snake_case schema for TFT."""
    result = FeatureEngineer().transform(sample_df)
    assert "sales" in result.columns
    assert "Sales" not in result.columns
    assert "competition_distance" in result.columns
