"""Unit tests for RossmannPreprocessor."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocessor import RossmannPreprocessor


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample dataframe with missing and categorical values.

    Returns:
        Input dataframe fixture.
    """
    return pd.DataFrame(
        {
            "Store": [1, 1, 1],
            "Date": pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03"]),
            "Sales": [100, 120, 140],
            "StoreType": ["a", "b", "a"],
            "Assortment": ["a", "b", "c"],
            "StateHoliday": ["0", "a", "0"],
            "PromoInterval": [None, "Jan,Apr,Jul,Oct", None],
            "CompetitionDistance": [100.0, None, 200.0],
            "CompetitionOpenSinceMonth": [1, None, 5],
            "CompetitionOpenSinceYear": [2013, None, 2015],
            "Promo2SinceWeek": [1, None, 2],
            "Promo2SinceYear": [2010, None, 2011],
        }
    )


def test_fit_transform_creates_encoders(sample_df: pd.DataFrame) -> None:
    """Verify fit_transform produces encoded categorical values.

    Args:
        sample_df: Preprocessing fixture.
    """
    preprocessor = RossmannPreprocessor()
    transformed = preprocessor.fit_transform(sample_df)
    assert preprocessor.label_encoders
    for col in preprocessor.categorical_cols:
        if col in transformed.columns:
            assert transformed[col].dtype.kind in {"i", "u"}


def test_transform_requires_fit(sample_df: pd.DataFrame) -> None:
    """Ensure transform raises before fit_transform.

    Args:
        sample_df: Preprocessing fixture.
    """
    preprocessor = RossmannPreprocessor()
    with pytest.raises(RuntimeError):
        preprocessor.transform(sample_df)


def test_transform_handles_unknown_categories(sample_df: pd.DataFrame) -> None:
    """Ensure unseen categories are mapped to known fallback class.

    Args:
        sample_df: Preprocessing fixture.
    """
    preprocessor = RossmannPreprocessor()
    preprocessor.fit_transform(sample_df)
    test_df = sample_df.copy()
    test_df.loc[0, "StoreType"] = "unknown"
    transformed = preprocessor.transform(test_df)
    assert transformed["StoreType"].notna().all()


def test_preprocessor_normalizes_snake_case_columns_in_place() -> None:
    """Snake_case aliases should be renamed to canonical Rossmann columns without duplicates."""
    df = pd.DataFrame(
        {
            "store": [1, 1],
            "date": pd.to_datetime(["2015-01-01", "2015-01-02"]),
            "sales": [100.0, 120.0],
            "store_type": ["a", "b"],
            "assortment": ["a", "a"],
            "state_holiday": ["0", "0"],
            "promo_interval": [None, None],
        }
    )

    preprocessor = RossmannPreprocessor()
    transformed = preprocessor.fit_transform(df)

    assert "Store" in transformed.columns
    assert "Date" in transformed.columns
    assert "Sales" in transformed.columns
    assert "store" not in transformed.columns
    assert "date" not in transformed.columns
    assert "sales" not in transformed.columns


def test_preprocessor_drops_duplicate_aliases_when_both_forms_exist() -> None:
    """When canonical and alias columns both exist, alias columns should be dropped."""
    df = pd.DataFrame(
        {
            "Store": [1],
            "store": [1],
            "Date": pd.to_datetime(["2015-01-01"]),
            "date": pd.to_datetime(["2015-01-01"]),
            "Sales": [100.0],
            "sales": [100.0],
            "StoreType": ["a"],
            "Assortment": ["a"],
            "StateHoliday": ["0"],
            "PromoInterval": [None],
        }
    )

    preprocessor = RossmannPreprocessor()
    transformed = preprocessor.fit_transform(df)

    assert "Store" in transformed.columns
    assert "Date" in transformed.columns
    assert "Sales" in transformed.columns
    assert "store" not in transformed.columns
    assert "date" not in transformed.columns
    assert "sales" not in transformed.columns
