"""Rossmann data preprocessor.

Handles missing values, type conversions, and normalization
before feature engineering.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.data.schemas import PreprocessorOutputSchema
from src.data.validator import validate_pipeline_schema


class RossmannPreprocessor:
    """Cleans and encodes raw Rossmann data.

    Attributes:
        label_encoders: Dict mapping column names to fitted LabelEncoders.

    Args:
        categorical_cols: List of column names to label-encode.

    Example:
        >>> pre = RossmannPreprocessor()
        >>> clean_df = pre.fit_transform(raw_df)
        >>> test_clean = pre.transform(test_df)
    """

    CATEGORICAL_COLS = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]

    def __init__(self, categorical_cols: list[str] | None = None) -> None:
        """Initialize preprocessor and categorical encoder store.

        Args:
            categorical_cols: Optional override for categorical columns.
        """
        self.categorical_cols = categorical_cols or self.CATEGORICAL_COLS
        self.label_encoders: dict[str, LabelEncoder] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit encoders and transform the training dataframe.

        Args:
            df: Raw training DataFrame.

        Returns:
            Cleaned and encoded DataFrame.
        """
        clean = self._normalize_column_names(df.copy())
        clean = self._fill_missing(clean)
        clean = self._encode_categoricals(clean, fit=True)
        return self.validate_output_schema(clean)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the test dataframe using fitted encoders.

        Args:
            df: Raw test DataFrame.

        Returns:
            Cleaned and encoded DataFrame.

        Raises:
            RuntimeError: If fit_transform has not been called first.
        """
        if not self.label_encoders:
            raise RuntimeError("Call fit_transform before transform.")
        clean = self._normalize_column_names(df.copy())
        clean = self._fill_missing(clean)
        clean = self._encode_categoricals(clean, fit=False)
        return self.validate_output_schema(clean)

    @validate_pipeline_schema(
        required_columns=cast(list[str], PreprocessorOutputSchema().required_columns),
        non_nullable_columns=cast(list[str], PreprocessorOutputSchema().non_nullable_columns),
    )
    def validate_output_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate preprocessing stage output schema."""
        return df

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize incoming column names to canonical Rossmann form.

        Uses in-place rename semantics instead of creating duplicate alias columns.
        """
        rename_candidates = {
            "store": "Store",
            "date": "Date",
            "sales": "Sales",
            "customers": "Customers",
            "open": "Open",
            "promo": "Promo",
            "promo2": "Promo2",
            "state_holiday": "StateHoliday",
            "school_holiday": "SchoolHoliday",
            "store_type": "StoreType",
            "assortment": "Assortment",
            "competition_distance": "CompetitionDistance",
            "competition_open_since_month": "CompetitionOpenSinceMonth",
            "competition_open_since_year": "CompetitionOpenSinceYear",
            "promo2_since_week": "Promo2SinceWeek",
            "promo2_since_year": "Promo2SinceYear",
            "promo_interval": "PromoInterval",
        }

        rename_map = {
            source: target
            for source, target in rename_candidates.items()
            if source in df.columns and target not in df.columns
        }
        if rename_map:
            df = df.rename(columns=rename_map)

        redundant_aliases = [
            source
            for source, target in rename_candidates.items()
            if source in df.columns and target in df.columns and source != target
        ]
        if redundant_aliases:
            df = df.drop(columns=redundant_aliases)

        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values for known Rossmann columns.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values imputed.
        """
        fill_zero_cols = [
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2SinceWeek",
            "Promo2SinceYear",
        ]
        for col in fill_zero_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        if "CompetitionDistance" in df.columns:
            median = df["CompetitionDistance"].median()
            df["CompetitionDistance"] = df["CompetitionDistance"].fillna(median)

        if "PromoInterval" in df.columns:
            df["PromoInterval"] = df["PromoInterval"].fillna("None")

        if "StateHoliday" in df.columns:
            df["StateHoliday"] = df["StateHoliday"].astype(str).replace("0", "None")

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Label-encode categorical columns.

        Args:
            df: Input DataFrame.
            fit: Whether to fit encoders or use existing encoders.

        Returns:
            Encoded DataFrame.
        """
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if fit:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.label_encoders[col] = encoder
            else:
                encoder = self.label_encoders[col]
                known_classes = set(encoder.classes_)
                fallback = encoder.classes_[0]
                df[col] = (
                    df[col]
                    .astype(str)
                    .apply(lambda value: value if value in known_classes else fallback)
                )
                df[col] = encoder.transform(df[col])
        return df
