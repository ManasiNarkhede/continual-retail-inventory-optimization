"""Feature engineering for the Rossmann dataset.

Creates temporal, lag, rolling, EWMA, cyclic, and holiday features required
by the Temporal Fusion Transformer.  Every derived feature is computed with
a strict causal guard (``shift >= 1``) so that **no future information leaks
into the current row**.

Mathematical notes
------------------
- Cyclic encoding:  ``sin(2π · x / T)``, ``cos(2π · x / T)`` maps an
  integer cycle of period *T* onto the unit circle, giving the model a
  continuous distance metric across the cycle boundary.
- EWMA:  ``EWMA_α[t] = α · y[t-1] + (1 - α) · EWMA_α[t-1]`` with
  ``span = (2/α) - 1``.  The one-step shift ensures strict causality.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.data.schemas import FeatureEngineerOutputSchema
from src.data.validator import validate_pipeline_schema


class FeatureEngineer:
    """Generates features required by the Temporal Fusion Transformer.

    Features created:
        - Calendar: year, month, week_of_year, day_of_week, day_of_month,
                    is_weekend, quarter
        - Cyclic:   dow_sin, dow_cos, month_sin, month_cos
        - Lag:      sales_7d_lag, sales_14d_lag, sales_28d_lag
        - Rolling:  rolling_mean_7d, rolling_mean_28d, rolling_std_7d
        - EWMA:     ewma_sales_7d, ewma_sales_28d
        - Holiday:  days_to_holiday, days_after_holiday
        - Competition: competition_open_since_months
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps.

        Args:
            df: Preprocessed DataFrame with a Date column.

        Returns:
            DataFrame with engineered features.
        """
        data = df.copy()
        data = self._normalize_column_names(data)
        data = data.sort_values(["Store", "Date"]).reset_index(drop=True)
        data = self._calendar_features(data)
        data = self._cyclic_features(data)
        data = self._lag_features(data)
        data = self._rolling_features(data)
        data = self._ewma_features(data)
        data = self._holiday_proximity(data)
        data = self._competition_features(data)
        data = self._time_index(data)
        return self.validate_output_schema(data)

    # ------------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------------

    def _calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create date-based calendar features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with calendar columns.
        """
        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
        df["day_of_month"] = df["Date"].dt.day
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["quarter"] = df["Date"].dt.quarter
        return df

    # ------------------------------------------------------------------
    # Cyclic encoding
    # ------------------------------------------------------------------

    def _cyclic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map day-of-week and month onto the unit circle.

        ``sin`` and ``cos`` transforms preserve the cyclical distance between
        Monday→Sunday and December→January.

        Args:
            df: Input DataFrame (must have ``day_of_week`` and ``month``).

        Returns:
            DataFrame with ``dow_sin``, ``dow_cos``, ``month_sin``, ``month_cos``.
        """
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
        return df

    # ------------------------------------------------------------------
    # Lag features (strictly causal — shift >= lag)
    # ------------------------------------------------------------------

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged sales features per store.

        Each lag uses ``shift(lag)`` so the value at row *t* is the sales
        from *t − lag*, with no access to future data.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with lag features.
        """
        for lag in [7, 14, 28]:
            df[f"sales_{lag}d_lag"] = df.groupby("Store")["sales"].shift(lag)
        return df

    # ------------------------------------------------------------------
    # Rolling statistics (shift *before* window to prevent leakage)
    # ------------------------------------------------------------------

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistics over causally shifted sales.

        The ``shift(1)`` is applied **before** the rolling window so that
        the window at time *t* only covers values ``[t-window, t-1]``.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with rolling features.
        """
        shifted = df.groupby("Store")["sales"].shift(1)
        grouped_shifted = shifted.groupby(df["Store"])
        df["rolling_mean_7d"] = (
            grouped_shifted.rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df["rolling_mean_28d"] = (
            grouped_shifted.rolling(28, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df["rolling_std_7d"] = (
            grouped_shifted.rolling(7, min_periods=1)
            .std()
            .fillna(0)
            .reset_index(level=0, drop=True)
        )
        return df

    # ------------------------------------------------------------------
    # EWMA features (exponentially weighted moving averages)
    # ------------------------------------------------------------------

    def _ewma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute exponentially weighted moving averages for trend capture.

        EWMA reacts faster to sudden trend shifts than a simple rolling
        mean, making it valuable for detecting demand surges.

        The ``shift(1)`` before ``ewm()`` ensures strict causality —
        the EWMA at time *t* only uses data up to *t − 1*.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with ``ewma_sales_7d`` and ``ewma_sales_28d``.
        """
        for span in [7, 28]:
            df[f"ewma_sales_{span}d"] = df.groupby("Store")["sales"].transform(
                lambda s: s.shift(1).ewm(span=span, min_periods=1).mean()
            )
        return df

    # ------------------------------------------------------------------
    # Holiday proximity (vectorized with np.searchsorted)
    # ------------------------------------------------------------------

    def _holiday_proximity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute distance to nearest holiday before and after each date.

        Uses ``np.searchsorted`` for O(n log h) vectorized lookup instead
        of the previous row-by-row ``apply`` approach.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with holiday proximity columns.
        """
        holiday_col = "StateHoliday" if "StateHoliday" in df.columns else "state_holiday"
        if holiday_col not in df.columns:
            df["days_to_holiday"] = 0
            df["days_after_holiday"] = 0
            return df

        holiday_values = df[holiday_col]
        if pd.api.types.is_numeric_dtype(holiday_values):
            holiday_mask = holiday_values.astype(float) != 0.0
        else:
            normalized = holiday_values.astype(str).str.lower()
            holiday_mask = normalized.isin(["1", "a", "b", "c", "true"])

        holiday_dates = np.sort(df.loc[holiday_mask, "Date"].unique())
        if len(holiday_dates) == 0:
            df["days_to_holiday"] = 0
            df["days_after_holiday"] = 0
            return df

        # Convert to int64 epoch for searchsorted
        holiday_epoch = holiday_dates.astype(np.int64)
        date_epoch = df["Date"].values.astype(np.int64)

        # Days to next holiday
        right_idx = np.searchsorted(holiday_epoch, date_epoch, side="left")
        right_idx = np.clip(right_idx, 0, len(holiday_epoch) - 1)
        days_to = (holiday_epoch[right_idx] - date_epoch) // (10**9 * 86400)
        # Where the found holiday is in the past, cap at 30
        days_to = np.where(days_to < 0, 30, days_to)
        df["days_to_holiday"] = np.clip(days_to.astype(int), 0, 30)

        # Days after last holiday
        left_idx = np.searchsorted(holiday_epoch, date_epoch, side="right") - 1
        left_idx = np.clip(left_idx, 0, len(holiday_epoch) - 1)
        days_after = (date_epoch - holiday_epoch[left_idx]) // (10**9 * 86400)
        days_after = np.where(days_after < 0, 30, days_after)
        df["days_after_holiday"] = np.clip(days_after.astype(int), 0, 30)

        return df

    # ------------------------------------------------------------------
    # Competition features
    # ------------------------------------------------------------------

    def _competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute months since competition opened.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with competition_open_since_months.
        """
        if (
            "competition_open_since_year" in df.columns
            and "competition_open_since_month" in df.columns
        ):
            months_open = (df["year"] - df["competition_open_since_year"]) * 12 + (
                df["month"] - df["competition_open_since_month"]
            )
            df["competition_open_since_months"] = months_open.clip(lower=0).fillna(0)
        else:
            df["competition_open_since_months"] = 0
        return df

    # ------------------------------------------------------------------
    # Column normalisation
    # ------------------------------------------------------------------

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize selected Rossmann columns to snake_case names.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with canonical snake_case columns.
        """
        rename_candidates = {
            "Store": "Store",
            "Date": "Date",
            "Sales": "sales",
            "Customers": "customers",
            "Open": "open",
            "Promo": "promo",
            "Promo2": "promo2",
            "SchoolHoliday": "school_holiday",
            "StoreType": "store_type",
            "Assortment": "assortment",
            "StateHoliday": "state_holiday",
            "CompetitionDistance": "competition_distance",
            "Promo2SinceWeek": "promo2_since_weeks",
            "Promo2SinceYear": "promo2_since_year",
            "CompetitionOpenSinceMonth": "competition_open_since_month",
            "CompetitionOpenSinceYear": "competition_open_since_year",
            "PromoInterval": "promo_interval",
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

    # ------------------------------------------------------------------
    # Time index
    # ------------------------------------------------------------------

    def _time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create integer time index per store.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with time_idx.
        """
        df["time_idx"] = df.groupby("Store").cumcount()
        return df

    @validate_pipeline_schema(
        required_columns=cast(list[str], FeatureEngineerOutputSchema().required_columns),
        non_nullable_columns=cast(list[str], FeatureEngineerOutputSchema().non_nullable_columns),
    )
    def validate_output_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate feature-engineering output schema."""
        return df
