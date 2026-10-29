"""Schema definitions for dataframe pipeline stages."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoaderOutputSchema(BaseModel):
    """Expected schema for loader train dataframe output."""

    required_columns: list[str] = Field(default_factory=lambda: ["Store", "Date", "Sales", "Open"])
    non_nullable_columns: list[str] = Field(default_factory=lambda: ["Store", "Date", "Sales"])


class PreprocessorOutputSchema(BaseModel):
    """Expected schema after preprocessing and categorical encoding."""

    required_columns: list[str] = Field(
        default_factory=lambda: ["Store", "Date", "Sales", "StoreType", "Assortment"]
    )
    non_nullable_columns: list[str] = Field(default_factory=lambda: ["Store", "Date", "Sales"])


class FeatureEngineerOutputSchema(BaseModel):
    """Expected schema after feature engineering."""

    required_columns: list[str] = Field(
        default_factory=lambda: [
            "Store",
            "Date",
            "sales",
            "time_idx",
            "rolling_mean_7d",
            "rolling_std_7d",
        ]
    )
    non_nullable_columns: list[str] = Field(
        default_factory=lambda: ["Store", "Date", "sales", "time_idx"]
    )
