"""Tests for pipeline schema definition models."""

from __future__ import annotations

from src.data.schemas import (
    FeatureEngineerOutputSchema,
    LoaderOutputSchema,
    PreprocessorOutputSchema,
)


def test_loader_schema_contains_core_columns() -> None:
    schema = LoaderOutputSchema()
    assert "Store" in schema.required_columns
    assert "Date" in schema.required_columns
    assert "Sales" in schema.required_columns


def test_preprocessor_schema_contains_encoded_categorical_columns() -> None:
    schema = PreprocessorOutputSchema()
    assert "StoreType" in schema.required_columns
    assert "Assortment" in schema.required_columns


def test_feature_engineer_schema_targets_snake_case_sales() -> None:
    schema = FeatureEngineerOutputSchema()
    assert "sales" in schema.required_columns
    assert "time_idx" in schema.required_columns
