"""Schema validation utilities for data pipeline stages."""

from __future__ import annotations

from functools import wraps
from typing import Callable

import pandas as pd
from pydantic import BaseModel, field_validator


class PipelineSchemaSpec(BaseModel):
    """Column and nullability requirements for a dataframe stage."""

    required_columns: list[str]
    non_nullable_columns: list[str] = []

    @field_validator("required_columns", "non_nullable_columns")
    @classmethod
    def _dedupe_columns(cls, values: list[str]) -> list[str]:
        return sorted(set([str(value) for value in values if str(value).strip()]))


def validate_pipeline_schema(
    required_columns: list[str],
    non_nullable_columns: list[str] | None = None,
) -> Callable:
    """Decorator enforcing dataframe schema across pipeline boundaries."""

    spec = PipelineSchemaSpec(
        required_columns=required_columns,
        non_nullable_columns=non_nullable_columns or [],
    )

    def _decorator(func: Callable) -> Callable:
        @wraps(func)
        def _wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            if not isinstance(result, pd.DataFrame):
                return result

            missing = [col for col in spec.required_columns if col not in result.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            nullable_violations = [
                col
                for col in spec.non_nullable_columns
                if col in result.columns and bool(result[col].isna().any())
            ]
            if nullable_violations:
                raise ValueError(f"NaN detected in non-nullable columns: {nullable_violations}")
            return result

        return _wrapped

    return _decorator
