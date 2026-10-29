"""Tests for JSON-schema-backed YAML config validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from configs.validator import validate_config_schema


def test_validate_config_schema_accepts_valid_tft_config() -> None:
    """Repository TFT config should satisfy its JSON schema."""
    validate_config_schema("configs/tft_config.yaml", "configs/schemas/tft_schema.json")


def test_validate_config_schema_rejects_invalid_tft_config(tmp_path: Path) -> None:
    """Missing required TFT keys should fail schema validation."""
    invalid = {"tft": {"quantiles": [0.1, 0.5, 0.9]}}
    cfg = tmp_path / "bad_tft.yaml"
    cfg.write_text(yaml.safe_dump(invalid), encoding="utf-8")

    with pytest.raises(Exception):
        validate_config_schema(str(cfg), "configs/schemas/tft_schema.json")


def test_validate_config_schema_accepts_valid_api_config() -> None:
    """Repository API config should satisfy its JSON schema."""
    validate_config_schema("configs/api_config.yaml", "configs/schemas/api_schema.json")


def test_validate_config_schema_accepts_valid_rl_config() -> None:
    """Repository RL config should satisfy its JSON schema."""
    validate_config_schema("configs/rl_config.yaml", "configs/schemas/rl_schema.json")


def test_validate_config_schema_accepts_valid_cl_config() -> None:
    """Repository CL config should satisfy its JSON schema."""
    validate_config_schema("configs/cl_config.yaml", "configs/schemas/cl_schema.json")
