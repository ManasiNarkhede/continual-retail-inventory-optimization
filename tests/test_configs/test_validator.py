"""Tests for global config validator."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from configs.validator import validate_configs


def test_validate_configs_passes_with_repo_configs() -> None:
    """Repository defaults should satisfy cross-config constraints."""
    validate_configs("configs/tft_config.yaml", "configs/rl_config.yaml")


def test_validate_configs_fails_on_horizon_conflict(tmp_path: Path) -> None:
    """Validator should reject RL horizon larger than TFT prediction length."""
    tft_cfg = {
        "tft": {
            "quantiles": [0.1, 0.5, 0.9],
            "max_encoder_length": 30,
            "max_prediction_length": 7,
            "target": "sales",
            "static_categoricals": ["store"],
            "static_reals": [],
            "time_varying_known_reals": ["day_of_week"],
            "time_varying_unknown_reals": ["sales"],
        }
    }
    rl_cfg = {
        "environment": {"forecast_horizon": 14, "episode_length": 30},
        "agent": {"algorithm": "PPO", "total_timesteps": 1000},
    }

    tft_path = tmp_path / "tft.yaml"
    rl_path = tmp_path / "rl.yaml"
    tft_path.write_text(yaml.safe_dump(tft_cfg), encoding="utf-8")
    rl_path.write_text(yaml.safe_dump(rl_cfg), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_configs(str(tft_path), str(rl_path))


def test_validate_configs_fails_on_api_warmup_horizon_conflict(tmp_path: Path) -> None:
    """Validator should reject API warmup horizon above TFT prediction length."""
    tft_cfg = {
        "tft": {
            "quantiles": [0.1, 0.5, 0.9],
            "max_encoder_length": 30,
            "max_prediction_length": 7,
            "target": "sales",
        }
    }
    rl_cfg = {
        "environment": {
            "forecast_horizon": 7,
            "episode_length": 30,
            "risk_aversion": 0.5,
        },
        "agent": {"algorithm": "PPO", "total_timesteps": 1000},
    }
    cl_cfg = {
        "continual_learning": {
            "strategy": "Replay",
            "train_mb_size": 16,
            "train_epochs": 1,
            "eval_mb_size": 16,
        }
    }
    api_cfg = {
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "reload": True,
            "log_level": "info",
            "allowed_origins": ["http://localhost:5173"],
            "title": "Retail Inventory Optimization API",
            "version": "0.1.0",
        },
        "models": {
            "tft_checkpoint": "checkpoints/tft/tft_latest.pt",
            "processed_data": "data/processed/tft_reference.parquet",
            "runtime_config": "data/processed/tft_runtime_config.yaml",
            "rl_checkpoint": "checkpoints/rl/inventory_agent.zip",
            "cl_metrics": "data/processed/cl_metrics_latest.json",
            "cache_ttl_seconds": 900,
            "forecast_interval_scale": 1.0,
            "forecast_warmup_top_n": 10,
            "forecast_warmup_horizon": 14,
            "model_version": "runtime",
        },
    }

    tft_path = tmp_path / "tft.yaml"
    rl_path = tmp_path / "rl.yaml"
    cl_path = tmp_path / "cl.yaml"
    api_path = tmp_path / "api.yaml"
    tft_path.write_text(yaml.safe_dump(tft_cfg), encoding="utf-8")
    rl_path.write_text(yaml.safe_dump(rl_cfg), encoding="utf-8")
    cl_path.write_text(yaml.safe_dump(cl_cfg), encoding="utf-8")
    api_path.write_text(yaml.safe_dump(api_cfg), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_configs(
            str(tft_path),
            str(rl_path),
            str(cl_path),
            str(api_path),
        )


def test_validate_configs_fails_on_risk_aversion_out_of_range(tmp_path: Path) -> None:
    """Validator should reject rl risk_aversion outside [0, 1]."""
    tft_cfg = {
        "tft": {
            "quantiles": [0.1, 0.5, 0.9],
            "max_encoder_length": 30,
            "max_prediction_length": 7,
            "target": "sales",
        }
    }
    rl_cfg = {
        "environment": {
            "forecast_horizon": 7,
            "episode_length": 30,
            "risk_aversion": 1.8,
        },
        "agent": {"algorithm": "PPO", "total_timesteps": 1000},
    }
    cl_cfg = {
        "continual_learning": {
            "strategy": "Replay",
            "train_mb_size": 16,
            "train_epochs": 1,
            "eval_mb_size": 16,
        }
    }
    api_cfg = {
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "reload": True,
            "log_level": "info",
            "allowed_origins": ["http://localhost:5173"],
            "title": "Retail Inventory Optimization API",
            "version": "0.1.0",
        },
        "models": {
            "tft_checkpoint": "checkpoints/tft/tft_latest.pt",
            "processed_data": "data/processed/tft_reference.parquet",
            "runtime_config": "data/processed/tft_runtime_config.yaml",
            "rl_checkpoint": "checkpoints/rl/inventory_agent.zip",
            "cl_metrics": "data/processed/cl_metrics_latest.json",
            "cache_ttl_seconds": 900,
            "forecast_interval_scale": 1.0,
            "forecast_warmup_top_n": 10,
            "forecast_warmup_horizon": 7,
            "model_version": "runtime",
        },
    }

    tft_path = tmp_path / "tft.yaml"
    rl_path = tmp_path / "rl.yaml"
    cl_path = tmp_path / "cl.yaml"
    api_path = tmp_path / "api.yaml"
    tft_path.write_text(yaml.safe_dump(tft_cfg), encoding="utf-8")
    rl_path.write_text(yaml.safe_dump(rl_cfg), encoding="utf-8")
    cl_path.write_text(yaml.safe_dump(cl_cfg), encoding="utf-8")
    api_path.write_text(yaml.safe_dump(api_cfg), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_configs(
            str(tft_path),
            str(rl_path),
            str(cl_path),
            str(api_path),
        )
