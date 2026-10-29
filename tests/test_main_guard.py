"""Tests for production guard environment and readiness behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import main as main_module


def test_production_guard_reports_missing_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard should flag missing required environment variables."""
    for key in ("API_HOST", "API_PORT", "ENV", "MLFLOW_TRACKING_URI"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PRODUCTION_GUARD_ALLOW_QUIET_TRAIN", "0")

    report = main_module.run_production_guard()

    assert report["env_ok"] is False
    env_errors = [message for message in report["errors"] if "Missing env var" in message]
    assert len(env_errors) == 4


def test_production_guard_fails_when_configs_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard should fail fast when required config files are missing."""
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "8000")
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    monkeypatch.setenv("PRODUCTION_GUARD_ALLOW_QUIET_TRAIN", "0")
    monkeypatch.setattr(main_module, "_REQUIRED_CONFIGS", [Path("configs/definitely_missing.yaml")])

    report = main_module.run_production_guard()

    assert report["configs_ok"] is False
    assert any("Missing config files" in message for message in report["errors"])
