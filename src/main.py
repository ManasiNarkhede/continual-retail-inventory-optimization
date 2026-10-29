"""Production entrypoint with startup guard and auto-recovery.

This module wraps the API app startup with artifact and environment checks.
It can trigger lightweight recovery training when artifacts are missing so
the serving process starts in a resilient, headless mode.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from configs.validator import load_yaml_config, validate_config_schema
from src.api.model_registry import recover_tft_artifacts

logger = logging.getLogger(__name__)

_REQUIRED_CONFIGS = [
    Path("configs/api_config.yaml"),
    Path("configs/tft_config.yaml"),
    Path("configs/rl_config.yaml"),
]
_REQUIRED_ENV_VARS = ["API_HOST", "API_PORT", "ENV", "MLFLOW_TRACKING_URI"]


def _load_api_models_config(path: str = "configs/api_config.yaml") -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    validate_config_schema(path, "configs/schemas/api_schema.json")
    payload = load_yaml_config(path)
    return payload.get("models", {}) if isinstance(payload, dict) else {}


def _raw_data_available(data_dir: str = "data/raw") -> bool:
    root = Path(data_dir)
    return all((root / name).exists() for name in ("train.csv", "test.csv", "store.csv"))


def _tft_bundle_ready(checkpoint_path: Path) -> bool:
    base = checkpoint_path.with_suffix("") if checkpoint_path.suffix == ".pt" else checkpoint_path
    return base.with_suffix(".pt").exists() and base.with_suffix(".meta.json").exists()


def _rl_agent_ready(rl_checkpoint_path: Path) -> bool:
    return rl_checkpoint_path.exists()


def _run_quiet_train_tft() -> bool:
    cmd = [
        sys.executable,
        "scripts/train_tft.py",
        "--config",
        "configs/tft_config.yaml",
        "--override-max-epochs",
        "1",
        "--max-rows-per-store",
        "128",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Quiet TFT training failed: %s", (result.stderr or result.stdout)[-400:])
        return False
    return True


def _run_quiet_train_rl() -> bool:
    cmd = [
        sys.executable,
        "scripts/train_rl.py",
        "--config",
        "configs/rl_config.yaml",
        "--total-timesteps",
        "500",
        "--max-levels",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Quiet RL training failed: %s", (result.stderr or result.stdout)[-400:])
        return False
    return True


def run_production_guard() -> dict[str, Any]:
    """Run startup guard checks and optional lightweight recovery."""
    report: dict[str, Any] = {
        "ready": False,
        "env_ok": False,
        "configs_ok": False,
        "tft_bundle_ready": False,
        "rl_agent_ready": False,
        "recovery_attempted": False,
        "quiet_train_triggered": False,
        "errors": [],
    }

    for env_key in _REQUIRED_ENV_VARS:
        if not os.getenv(env_key):
            report["errors"].append(f"Missing env var: {env_key}")
    report["env_ok"] = len([err for err in report["errors"] if "env var" in err]) == 0

    missing_configs = [str(path) for path in _REQUIRED_CONFIGS if not path.exists()]
    if missing_configs:
        report["errors"].append(f"Missing config files: {', '.join(missing_configs)}")
        report["configs_ok"] = False
        return report
    report["configs_ok"] = True

    model_cfg = _load_api_models_config()
    tft_checkpoint = Path(model_cfg.get("tft_checkpoint", "checkpoints/tft/tft_latest.pt"))
    rl_checkpoint = Path(model_cfg.get("rl_checkpoint", "checkpoints/rl/inventory_agent.zip"))

    report["tft_bundle_ready"] = _tft_bundle_ready(tft_checkpoint)
    report["rl_agent_ready"] = _rl_agent_ready(rl_checkpoint)

    if not report["tft_bundle_ready"]:
        report["recovery_attempted"] = True
        report["tft_bundle_ready"] = bool(recover_tft_artifacts())

    needs_quiet_train = (not report["tft_bundle_ready"]) or (not report["rl_agent_ready"])
    allow_quiet_train = os.getenv("PRODUCTION_GUARD_ALLOW_QUIET_TRAIN", "1") == "1"
    if needs_quiet_train and allow_quiet_train and _raw_data_available():
        report["quiet_train_triggered"] = True
        if not report["tft_bundle_ready"]:
            _ = _run_quiet_train_tft()
            report["tft_bundle_ready"] = _tft_bundle_ready(tft_checkpoint)
        if not report["rl_agent_ready"]:
            _ = _run_quiet_train_rl()
            report["rl_agent_ready"] = _rl_agent_ready(rl_checkpoint)

    report["ready"] = bool(
        report["configs_ok"] and report["tft_bundle_ready"] and report["rl_agent_ready"]
    )
    if not report["ready"]:
        if not _raw_data_available():
            report["errors"].append("Raw Rossmann files missing in data/raw")
        if not report["tft_bundle_ready"]:
            report["errors"].append("TFT bundle unavailable")
        if not report["rl_agent_ready"]:
            report["errors"].append("RL checkpoint unavailable")

    return report


GUARD_REPORT = run_production_guard()
logger.info("Production guard report: %s", GUARD_REPORT)

from src.api.main import app as _api_app  # noqa: E402

app = _api_app
app.state.production_guard = GUARD_REPORT
