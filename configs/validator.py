"""Global configuration validation helpers."""

from __future__ import annotations

from pathlib import Path
import json
from functools import lru_cache
from typing import Any

import jsonschema
import yaml


_KNOWN_SCHEMA_BY_CONFIG_BASENAME: dict[str, str] = {
    "tft_config.yaml": "configs/schemas/tft_schema.json",
    "rl_config.yaml": "configs/schemas/rl_schema.json",
    "cl_config.yaml": "configs/schemas/cl_schema.json",
    "api_config.yaml": "configs/schemas/api_schema.json",
}


def _mtime_ns(path: str) -> int:
    """Return file modification timestamp in ns, or -1 when missing."""
    target = Path(path)
    if not target.exists():
        return -1
    return int(target.stat().st_mtime_ns)


@lru_cache(maxsize=16)
def _validate_cross_configs_cached(
    tft_path: str,
    rl_path: str,
    cl_path: str,
    api_path: str,
    tft_mtime_ns: int,
    rl_mtime_ns: int,
    cl_mtime_ns: int,
    api_mtime_ns: int,
) -> None:
    """Run cross-config validation behind an mtime-aware cache."""
    _ = (tft_mtime_ns, rl_mtime_ns, cl_mtime_ns, api_mtime_ns)
    validate_configs(
        tft_config_path=tft_path,
        rl_config_path=rl_path,
        cl_config_path=cl_path,
        api_config_path=api_path,
    )


def _infer_schema_path(config_path: str) -> str | None:
    """Infer schema path for known config filenames."""
    basename = Path(config_path).name
    return _KNOWN_SCHEMA_BY_CONFIG_BASENAME.get(basename)


@lru_cache(maxsize=16)
def _load_schema(schema_path: str) -> dict[str, Any]:
    """Load JSON schema payload with caching."""
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def load_yaml_config(
    config_path: str,
    *,
    validate_schema: bool = True,
    validate_cross_config: bool | None = None,
    cross_config_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load YAML config and enforce mapping payload shape.

    Args:
        config_path: YAML configuration path.
        validate_schema: If ``True``, validate known config files against
            their JSON schema at load time.
        validate_cross_config: If ``True``, run cross-config compatibility
            checks using ``validate_configs``. When ``None`` (default), this
            is enabled automatically for known repo configs.
        cross_config_paths: Optional override map for cross-config paths.

    Returns:
        Parsed YAML mapping.
    """
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config payload in '{config_path}' must be a YAML mapping.")

    if validate_schema:
        schema_path = _infer_schema_path(config_path)
        if schema_path is not None and Path(schema_path).exists():
            schema_payload = _load_schema(schema_path)
            jsonschema.validate(instance=payload, schema=schema_payload)

    basename = Path(config_path).name
    repo_default_path = (Path("configs") / basename).resolve()
    is_repo_default_path = Path(config_path).resolve() == repo_default_path

    should_cross_validate = bool(validate_cross_config)
    if validate_cross_config is None:
        should_cross_validate = _infer_schema_path(config_path) is not None and is_repo_default_path

    if should_cross_validate:
        overrides = dict(cross_config_paths or {})
        if basename == "tft_config.yaml":
            overrides["tft"] = config_path
        elif basename == "rl_config.yaml":
            overrides["rl"] = config_path
        elif basename == "cl_config.yaml":
            overrides["cl"] = config_path
        elif basename == "api_config.yaml":
            overrides["api"] = config_path

        tft_path = overrides.get("tft", "configs/tft_config.yaml")
        rl_path = overrides.get("rl", "configs/rl_config.yaml")
        cl_path = overrides.get("cl", "configs/cl_config.yaml")
        api_path = overrides.get("api", "configs/api_config.yaml")

        _validate_cross_configs_cached(
            tft_path=tft_path,
            rl_path=rl_path,
            cl_path=cl_path,
            api_path=api_path,
            tft_mtime_ns=_mtime_ns(tft_path),
            rl_mtime_ns=_mtime_ns(rl_path),
            cl_mtime_ns=_mtime_ns(cl_path),
            api_mtime_ns=_mtime_ns(api_path),
        )

    return payload


def validate_config_schema(config_path: str, schema_path: str) -> dict[str, Any]:
    """Validate YAML config payload against JSON schema."""
    config_payload = load_yaml_config(config_path, validate_schema=False, validate_cross_config=False)
    schema_payload = _load_schema(schema_path)
    jsonschema.validate(instance=config_payload, schema=schema_payload)
    return config_payload


def validate_configs(
    tft_config_path: str = "configs/tft_config.yaml",
    rl_config_path: str = "configs/rl_config.yaml",
    cl_config_path: str = "configs/cl_config.yaml",
    api_config_path: str = "configs/api_config.yaml",
) -> None:
    """Validate cross-config compatibility before pipeline startup.

    Checks:
    - TFT quantiles include valid p10/p50/p90 ordering.
    - RL forecast_horizon is positive and compatible with TFT prediction length.
    - No conflicting quantile index mapping.
    """
    tft_payload = validate_config_schema(tft_config_path, "configs/schemas/tft_schema.json")
    rl_payload = validate_config_schema(rl_config_path, "configs/schemas/rl_schema.json")
    validate_config_schema(cl_config_path, "configs/schemas/cl_schema.json")
    api_payload = validate_config_schema(api_config_path, "configs/schemas/api_schema.json")

    tft_cfg = tft_payload.get("tft", {})
    rl_cfg = rl_payload.get("environment", {})
    api_models = api_payload.get("models", {})

    quantiles = list(tft_cfg.get("quantiles", []))
    if not quantiles:
        raise ValueError("TFT quantiles list is empty.")

    from src.models.tft_model import resolve_quantile_indices

    q_idx = resolve_quantile_indices(quantiles)
    p10 = q_idx.get("p10")
    p50 = q_idx.get("p50")
    p90 = q_idx.get("p90")
    if p10 is None or p50 is None or p90 is None:
        raise ValueError("Could not resolve p10/p50/p90 indices from TFT quantiles.")
    if not (p10 < p50 < p90):
        raise ValueError(f"Invalid quantile ordering: p10={p10}, p50={p50}, p90={p90}")

    horizon = int(rl_cfg.get("forecast_horizon", 0))
    max_prediction = int(tft_cfg.get("max_prediction_length", 0))
    if horizon <= 0:
        raise ValueError("RL forecast_horizon must be > 0.")
    if max_prediction <= 0:
        raise ValueError("TFT max_prediction_length must be > 0.")
    if horizon > max_prediction:
        raise ValueError(
            f"RL forecast_horizon ({horizon}) exceeds TFT max_prediction_length ({max_prediction})."
        )

    risk_aversion = float(rl_cfg.get("risk_aversion", 0.5))
    if not (0.0 <= risk_aversion <= 1.0):
        raise ValueError(f"RL risk_aversion must be in [0, 1], got {risk_aversion}.")

    cache_ttl = float(api_models.get("cache_ttl_seconds", 0.0))
    if cache_ttl <= 0.0:
        raise ValueError("API cache_ttl_seconds must be > 0.")

    forecast_interval_scale = float(api_models.get("forecast_interval_scale", 1.0))
    if forecast_interval_scale < 1.0:
        raise ValueError("API forecast_interval_scale must be >= 1.0.")

    warmup_top_n = int(api_models.get("forecast_warmup_top_n", 0))
    if warmup_top_n < 0:
        raise ValueError("API forecast_warmup_top_n must be >= 0.")

    warmup_horizon = int(api_models.get("forecast_warmup_horizon", 0))
    if warmup_horizon <= 0:
        raise ValueError("API forecast_warmup_horizon must be > 0.")
    if warmup_horizon > max_prediction:
        raise ValueError(
            "API forecast_warmup_horizon "
            f"({warmup_horizon}) exceeds TFT max_prediction_length ({max_prediction})."
        )
