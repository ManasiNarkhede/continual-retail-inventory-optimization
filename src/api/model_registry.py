"""Singleton model registry for API inference dependencies.

The registry exposes lazy-loading getters for TFT, RL, and CL metrics.
Models are cached in-process and can be invalidated for test isolation.

MLflow integration
------------------
``fetch_latest_mlflow_metrics`` queries the MLflow tracking server for the
most recent run in a given experiment and extracts scalar metrics.

Live simulation
---------------
``run_rl_simulation_metrics`` executes a deterministic RL rollout and
computes actual service-level, fill-rate, and total-cost KPIs.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import shutil
import socket
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock, RLock
from typing import Any
from urllib.parse import urlparse

import numpy as np
import numpy.typing as npt
import pandas as pd

from configs.validator import load_yaml_config, validate_config_schema
from src.eval.demand_scenario import build_scenario_batch
from src.eval.metric_names import MetricNames, read_metric
from src.utils.json_logger import log_json

logger = logging.getLogger(__name__)

try:
    import structlog  # noqa: F401
except ImportError:  # pragma: no cover
    structlog = None


@dataclass
class RegistryState:
    """Container for globally shared model objects."""

    tft_model: Any | None = None
    tft_dataset: Any | None = None
    tft_reference_df: Any | None = None
    rl_agent: Any | None = None
    cl_metrics: dict[str, Any] | None = None
    latest_simulation: dict[str, float | str] = field(default_factory=dict)
    forecast_cache: dict[str, dict[str, Any]] = field(default_factory=dict)


class SimulationUnavailableError(RuntimeError):
    """Raised when an RL simulation is requested but no RL agent is loaded."""


_STATE = RegistryState()
_LOCK = Lock()
# Held for the whole lazy load so concurrent first requests load each model only once.
_TFT_LOAD_LOCK = RLock()
_RL_LOAD_LOCK = RLock()
_CACHE_LOADED_AT: dict[str, float | None] = {
    "tft": None,
    "rl": None,
    "cl_metrics": None,
    "simulation": None,
}
_SIM_ENV_POOL: list[Any] = []
_SIM_ENV_POOL_SIGNATURE: tuple[str, float | None, str, int, int] | None = None


def _agent_has_predict(candidate: object) -> bool:
    """Check whether loaded agent exposes predict method expected by simulation."""
    return hasattr(candidate, "predict") and callable(getattr(candidate, "predict", None))


def _extract_env(candidate: object) -> Any | None:
    """Extract underlying environment from common agent wrappers."""
    env_container = getattr(candidate, "env", None)
    if env_container is None:
        return None
    wrapped_env = getattr(env_container, "env", None)
    if wrapped_env is not None:
        return wrapped_env
    return env_container


def _registry_ttl_seconds(config: dict[str, Any] | None = None) -> float:
    """Resolve cache TTL for lazy-loaded registry artifacts."""
    cfg = config or _load_config()
    ttl = cfg.get("cache_ttl_seconds", 900)
    try:
        return max(float(ttl), 1.0)
    except (TypeError, ValueError):
        return 900.0


def _forecast_interval_scale(config: dict[str, Any] | None = None) -> float:
    """Resolve forecast interval scaling factor from API model config."""
    cfg = config or _load_config()
    raw = cfg.get("forecast_interval_scale", 1.0)
    try:
        scale = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return scale if scale >= 1.0 else 1.0


def calibrate_prediction_interval(
    p10: npt.ArrayLike,
    p50: npt.ArrayLike,
    p90: npt.ArrayLike,
    interval_scale: float | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calibrate interval width around median forecast.

    The calibration keeps the median unchanged while widening or preserving
    lower/upper spreads around ``p50``.

    Args:
        p10: Lower-quantile series.
        p50: Median-quantile series.
        p90: Upper-quantile series.
        interval_scale: Optional spread multiplier. Values below ``1.0`` are
            treated as ``1.0``.

    Returns:
        Tuple of calibrated ``(p10, p50, p90)`` arrays.
    """
    lower = np.asarray(p10, dtype=float)
    median = np.asarray(p50, dtype=float)
    upper = np.asarray(p90, dtype=float)

    scale = max(
        float(interval_scale) if interval_scale is not None else _forecast_interval_scale(), 1.0
    )
    if np.isclose(scale, 1.0):
        clipped_median = np.clip(median, a_min=0.0, a_max=None)
        clipped_lower = np.clip(np.minimum(lower, clipped_median), a_min=0.0, a_max=None)
        clipped_upper = np.maximum(upper, clipped_median)
        return clipped_lower, clipped_median, clipped_upper

    lower_gap = np.maximum(median - lower, 0.0)
    upper_gap = np.maximum(upper - median, 0.0)

    calibrated_median = np.clip(median, a_min=0.0, a_max=None)
    calibrated_lower = np.clip(calibrated_median - lower_gap * scale, a_min=0.0, a_max=None)
    calibrated_upper = np.maximum(calibrated_median + upper_gap * scale, calibrated_median)
    calibrated_lower = np.minimum(calibrated_lower, calibrated_median)
    return calibrated_lower, calibrated_median, calibrated_upper


def _normalize_risk_aversion(value: float | None) -> float | None:
    """Clamp risk-aversion value into the supported [0, 1] range."""
    if value is None:
        return None
    return float(np.clip(float(value), 0.0, 1.0))


def _cache_expired(cache_key: str, config: dict[str, Any] | None = None) -> bool:
    """Return whether a specific cache key exceeded TTL."""
    loaded_at = _CACHE_LOADED_AT.get(cache_key)
    if loaded_at is None:
        return False
    return (time.time() - loaded_at) > _registry_ttl_seconds(config=config)


def _mark_cache_loaded(cache_key: str) -> None:
    """Mark a cache key with current load timestamp."""
    _CACHE_LOADED_AT[cache_key] = time.time()


def _get_or_create_sim_env_pool(
    *,
    config_path: str,
    risk_aversion: float | None,
    reward_type: str,
    pool_size: int,
    default_days: int,
    env_class_id: int,
) -> list[Any]:
    """Get reusable simulation env pool for Monte Carlo runs."""
    global _SIM_ENV_POOL, _SIM_ENV_POOL_SIGNATURE
    signature = (
        str(config_path),
        risk_aversion,
        str(reward_type),
        int(pool_size),
        int(env_class_id),
    )
    if _SIM_ENV_POOL_SIGNATURE != signature or len(_SIM_ENV_POOL) != int(pool_size):
        from src.rl.inventory_env import RetailInventoryEnv

        _SIM_ENV_POOL = [
            RetailInventoryEnv(
                config_path=str(config_path),
                demand_series=np.ones(max(int(default_days), 1), dtype=float),
                risk_aversion=risk_aversion,
                reward_type=reward_type,
            )
            for _ in range(max(int(pool_size), 1))
        ]
        _SIM_ENV_POOL_SIGNATURE = signature
    return _SIM_ENV_POOL


def _cached_forecast_key(store_id: int, horizon: int) -> str:
    """Build cache key for forecast payload entries."""
    return f"{int(store_id)}:{int(horizon)}"


def get_cached_forecast(store_id: int, horizon: int) -> dict[str, Any] | None:
    """Read warmed forecast payload from in-memory cache."""
    key = _cached_forecast_key(store_id=store_id, horizon=horizon)
    return copy.deepcopy(_STATE.forecast_cache.get(key)) if _STATE.forecast_cache else None


def cache_forecast_payload(store_id: int, horizon: int, payload: dict[str, Any]) -> None:
    """Store forecast payload in in-memory cache."""
    cache = dict(_STATE.forecast_cache)
    cache[_cached_forecast_key(store_id=store_id, horizon=horizon)] = copy.deepcopy(payload)
    _atomic_swap(forecast_cache=cache)


def warmup_forecast_cache(top_n: int = 10, horizon: int = 14) -> int:
    """Precompute forecast payloads for top-N stores and cache them."""
    model, dataset, reference_df = get_tft_model()
    if model is None or dataset is None or reference_df is None:
        return 0

    stores: list[int] = []
    if "Store" in reference_df.columns:
        stores = sorted(reference_df["Store"].dropna().astype(int).unique().tolist())
    stores = stores[: max(int(top_n), 0)]
    if not stores:
        return 0

    q_idx = get_tft_quantile_indices()
    idx_p10 = int(q_idx.get("p10", 1))
    idx_p50 = int(q_idx.get("p50", 3))
    idx_p90 = int(q_idx.get("p90", 5))
    warmed = 0
    for store_id in stores:
        try:
            predictions, dates = model.predict_for_store(
                store_id=int(store_id),
                horizon=int(horizon),
                training_dataset=dataset,
                reference_df=reference_df,
            )
            p10_raw = predictions[:, idx_p10].detach().cpu().numpy()
            p50_raw = predictions[:, idx_p50].detach().cpu().numpy()
            p90_raw = predictions[:, idx_p90].detach().cpu().numpy()
            p10, p50, p90 = calibrate_prediction_interval(p10_raw, p50_raw, p90_raw)
            payload = {
                "store_id": int(store_id),
                "horizon": int(horizon),
                "dates": [date.strftime("%Y-%m-%d") for date in dates],
                "p10": p10.tolist(),
                "p50": p50.tolist(),
                "p90": p90.tolist(),
                "model_version": "warmup_cache",
            }
            cache_forecast_payload(store_id=int(store_id), horizon=int(horizon), payload=payload)
            warmed += 1
        except (RuntimeError, ValueError, KeyError, AttributeError):
            continue
    return int(warmed)


def _atomic_swap(**updates: Any) -> None:
    """Atomically replace registry state with updated fields."""
    global _STATE
    with _LOCK:
        _STATE = replace(_STATE, **updates)


def resolve_quantile_indices(quantiles: list[float]) -> dict[str, int]:
    """Resolve p10/p50/p90 indices from dynamic quantile config."""
    from src.models.tft_model import resolve_quantile_indices as _resolve

    result = _resolve(quantiles)
    return {str(k): int(v) for k, v in result.items()}


def _load_config(path: str = "configs/api_config.yaml") -> dict[str, Any]:
    """Load API model configuration with defaults.

    Args:
        path: Config YAML path.

    Returns:
        Flattened model configuration dictionary.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {
            "tft_checkpoint": "checkpoints/tft/tft_latest.pt",
            "processed_data": "data/processed/tft_reference.parquet",
            "runtime_config": "data/processed/tft_runtime_config.yaml",
            "rl_checkpoint": "checkpoints/rl/inventory_agent.zip",
            "cl_metrics": "data/processed/cl_metrics_latest.json",
            "forecast_interval_scale": 1.0,
        }

    validate_config_schema(path, "configs/schemas/api_schema.json")

    payload = load_yaml_config(path)
    models = payload.get("models", {})
    return {
        "tft_checkpoint": models.get("tft_checkpoint", "checkpoints/tft/tft_latest.pt"),
        "processed_data": models.get("processed_data", "data/processed/tft_reference.parquet"),
        "runtime_config": models.get("runtime_config", "data/processed/tft_runtime_config.yaml"),
        "rl_checkpoint": models.get("rl_checkpoint", "checkpoints/rl/inventory_agent.zip"),
        "cl_metrics": models.get("cl_metrics", "data/processed/cl_metrics_latest.json"),
        "cache_ttl_seconds": models.get("cache_ttl_seconds", 900),
        "forecast_warmup_top_n": models.get("forecast_warmup_top_n", 10),
        "forecast_warmup_horizon": models.get("forecast_warmup_horizon", 14),
        "forecast_interval_scale": models.get("forecast_interval_scale", 1.0),
        "model_version": models.get("model_version", "runtime"),
    }


def _latest_file(pattern: str) -> Path | None:
    """Return newest file matching glob pattern.

    Args:
        pattern: Glob pattern.

    Returns:
        Latest matching path or ``None``.
    """
    candidates = list(Path().glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _raw_rossmann_data_ready(data_dir: str = "data/raw") -> bool:
    """Check whether required Rossmann raw files are present."""
    root = Path(data_dir)
    required = [root / "train.csv", root / "test.csv", root / "store.csv"]
    return all(path.exists() for path in required)


def _recover_runtime_config(runtime_config_path: Path) -> bool:
    """Ensure runtime TFT config exists by copying base config if needed."""
    if runtime_config_path.exists():
        return True

    base_cfg = Path("configs/tft_config.yaml")
    if not base_cfg.exists():
        return False

    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base_cfg, runtime_config_path)
    return True


def _recover_reference_parquet(reference_path: Path) -> bool:
    """Rebuild feature-engineered reference parquet from raw Rossmann CSVs."""
    if reference_path.exists():
        return True
    if not _raw_rossmann_data_ready():
        return False

    try:
        from src.data.feature_engineer import FeatureEngineer
        from src.data.loader import RossmannDataLoader
        from src.data.preprocessor import RossmannPreprocessor

        loader = RossmannDataLoader("data/raw")
        train_df, _ = loader.load()
        train_df = RossmannPreprocessor().fit_transform(train_df)
        train_df = FeatureEngineer().transform(train_df)

        reference_path.parent.mkdir(parents=True, exist_ok=True)
        train_df.sort_values(["Store", "time_idx"]).to_parquet(reference_path, index=False)
        return True
    except (FileNotFoundError, PermissionError, ValueError, OSError) as exc:
        log_json(logger, logging.WARNING, "reference_parquet_recovery_failed", error=str(exc))
        return False


def _recover_bundle_metadata(checkpoint_path: Path, config_path: Path) -> bool:
    """Create TFT metadata sidecar for checkpoint when missing."""
    if checkpoint_path.suffix != ".pt":
        checkpoint_path = checkpoint_path.with_suffix(".pt")
    meta_path = checkpoint_path.with_suffix(".meta.json")
    if meta_path.exists():
        return True
    if not checkpoint_path.exists() or not config_path.exists():
        return False

    try:
        payload = (
            load_yaml_config(str(config_path), validate_schema=False, validate_cross_config=False)
            or {}
        )
        tft_cfg = payload.get("tft", {})
        metadata = {
            "quantiles": tft_cfg.get("quantiles", []),
            "target": tft_cfg.get("target", "sales"),
            "max_encoder_length": tft_cfg.get("max_encoder_length"),
            "max_prediction_length": tft_cfg.get("max_prediction_length"),
            "group_ids": ["Store"],
            "static_categoricals": tft_cfg.get("static_categoricals", []),
            "static_reals": tft_cfg.get("static_reals", []),
            "time_varying_known_reals": tft_cfg.get("time_varying_known_reals", []),
            "time_varying_unknown_reals": tft_cfg.get("time_varying_unknown_reals", []),
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return True
    except (ValueError, OSError) as exc:
        log_json(logger, logging.WARNING, "bundle_metadata_recovery_failed", error=str(exc))
        return False


def recover_tft_artifacts(config: dict[str, Any] | None = None) -> bool:
    """Attempt lightweight recovery for missing TFT serving artifacts.

    Recovery actions:
      1) Rebuild ``data/processed/tft_reference.parquet`` from raw CSVs.
      2) Ensure ``data/processed/tft_runtime_config.yaml`` exists.
      3) Recreate ``.meta.json`` sidecar when checkpoint exists.
    """
    cfg = config or _load_config()
    checkpoint_path = Path(str(cfg.get("tft_checkpoint", "checkpoints/tft/tft_latest.pt")))
    reference_path = Path(str(cfg.get("processed_data", "data/processed/tft_reference.parquet")))
    runtime_config_path = Path(
        str(cfg.get("runtime_config", "data/processed/tft_runtime_config.yaml"))
    )

    runtime_ok = _recover_runtime_config(runtime_config_path)
    reference_ok = _recover_reference_parquet(reference_path)
    metadata_ok = _recover_bundle_metadata(
        checkpoint_path=checkpoint_path, config_path=runtime_config_path
    )

    return bool(runtime_ok and reference_ok and metadata_ok)


# -----------------------------------------------------------------------
# MLflow integration
# -----------------------------------------------------------------------


def fetch_latest_mlflow_metrics(
    experiment_name: str = "retail_tft_forecasting",
    tracking_uri: str | None = None,
) -> dict[str, float]:
    """Query MLflow for the latest run metrics in a given experiment.

    Falls back gracefully if MLflow is unreachable or the experiment
    does not exist.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: Optional MLflow tracking server URI.  Reads from
            ``configs/rl_config.yaml`` if not provided.

    Returns:
        Dictionary of metric name → latest value.  Empty if unavailable.
    """
    try:
        import mlflow
    except ImportError:
        log_json(logger, logging.WARNING, "mlflow_unavailable", error="mlflow not installed")
        return {}

    if tracking_uri is None:
        rl_cfg_path = Path("configs/rl_config.yaml")
        if rl_cfg_path.exists():
            validate_config_schema(str(rl_cfg_path), "configs/schemas/rl_schema.json")
            rl_cfg = load_yaml_config(str(rl_cfg_path))
            tracking_uri = rl_cfg.get("mlflow", {}).get("tracking_uri")

    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return {}

        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return {}

        latest_run = runs[0]
        metrics = {k: float(v) for k, v in latest_run.data.metrics.items()}
        canonical = {
            MetricNames.SMAPE.value: read_metric(metrics, MetricNames.SMAPE, default=0.0),
            MetricNames.RMSE.value: read_metric(metrics, MetricNames.RMSE, default=0.0),
            MetricNames.COVERAGE_80.value: read_metric(
                metrics, MetricNames.COVERAGE_80, default=0.0
            ),
            MetricNames.BASELINE_SMAPE.value: read_metric(
                metrics, MetricNames.BASELINE_SMAPE, default=0.0
            ),
            MetricNames.BASELINE_RMSE.value: read_metric(
                metrics, MetricNames.BASELINE_RMSE, default=0.0
            ),
        }
        metrics.update(canonical)
        return metrics
    except (AttributeError, OSError, RuntimeError, ValueError) as exc:
        log_json(logger, logging.WARNING, "mlflow_metric_fetch_failed", error=str(exc))
        return {}


# -----------------------------------------------------------------------
# Live RL simulation
# -----------------------------------------------------------------------


def run_rl_simulation_metrics(
    n_days: int = 90,
    config_path: str = "configs/rl_config.yaml",
    store_id: int | None = None,
    risk_aversion: float | None = None,
    n_mc: int = 32,
    request_id: str | None = None,
) -> dict[str, float | str]:
    """Run a deterministic RL rollout and return live KPIs.

    Uses the currently loaded RL agent and its environment to produce
    real service-level, fill-rate, and total-cost values.

    Args:
        n_days: Simulation horizon in days.
        config_path: RL config path.

    Returns:
        Dictionary with ``service_level``, ``fill_rate``, ``total_cost``.
        Empty dict if the agent is unavailable.
    """
    agent_obj = get_rl_agent()
    agent = agent_obj if _agent_has_predict(agent_obj) else None
    normalized_risk = _normalize_risk_aversion(risk_aversion)

    if agent is None:
        log_json(
            logger,
            logging.WARNING,
            "simulation_agent_missing",
            store_id=store_id,
            model_version="unknown",
            request_id=request_id,
        )
        return {}

    env = _extract_env(agent)
    if env is None:
        log_json(
            logger,
            logging.WARNING,
            "simulation_env_missing",
            store_id=store_id,
            model_version="unknown",
            request_id=request_id,
        )
        return {}

    tft_model, tft_dataset, tft_reference_df = get_tft_model()
    if tft_model is None or tft_dataset is None or tft_reference_df is None:
        from src.rl.inventory_env import RetailInventoryEnv

        model_version = str(_load_config().get("model_version", "runtime"))
        reward_type = "multi_objective" if normalized_risk is not None else "auto"
        base_demand = getattr(env, "demand_series", None)
        if base_demand is None:
            base_demand = get_reference_demand_series(min_length=max(int(n_days), 365))

        eval_env = RetailInventoryEnv(
            config_path=config_path,
            demand_series=np.asarray(base_demand, dtype=float),
            forecast_fn=getattr(env, "forecast_fn", None),
            forecast_quantiles_fn=getattr(env, "forecast_quantiles_fn", None),
            store_open_series=getattr(env, "store_open_series", None),
            risk_aversion=normalized_risk,
            reward_type=reward_type,
        )

        observation, _ = eval_env.reset()
        for _ in range(min(int(n_days), int(eval_env.episode_length))):
            action = agent.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = eval_env.step(action)
            if terminated or truncated:
                break

        result = {
            "service_level": float(eval_env.service_level()),
            "fill_rate": float(eval_env.fill_rate()) if hasattr(eval_env, "fill_rate") else 0.0,
            "total_cost": float(eval_env.total_cost()),
            "mc_runs": 1.0,
            "risk_aversion": float(normalized_risk) if normalized_risk is not None else 0.5,
            "model_version": model_version,
        }
        _atomic_swap(latest_simulation=result)
        _mark_cache_loaded("simulation")
        log_json(
            logger,
            logging.INFO,
            "simulation_completed",
            store_id=store_id,
            model_version=model_version,
            request_id=request_id,
            mc_runs=1,
        )
        return result

    if store_id is None:
        stores = (
            sorted(tft_reference_df["Store"].dropna().astype(int).unique().tolist())
            if "Store" in tft_reference_df.columns
            else [1]
        )
        store_id = int(stores[0]) if stores else 1

    try:
        predictions, _ = tft_model.predict_for_store(
            store_id=int(store_id),
            horizon=max(int(n_days), int(getattr(env, "horizon", 7))),
            training_dataset=tft_dataset,
            reference_df=tft_reference_df,
        )
    except (KeyError, RuntimeError, ValueError, AttributeError) as exc:
        log_json(
            logger,
            logging.WARNING,
            "simulation_forecast_failed",
            store_id=store_id,
            model_version="unknown",
            request_id=request_id,
            error=str(exc),
        )
        return {}

    q_idx = get_tft_quantile_indices()
    p10 = predictions[:, q_idx.get("p10", 1)]
    p50 = predictions[:, q_idx.get("p50", 3)]
    p90 = predictions[:, q_idx.get("p90", 5)]
    if hasattr(p10, "detach"):
        p10 = p10.detach().cpu().numpy()
        p50 = p50.detach().cpu().numpy()
        p90 = p90.detach().cpu().numpy()
    else:
        p10 = np.asarray(p10, dtype=float)
        p50 = np.asarray(p50, dtype=float)
        p90 = np.asarray(p90, dtype=float)

    p10, p50, p90 = calibrate_prediction_interval(p10=p10, p50=p50, p90=p90)

    runs: list[dict[str, float]] = []
    reward_type = "multi_objective" if normalized_risk is not None else "auto"
    run_count = max(int(n_mc), 1)
    sim_days = min(int(n_days), int(getattr(env, "episode_length", n_days)))
    scenario_batch = build_scenario_batch(
        p10=np.asarray(p10, dtype=float),
        p50=np.asarray(p50, dtype=float),
        p90=np.asarray(p90, dtype=float),
        n_days=sim_days,
        n_scenarios=run_count,
    )

    from src.rl.inventory_env import RetailInventoryEnv

    env_pool = _get_or_create_sim_env_pool(
        config_path=config_path,
        risk_aversion=normalized_risk,
        reward_type=reward_type,
        pool_size=run_count,
        default_days=sim_days,
        env_class_id=id(RetailInventoryEnv),
    )

    for idx in range(run_count):
        eval_env = env_pool[idx]
        eval_env.demand_series = np.asarray(scenario_batch[idx], dtype=float)

        obs, _ = eval_env.reset()
        for _step in range(min(int(n_days), eval_env.episode_length)):
            action = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env.step(action)
            if terminated or truncated:
                break

        runs.append(
            {
                "service_level": float(eval_env.service_level()),
                "fill_rate": float(eval_env.fill_rate()),
                "total_cost": float(eval_env.total_cost()),
            }
        )

    model_version = str(_load_config().get("model_version", "runtime"))

    result = {
        "service_level": float(np.mean([item["service_level"] for item in runs])) if runs else 0.0,
        "fill_rate": float(np.mean([item["fill_rate"] for item in runs])) if runs else 0.0,
        "total_cost": float(np.mean([item["total_cost"] for item in runs])) if runs else 0.0,
        "mc_runs": float(max(int(n_mc), 1)),
        "risk_aversion": float(normalized_risk) if normalized_risk is not None else 0.5,
        "model_version": model_version,
    }
    _atomic_swap(latest_simulation=result)
    _mark_cache_loaded("simulation")
    log_json(
        logger,
        logging.INFO,
        "simulation_completed",
        store_id=store_id,
        model_version=model_version,
        request_id=request_id,
        mc_runs=run_count,
    )
    return result


async def run_rl_simulation_metrics_async(
    store_ids: list[int],
    n_days: int = 90,
    risk_aversion: float = 0.5,
    n_mc: int = 16,
    request_id: str | None = None,
) -> list[dict[str, float]]:
    """Run simulations concurrently for multiple stores without blocking.

    Raises:
        SimulationUnavailableError: If no RL agent (or its environment) is loaded.
    """

    async def _one(store_id: int) -> dict[str, float]:
        result = await asyncio.to_thread(
            run_rl_simulation_metrics,
            n_days,
            "configs/rl_config.yaml",
            int(store_id),
            float(risk_aversion),
            int(n_mc),
            request_id,
        )
        if not result:
            raise SimulationUnavailableError("no RL agent loaded")
        return {
            "store_id": float(store_id),
            "service_level": float(result.get("service_level", 0.0)),
            "fill_rate": float(result.get("fill_rate", 0.0)),
            "total_cost": float(result.get("total_cost", 0.0)),
            "risk_aversion": float(result.get("risk_aversion", risk_aversion)),
        }

    return await asyncio.gather(*[_one(sid) for sid in store_ids])


def get_reference_demand_series(min_length: int = 365) -> npt.NDArray[np.float64]:
    """Load robust demand series from TFT reference artifacts.

    Args:
        min_length: Minimum target length for downstream RL rollouts.

    Returns:
        Demand series array. Falls back to train.csv-derived profile when
        reference artifacts are unavailable.
    """
    cfg = _load_config()

    # 1) Prefer loaded reference dataframe from registry (thread-safe read)
    reference_df = copy.deepcopy(_STATE.tft_reference_df)
    if reference_df is not None:
        for column in ("Sales", "sales"):
            if column in reference_df.columns:
                values = reference_df[column].dropna().to_numpy(dtype=float)
                if values.size > 0:
                    return _expand_series(values, min_length=min_length)

    # 2) Fallback to persisted reference parquet
    ref_path = Path(str(cfg.get("processed_data", "")))
    if ref_path.exists():
        try:
            ref_df = pd.read_parquet(ref_path)
            for column in ("Sales", "sales"):
                if column in ref_df.columns:
                    values = ref_df[column].dropna().to_numpy(dtype=float)
                    if values.size > 0:
                        return _expand_series(values, min_length=min_length)
        except (OSError, ValueError):
            pass

    # 3) Final fallback from raw train CSV historical distribution
    train_path = Path("data/raw/train.csv")
    if train_path.exists():
        try:
            train_df = pd.read_csv(train_path, usecols=["Sales"])
            values = train_df["Sales"].dropna().to_numpy(dtype=float)
            if values.size > 0:
                return _expand_series(values, min_length=min_length)
        except (OSError, ValueError):
            pass

    # 4) Defensive fallback when no data files exist
    raise FileNotFoundError("Unable to derive reference demand series from registry artifacts.")


def _expand_series(
    values: npt.NDArray[np.float64], min_length: int = 365
) -> npt.NDArray[np.float64]:
    """Repeat/truncate series to satisfy minimum rollout length."""
    if values.size >= min_length:
        return values[:min_length].astype(float)
    repeats = int(np.ceil(float(min_length) / max(values.size, 1)))
    return np.tile(values, repeats)[:min_length].astype(float)


# -----------------------------------------------------------------------
# TFT model loading
# -----------------------------------------------------------------------


def load_tft_model(
    checkpoint_path: str | None = None,
    dataset_df: pd.DataFrame | None = None,
    reference_df: pd.DataFrame | None = None,
    config_path: str = "configs/tft_config.yaml",
) -> None:
    """Load TFT model and dataset into singleton registry.

    Args:
        checkpoint_path: Optional model checkpoint path.
        dataset_df: Optional DataFrame used to build TimeSeriesDataSet.
        reference_df: Optional reference DataFrame for inference lookups.
        config_path: TFT config path.

    Raises:
        FileNotFoundError: If checkpoint path is provided but missing.
    """
    from src.models.tft_dataset import build_tft_dataset
    from src.models.tft_model import TFTForecaster

    dataset = (
        build_tft_dataset(dataset_df, config_path=config_path) if dataset_df is not None else None
    )
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"TFT checkpoint not found: {ckpt}")
        forecaster = TFTForecaster.load_any(str(ckpt), config_path, dataset)
    else:
        forecaster = TFTForecaster(config_path, dataset)
        forecaster.build()

    # Ensure reference dataframe has both 'Store' and 'store' columns
    # for compatibility with model config and inference code
    reference_df_out = reference_df if reference_df is not None else dataset_df
    if reference_df_out is not None:
        if "store" in reference_df_out.columns and "Store" not in reference_df_out.columns:
            reference_df_out = reference_df_out.copy()
            reference_df_out["Store"] = reference_df_out["store"]
        if "Store" in reference_df_out.columns and "store" not in reference_df_out.columns:
            reference_df_out = reference_df_out.copy()
            reference_df_out["store"] = reference_df_out["Store"]

    _atomic_swap(
        tft_dataset=dataset,
        tft_reference_df=reference_df_out,
        tft_model=forecaster,
    )


def load_tft_from_artifacts(
    checkpoint_path: str = "checkpoints/tft/tft_latest.pt",
    reference_path: str = "data/processed/tft_reference.parquet",
    config_path: str = "configs/tft_config.yaml",
    fingerprint_path: str = "data/processed/tft_fingerprint.json",
) -> None:
    """Load TFT model from persisted artifacts.

    Args:
        checkpoint_path: Path to model state dict.
        reference_path: Path to feature-engineered reference dataframe.
        config_path: Fallback config path.
        fingerprint_path: Optional fingerprint metadata path.

    Raises:
        FileNotFoundError: If required artifact files are missing.
    """
    ckpt = Path(checkpoint_path)
    reference = Path(reference_path)
    runtime_cfg = Path(config_path)

    if not ckpt.exists() or not reference.exists():
        recover_tft_artifacts(
            {
                "tft_checkpoint": str(ckpt),
                "processed_data": str(reference),
                "runtime_config": str(runtime_cfg),
            }
        )

    if not ckpt.exists():
        raise FileNotFoundError(f"Missing TFT checkpoint: {ckpt}")
    if not reference.exists():
        raise FileNotFoundError(f"Missing TFT reference dataset: {reference}")

    effective_config = config_path
    fingerprint = Path(fingerprint_path)
    if fingerprint.exists():
        try:
            metadata = json.loads(fingerprint.read_text(encoding="utf-8"))
            candidate = metadata.get("config_path")
            if isinstance(candidate, str) and Path(candidate).exists():
                effective_config = candidate
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    reference_df = pd.read_parquet(reference)
    load_tft_model(
        checkpoint_path=str(ckpt),
        dataset_df=reference_df,
        reference_df=reference_df,
        config_path=effective_config,
    )


def get_tft_model() -> tuple[Any | None, Any | None, Any | None]:
    """Get loaded TFT model and dataset, loading lazily if needed.

    Returns:
        Tuple of forecaster, dataset, and reference dataframe.
    """
    cfg = _load_config()
    if _cache_expired("tft", config=cfg):
        invalidate_cache("tft")

    if _STATE.tft_model is not None:
        return _STATE.tft_model, _STATE.tft_dataset, _STATE.tft_reference_df

    with _TFT_LOAD_LOCK:
        if _STATE.tft_model is not None:
            return _STATE.tft_model, _STATE.tft_dataset, _STATE.tft_reference_df
        return _load_tft_model(cfg)


def _load_tft_model(cfg: dict[str, Any]) -> tuple[Any | None, Any | None, Any | None]:
    """Load TFT artifacts, recovering them if missing. Caller holds ``_TFT_LOAD_LOCK``."""
    try:
        for attempt in range(2):
            checkpoint_path = str(cfg.get("tft_checkpoint"))
            if not Path(checkpoint_path).exists():
                latest_tft = _latest_file("checkpoints/tft/*.pt")
                if latest_tft is not None:
                    checkpoint_path = str(latest_tft)

            ref_path = Path(str(cfg.get("processed_data", "")))
            if not ref_path.exists() or not Path(checkpoint_path).exists():
                recover_tft_artifacts(cfg)

            load_tft_from_artifacts(
                checkpoint_path=checkpoint_path,
                reference_path=str(cfg.get("processed_data")),
                config_path=str(cfg.get("runtime_config")),
            )
            if _STATE.tft_model is not None:
                break

            if attempt == 0:
                recover_tft_artifacts(cfg)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, OSError):
        # Even if model loading failed, try to load reference_df for fallback forecasts
        ref_path = Path(str(cfg.get("processed_data", "data/processed/tft_reference.parquet")))
        if _STATE.tft_reference_df is None and ref_path.exists():
            try:
                _atomic_swap(tft_reference_df=pd.read_parquet(ref_path))
            except Exception:
                pass
        return None, None, _STATE.tft_reference_df

    if _STATE.tft_model is not None:
        _mark_cache_loaded("tft")

    return _STATE.tft_model, _STATE.tft_dataset, _STATE.tft_reference_df


def get_tft_quantile_indices() -> dict[str, int]:
    """Return resolved TFT quantile indices from loaded model/config."""
    model, _dataset, _reference = get_tft_model()
    if model is not None:
        quantiles = list(getattr(model, "config", {}).get("quantiles", []))
        if quantiles:
            return resolve_quantile_indices(quantiles)

    cfg = _load_config()
    runtime_cfg = Path(str(cfg.get("runtime_config", "data/processed/tft_runtime_config.yaml")))
    if runtime_cfg.exists():
        validate_config_schema(str(runtime_cfg), "configs/schemas/tft_schema.json")
        payload = load_yaml_config(str(runtime_cfg))
        quantiles = list(payload.get("tft", {}).get("quantiles", []))
        if quantiles:
            return resolve_quantile_indices(quantiles)

    return {"p10": 1, "p50": 3, "p90": 5}


# -----------------------------------------------------------------------
# RL agent loading
# -----------------------------------------------------------------------


def load_rl_agent(
    config_path: str = "configs/rl_config.yaml",
    demand_series: npt.NDArray[np.float64] | None = None,
) -> None:
    """Load RL agent into singleton registry.

    Args:
        config_path: RL config path.
        demand_series: Optional demand series used by environment.
    """
    from src.rl.agent import RLInventoryAgent
    from src.rl.inventory_env import RetailInventoryEnv

    env = RetailInventoryEnv(config_path=config_path, demand_series=demand_series)
    agent = RLInventoryAgent(config_path=config_path, env=env)
    agent.build()
    _atomic_swap(rl_agent=agent)


def get_rl_agent() -> Any | None:
    """Get loaded RL agent, loading lazily if possible.

    Returns:
        RL agent instance or None.
    """
    cfg = _load_config()
    if _cache_expired("rl", config=cfg):
        invalidate_cache("rl")

    if _STATE.rl_agent is not None:
        return _STATE.rl_agent

    with _RL_LOAD_LOCK:
        if _STATE.rl_agent is not None:
            return _STATE.rl_agent
        return _load_rl_agent(cfg)


def _load_rl_agent(cfg: dict[str, Any]) -> Any | None:
    """Load the RL agent from checkpoint or build a fresh one. Caller holds ``_RL_LOAD_LOCK``."""
    ckpt = Path(str(cfg.get("rl_checkpoint")))
    if not ckpt.exists():
        latest_rl = _latest_file("checkpoints/rl/*.zip")
        if latest_rl is not None:
            ckpt = latest_rl

    if ckpt.exists():
        try:
            from stable_baselines3 import PPO

            from src.rl.agent import RLInventoryAgent
            from src.rl.inventory_env import RetailInventoryEnv

            demand_series = get_reference_demand_series(min_length=365)

            env = RetailInventoryEnv(
                config_path="configs/rl_config.yaml",
                demand_series=demand_series,
            )
            agent = RLInventoryAgent(config_path="configs/rl_config.yaml", env=env)
            agent.model = PPO.load(str(ckpt))
            _atomic_swap(rl_agent=agent)
        except (FileNotFoundError, RuntimeError, ValueError, OSError):
            _atomic_swap(rl_agent=None)
    else:
        try:
            load_rl_agent(
                config_path="configs/rl_config.yaml",
                demand_series=get_reference_demand_series(min_length=365),
            )
        except (FileNotFoundError, RuntimeError, ValueError, OSError):
            _atomic_swap(rl_agent=None)

    if _STATE.rl_agent is not None:
        _mark_cache_loaded("rl")

    return _STATE.rl_agent


# -----------------------------------------------------------------------
# CL metrics
# -----------------------------------------------------------------------


def get_cl_metrics(path: str | None = None) -> dict[str, Any]:
    """Load latest CL metrics from artifact JSON with lazy caching.

    Tries local artifact first, then falls back to MLflow query.

    Args:
        path: Optional override metrics path.

    Returns:
        Parsed metrics dictionary, or empty dict if unavailable.
    """
    cfg = _load_config()
    if _cache_expired("cl_metrics", config=cfg):
        invalidate_cache("cl_metrics")

    if _STATE.cl_metrics is not None:
        return dict(_STATE.cl_metrics)

    metrics_path = Path(path or str(cfg.get("cl_metrics")))
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            payload = {}
        _atomic_swap(cl_metrics=payload)
        _mark_cache_loaded("cl_metrics")
        return dict(_STATE.cl_metrics or {})

    # Fallback: try MLflow
    mlflow_metrics = fetch_latest_mlflow_metrics(experiment_name="retail_cl_training")
    if mlflow_metrics:
        _atomic_swap(cl_metrics=mlflow_metrics)
        _mark_cache_loaded("cl_metrics")
        return dict(_STATE.cl_metrics or {})

    _atomic_swap(cl_metrics={})
    _mark_cache_loaded("cl_metrics")
    return dict(_STATE.cl_metrics or {})


def _extract_metric_from_runs(runs: list[Any], metric_name: str) -> float | None:
    """Extract first available metric from sorted run list.

    Args:
        runs: MLflow run list sorted by start time desc.
        metric_name: Metric key.

    Returns:
        Metric value if found.
    """
    for run in runs:
        value = run.data.metrics.get(metric_name)
        if value is not None:
            return float(value)
    return None


def _tracking_uri_reachable(tracking_uri: str | None, timeout_s: float = 0.5) -> bool:
    """Check whether MLflow tracking URI is reachable.

    Args:
        tracking_uri: Tracking URI string.
        timeout_s: Socket timeout in seconds.

    Returns:
        ``True`` when URI is local file backend or reachable network endpoint.
    """
    if tracking_uri is None or tracking_uri == "":
        return True
    if tracking_uri.startswith("file:"):
        return True
    parsed = urlparse(tracking_uri)
    if parsed.scheme not in {"http", "https"}:
        return True
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if host is None:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def fetch_live_kpi_payload() -> dict[str, float | str] | None:
    """Build real-time KPI payload from MLflow runs.

    The function searches the latest successful runs in:
      - ``retail_tft_forecasting`` for forecast KPIs
      - ``retail_rl_inventory`` for service/cost KPIs

    Returns:
        KPI payload dictionary, or ``None`` when unavailable.
    """
    try:
        import mlflow
    except ImportError:
        return None

    rl_cfg_path = Path("configs/rl_config.yaml")
    tracking_uri = None
    if rl_cfg_path.exists():
        validate_config_schema(str(rl_cfg_path), "configs/schemas/rl_schema.json")
        rl_cfg = load_yaml_config(str(rl_cfg_path))
        tracking_uri = rl_cfg.get("mlflow", {}).get("tracking_uri")
    if tracking_uri and not _tracking_uri_reachable(str(tracking_uri)):
        log_json(
            logger, logging.WARNING, "mlflow_tracking_unreachable", tracking_uri=str(tracking_uri)
        )
        return None

    if tracking_uri:
        mlflow.set_tracking_uri(str(tracking_uri))

    try:
        client = mlflow.tracking.MlflowClient()

        forecast_exp = mlflow.get_experiment_by_name("retail_tft_forecasting")
        rl_exp = mlflow.get_experiment_by_name("retail_rl_inventory")
        if forecast_exp is None and rl_exp is None:
            return None

        forecast_runs = (
            client.search_runs(
                experiment_ids=[forecast_exp.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=25,
            )
            if forecast_exp is not None
            else []
        )
        rl_runs = (
            client.search_runs(
                experiment_ids=[rl_exp.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=25,
            )
            if rl_exp is not None
            else []
        )
    except (AttributeError, OSError, RuntimeError, ValueError) as exc:
        log_json(logger, logging.WARNING, "live_kpi_query_failed", error=str(exc))
        return None

    forecast_metrics = forecast_runs[0].data.metrics if forecast_runs else {}
    rl_metrics = rl_runs[0].data.metrics if rl_runs else {}
    merged_payload: dict[str, float | int | str] = {}
    merged_payload.update({k: float(v) for k, v in forecast_metrics.items()})
    merged_payload.update({k: float(v) for k, v in rl_metrics.items()})

    val_smape = read_metric(merged_payload, MetricNames.SMAPE, default=0.0)
    baseline_smape = read_metric(merged_payload, MetricNames.BASELINE_SMAPE, default=val_smape)
    val_rmse = read_metric(merged_payload, MetricNames.RMSE, default=0.0)
    baseline_rmse = read_metric(merged_payload, MetricNames.BASELINE_RMSE, default=val_rmse)
    coverage = read_metric(merged_payload, MetricNames.COVERAGE_80, default=0.0)
    service_level = read_metric(merged_payload, MetricNames.SERVICE_LEVEL, default=0.0)
    fill_rate = read_metric(merged_payload, MetricNames.FILL_RATE, default=0.0)
    total_cost = read_metric(merged_payload, MetricNames.TOTAL_COST, default=0.0)

    if val_smape <= 0.0 and service_level <= 0.0:
        return None

    model_version = "mlflow_latest"
    if forecast_runs:
        model_version = forecast_runs[0].info.run_id

    return {
        MetricNames.SMAPE.value: float(val_smape),
        MetricNames.BASELINE_SMAPE.value: float(baseline_smape),
        MetricNames.SMAPE_UPLIFT.value: float(baseline_smape - val_smape),
        MetricNames.RMSE.value: float(val_rmse),
        MetricNames.BASELINE_RMSE.value: float(baseline_rmse),
        MetricNames.COVERAGE_80.value: float(coverage),
        MetricNames.SERVICE_LEVEL.value: float(service_level),
        MetricNames.FILL_RATE.value: float(fill_rate),
        MetricNames.TOTAL_COST.value: float(total_cost),
        "model_version": str(model_version),
    }


def get_latest_simulation() -> dict[str, float | str]:
    """Return cached results from the last RL simulation.

    This function is intentionally non-blocking for API KPI endpoints.
    It returns only cached simulation values and does not trigger a new
    rollout implicitly.

    Returns:
        Dictionary with ``service_level``, ``fill_rate``, ``total_cost``.
    """
    return dict(_STATE.latest_simulation)


# -----------------------------------------------------------------------
# Cache management
# -----------------------------------------------------------------------


def invalidate_cache(key: str | None = None) -> None:
    """Invalidate registry cache.

    Args:
        key: Optional key among ``tft``, ``rl``, ``cl_metrics``, ``simulation``.
    """
    global _SIM_ENV_POOL_SIGNATURE

    if key is None:
        _atomic_swap(
            tft_model=None,
            tft_dataset=None,
            tft_reference_df=None,
            rl_agent=None,
            cl_metrics=None,
            latest_simulation={},
            forecast_cache={},
        )
        _CACHE_LOADED_AT.update({"tft": None, "rl": None, "cl_metrics": None, "simulation": None})
        _SIM_ENV_POOL.clear()
        _SIM_ENV_POOL_SIGNATURE = None
        return

    if key == "tft":
        _atomic_swap(tft_model=None, tft_dataset=None, tft_reference_df=None, forecast_cache={})
        _CACHE_LOADED_AT["tft"] = None
        _SIM_ENV_POOL.clear()
        _SIM_ENV_POOL_SIGNATURE = None
    elif key == "rl":
        _atomic_swap(rl_agent=None)
        _CACHE_LOADED_AT["rl"] = None
    elif key == "cl_metrics":
        _atomic_swap(cl_metrics=None)
        _CACHE_LOADED_AT["cl_metrics"] = None
    elif key == "simulation":
        _atomic_swap(latest_simulation={})
        _CACHE_LOADED_AT["simulation"] = None
        _SIM_ENV_POOL.clear()
        _SIM_ENV_POOL_SIGNATURE = None


def clear_registry() -> None:
    """Compatibility helper used by tests.

    Equivalent to ``invalidate_cache()``.
    """
    invalidate_cache()


def model_registry_ready() -> tuple[bool, str]:
    """Return readiness state for container health checks."""
    cfg = _load_config()

    tft_ckpt = Path(str(cfg.get("tft_checkpoint", "")))
    tft_ref = Path(str(cfg.get("processed_data", "")))
    rl_cfg_path = Path("configs/rl_config.yaml")

    if not tft_ckpt.exists() or not tft_ref.exists():
        recovered = recover_tft_artifacts(cfg)
        tft_ckpt = Path(str(cfg.get("tft_checkpoint", "")))
        tft_ref = Path(str(cfg.get("processed_data", "")))
        if (not recovered) or (not tft_ckpt.exists()) or (not tft_ref.exists()):
            return False, "TFT artifacts missing"
    if not rl_cfg_path.exists():
        return False, "RL config missing"

    tracking_uri = None
    try:
        validate_config_schema(str(rl_cfg_path), "configs/schemas/rl_schema.json")
        rl_cfg = load_yaml_config(str(rl_cfg_path))
        tracking_uri = rl_cfg.get("mlflow", {}).get("tracking_uri")
    except (OSError, ValueError):
        tracking_uri = None

    if tracking_uri and not _tracking_uri_reachable(str(tracking_uri)):
        log_json(
            logger,
            logging.WARNING,
            "mlflow_unreachable_but_artifacts_present",
            tracking_uri=str(tracking_uri),
        )
        # MLflow unreachable is not a hard failure when model artifacts exist locally
    return True, "ok"
