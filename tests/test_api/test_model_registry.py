"""Tests for ``src.api.model_registry``: helpers, artifact recovery, MLflow, loaders, caches.

Every test runs in an empty ``tmp_path`` cwd with a clean registry singleton, so no repo
artifacts (data/, checkpoints/, mlruns/) are read. Heavy dependencies are replaced with
lightweight fakes via ``sys.modules`` / ``monkeypatch``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.api import model_registry as reg
from tests.test_api.conftest import FakeTensor

pytestmark = pytest.mark.usefixtures("isolated_registry")


# -----------------------------------------------------------------------
# Shared fakes
# -----------------------------------------------------------------------


class _FakeEnv:
    """Stand-in for RetailInventoryEnv that records construction kwargs."""

    episode_length = 5
    instances: list[_FakeEnv] = []

    def __init__(self, config_path: str = "", demand_series: Any = None, **kwargs: Any) -> None:
        self.config_path = config_path
        self.demand_series = demand_series
        self.kwargs = kwargs
        self.steps = 0
        _FakeEnv.instances.append(self)

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self.steps = 0
        return np.zeros(1), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        return np.zeros(1), 0.0, self.steps >= 2, False, {}

    def service_level(self) -> float:
        return 0.9

    def fill_rate(self) -> float:
        return 0.8

    def total_cost(self) -> float:
        return 100.0 + self.steps


@pytest.fixture()
def fake_env(monkeypatch: pytest.MonkeyPatch) -> type[_FakeEnv]:
    """Patch the RL environment class used by registry simulations."""
    _FakeEnv.instances = []
    monkeypatch.setattr("src.rl.inventory_env.RetailInventoryEnv", _FakeEnv)
    return _FakeEnv


class _Agent:
    """Agent exposing ``predict`` and a wrapped env."""

    def __init__(self, env: Any) -> None:
        self.env = types.SimpleNamespace(env=env)
        self.calls = 0

    def predict(self, obs: Any, deterministic: bool = True) -> np.ndarray:
        self.calls += 1
        return np.array([1.0])


def _fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _run(metrics: dict[str, float], run_id: str = "run-1") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        data=types.SimpleNamespace(metrics=metrics), info=types.SimpleNamespace(run_id=run_id)
    )


def _fake_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    experiments: dict[str, Any],
    runs_by_exp: dict[str, list[Any]],
    search_error: Exception | None = None,
) -> dict[str, Any]:
    """Install a fake ``mlflow`` module; return a dict recording calls."""
    calls: dict[str, Any] = {"tracking_uri": None, "search": []}

    class _Client:
        def search_runs(self, experiment_ids: list[str], **kwargs: Any) -> list[Any]:
            if search_error is not None:
                raise search_error
            calls["search"].append((experiment_ids, kwargs))
            return runs_by_exp.get(experiment_ids[0], [])

    def _set_uri(uri: str) -> None:
        calls["tracking_uri"] = uri

    _fake_module(
        monkeypatch,
        "mlflow",
        set_tracking_uri=_set_uri,
        get_experiment_by_name=lambda name: experiments.get(name),
        tracking=types.SimpleNamespace(MlflowClient=_Client),
    )
    return calls


def _write_rl_config(monkeypatch: pytest.MonkeyPatch, tracking_uri: str | None) -> None:
    """Create a dummy rl_config and patch config loaders to return given URI."""
    Path("configs").mkdir(exist_ok=True)
    Path("configs/rl_config.yaml").write_text("mlflow: {}\n", encoding="utf-8")
    monkeypatch.setattr(reg, "validate_config_schema", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        reg, "load_yaml_config", lambda *args, **kwargs: {"mlflow": {"tracking_uri": tracking_uri}}
    )


# -----------------------------------------------------------------------
# Small pure helpers
# -----------------------------------------------------------------------


def test_agent_has_predict_and_extract_env() -> None:
    inner = object()
    wrapped = types.SimpleNamespace(env=types.SimpleNamespace(env=inner))
    plain_env = types.SimpleNamespace(horizon=7)
    unwrapped = types.SimpleNamespace(env=plain_env)

    assert reg._agent_has_predict(_Agent(inner)) is True
    assert reg._agent_has_predict(types.SimpleNamespace(predict=5)) is False
    assert reg._agent_has_predict(object()) is False
    assert reg._extract_env(object()) is None
    assert reg._extract_env(wrapped) is inner
    assert reg._extract_env(unwrapped) is plain_env


@pytest.mark.parametrize(
    ("parser", "config", "expected"),
    [
        ("_registry_ttl_seconds", {"cache_ttl_seconds": 30}, 30.0),
        ("_registry_ttl_seconds", {"cache_ttl_seconds": 0}, 1.0),
        ("_registry_ttl_seconds", {"cache_ttl_seconds": "abc"}, 900.0),
        ("_registry_ttl_seconds", None, 900.0),  # empty cwd -> default config, no TTL key
        ("_forecast_interval_scale", {"forecast_interval_scale": 2.5}, 2.5),
        ("_forecast_interval_scale", {"forecast_interval_scale": 0.5}, 1.0),
        ("_forecast_interval_scale", {"forecast_interval_scale": "bad"}, 1.0),
        ("_forecast_interval_scale", {"forecast_interval_scale": None}, 1.0),
        ("_forecast_interval_scale", None, 1.0),
    ],
)
def test_numeric_config_parsing(
    parser: str, config: dict[str, Any] | None, expected: float
) -> None:
    assert getattr(reg, parser)(config) == expected


@pytest.mark.parametrize(
    ("bands", "scale", "expected"),
    [
        # Identity scale clips negatives at zero and re-orders crossed bands.
        (([-5.0, 12.0], [-1.0, 10.0], [3.0, 8.0]), 1.0, ([0.0, 10.0], [0.0, 10.0], [3.0, 10.0])),
        # Scale below 1 is treated as identity.
        (([90.0], [100.0], [110.0]), 0.5, ([90.0], [100.0], [110.0])),
        # Scale > 1 widens around p50; the lower band is clipped at zero.
        (
            ([90.0, 0.0], [100.0, 10.0], [110.0, 20.0]),
            3.0,
            ([70.0, 0.0], [100.0, 10.0], [130.0, 40.0]),
        ),
    ],
)
def test_calibrate_prediction_interval(
    bands: tuple[list[float], ...], scale: float, expected: tuple[list[float], ...]
) -> None:
    p10, p50, p90 = bands
    result = reg.calibrate_prediction_interval(p10=p10, p50=p50, p90=p90, interval_scale=scale)
    for actual, want in zip(result, expected):
        np.testing.assert_allclose(actual, want)


def test_normalize_risk_aversion_clamps() -> None:
    assert reg._normalize_risk_aversion(None) is None
    assert reg._normalize_risk_aversion(-0.3) == 0.0
    assert reg._normalize_risk_aversion(0.4) == pytest.approx(0.4)
    assert reg._normalize_risk_aversion(2) == 1.0


def test_cache_expiry_tracking() -> None:
    assert reg._cache_expired("tft") is False
    reg._CACHE_LOADED_AT["tft"] = time.time() - 10
    assert reg._cache_expired("tft", config={"cache_ttl_seconds": 5}) is True
    assert reg._cache_expired("tft", config={"cache_ttl_seconds": 100}) is False

    reg._mark_cache_loaded("rl")
    assert reg._CACHE_LOADED_AT["rl"] == pytest.approx(time.time(), abs=5)


def test_forecast_cache_roundtrip_is_deep_copied() -> None:
    assert reg.get_cached_forecast(store_id=1, horizon=7) is None

    payload = {"store_id": 1, "p50": [1.0, 2.0]}
    reg.cache_forecast_payload(store_id=1, horizon=7, payload=payload)
    payload["p50"].append(99.0)

    cached = reg.get_cached_forecast(store_id=1, horizon=7)
    assert cached == {"store_id": 1, "p50": [1.0, 2.0]}
    cached["p50"].clear()
    assert reg.get_cached_forecast(store_id=1, horizon=7)["p50"] == [1.0, 2.0]
    assert reg.get_cached_forecast(store_id=2, horizon=7) is None


def test_extract_metric_from_runs() -> None:
    runs = [_run({"other": 1.0}), _run({"smape": 12}), _run({"smape": 99.0})]
    assert reg._extract_metric_from_runs(runs, "smape") == 12.0
    assert reg._extract_metric_from_runs(runs, "missing") is None
    assert reg._extract_metric_from_runs([], "smape") is None


def test_resolve_quantile_indices_casts_to_int(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[float]] = []

    def _resolve(quantiles: list[float]) -> dict[Any, Any]:
        seen.append(quantiles)
        return {"p10": np.int64(0), "p50": np.int64(2), "p90": np.int64(4)}

    _fake_module(monkeypatch, "src.models.tft_model", resolve_quantile_indices=_resolve)

    result = reg.resolve_quantile_indices([0.1, 0.3, 0.5, 0.7, 0.9])
    assert result == {"p10": 0, "p50": 2, "p90": 4}
    assert all(isinstance(v, int) and not isinstance(v, np.integer) for v in result.values())
    assert seen == [[0.1, 0.3, 0.5, 0.7, 0.9]]


# -----------------------------------------------------------------------
# Config + filesystem helpers / artifact recovery
# -----------------------------------------------------------------------


def test_load_config_defaults_when_missing() -> None:
    cfg = reg._load_config("does/not/exist.yaml")
    assert cfg["tft_checkpoint"] == "checkpoints/tft/tft_latest.pt"
    assert cfg["processed_data"] == "data/processed/tft_reference.parquet"
    assert cfg["forecast_interval_scale"] == 1.0
    assert "cache_ttl_seconds" not in cfg


def test_load_config_reads_models_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    validated: list[tuple[str, str]] = []
    monkeypatch.setattr(reg, "validate_config_schema", lambda p, s: validated.append((p, s)))
    cfg_path = tmp_path / "api_model.yaml"
    cfg_path.write_text(
        "models:\n  tft_checkpoint: custom.pt\n  cache_ttl_seconds: 60\n", encoding="utf-8"
    )

    cfg = reg._load_config(str(cfg_path))

    assert validated == [(str(cfg_path), "configs/schemas/api_schema.json")]
    assert cfg["tft_checkpoint"] == "custom.pt"
    assert cfg["cache_ttl_seconds"] == 60
    assert cfg["processed_data"] == "data/processed/tft_reference.parquet"
    assert cfg["model_version"] == "runtime"
    assert cfg["forecast_warmup_top_n"] == 10


def test_latest_file_picks_newest(tmp_path: Path) -> None:
    assert reg._latest_file("ckpt/*.pt") is None
    (tmp_path / "ckpt").mkdir()
    old = tmp_path / "ckpt" / "old.pt"
    new = tmp_path / "ckpt" / "new.pt"
    old.write_bytes(b"a")
    new.write_bytes(b"b")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    latest = reg._latest_file("ckpt/*.pt")
    assert latest is not None and latest.name == "new.pt"


def test_raw_rossmann_data_ready_requires_all_three_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "train.csv").write_text("x\n", encoding="utf-8")
    assert reg._raw_rossmann_data_ready(str(raw)) is False
    (raw / "test.csv").write_text("x\n", encoding="utf-8")
    (raw / "store.csv").write_text("x\n", encoding="utf-8")
    assert reg._raw_rossmann_data_ready(str(raw)) is True


def test_recover_runtime_config_existing_and_missing_base(tmp_path: Path) -> None:
    existing = tmp_path / "rt.yaml"
    existing.write_text("tft: {}\n", encoding="utf-8")
    assert reg._recover_runtime_config(existing) is True
    # No configs/tft_config.yaml in the isolated cwd.
    missing = tmp_path / "out" / "rt.yaml"
    assert reg._recover_runtime_config(missing) is False
    assert not missing.exists()


def _write_raw_csvs(root: Path) -> None:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    for name in ("train.csv", "test.csv", "store.csv"):
        (raw / name).write_text("Store\n1\n", encoding="utf-8")


def test_recover_reference_parquet_short_circuits(tmp_path: Path) -> None:
    existing = tmp_path / "ref.parquet"
    existing.write_bytes(b"x")
    assert reg._recover_reference_parquet(existing) is True
    # Raw CSVs absent -> cannot rebuild.
    assert reg._recover_reference_parquet(tmp_path / "missing.parquet") is False


def test_recover_reference_parquet_rebuilds_from_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_raw_csvs(tmp_path)
    raw_df = pd.DataFrame({"Store": [2, 1, 1], "time_idx": [0, 1, 0], "sales": [5.0, 7.0, 6.0]})

    class _Loader:
        def __init__(self, data_dir: str) -> None:
            assert data_dir == "data/raw"

        def load(self) -> tuple[pd.DataFrame, None]:
            return raw_df.copy(), None

    class _Pre:
        def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
            return df.assign(pre=True)

    class _FE:
        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            return df.assign(fe=True)

    _fake_module(monkeypatch, "src.data.loader", RossmannDataLoader=_Loader)
    _fake_module(monkeypatch, "src.data.preprocessor", RossmannPreprocessor=_Pre)
    _fake_module(monkeypatch, "src.data.feature_engineer", FeatureEngineer=_FE)

    target = tmp_path / "data" / "processed" / "tft_reference.parquet"
    assert reg._recover_reference_parquet(target) is True

    out = pd.read_parquet(target)
    assert list(zip(out["Store"], out["time_idx"])) == [(1, 0), (1, 1), (2, 0)]
    assert bool(out["pre"].all()) and bool(out["fe"].all())


def test_recover_reference_parquet_handles_loader_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_raw_csvs(tmp_path)

    class _Loader:
        def __init__(self, data_dir: str) -> None:
            raise FileNotFoundError("train.csv unreadable")

    _fake_module(monkeypatch, "src.data.loader", RossmannDataLoader=_Loader)
    _fake_module(monkeypatch, "src.data.preprocessor", RossmannPreprocessor=object)
    _fake_module(monkeypatch, "src.data.feature_engineer", FeatureEngineer=object)

    target = tmp_path / "ref.parquet"
    assert reg._recover_reference_parquet(target) is False
    assert not target.exists()


def test_recover_bundle_metadata_edge_cases(tmp_path: Path) -> None:
    cfg = tmp_path / "rt.yaml"
    cfg.write_text("tft: {}\n", encoding="utf-8")

    # Non-.pt suffix is normalised; an existing sidecar short-circuits.
    (tmp_path / "model.meta.json").write_text("{}", encoding="utf-8")
    assert reg._recover_bundle_metadata(tmp_path / "model.ckpt", cfg) is True

    # Missing checkpoint -> cannot create sidecar.
    assert reg._recover_bundle_metadata(tmp_path / "absent.pt", cfg) is False

    # Invalid (non-mapping) config -> ValueError handled -> False.
    ckpt = tmp_path / "bad.pt"
    ckpt.write_bytes(b"w")
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("- a\n- b\n", encoding="utf-8")
    assert reg._recover_bundle_metadata(ckpt, bad_cfg) is False
    assert not (tmp_path / "bad.meta.json").exists()


def test_recover_tft_artifacts_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "tft_config.yaml").write_text(
        "tft:\n  quantiles: [0.1, 0.5, 0.9]\n  target: sales\n", encoding="utf-8"
    )
    cfg = {
        "tft_checkpoint": "ck/m.pt",
        "processed_data": "ref.parquet",
        "runtime_config": "rt.yaml",
    }

    # Reference cannot be rebuilt (no raw data) -> overall False, but runtime config is copied.
    assert reg.recover_tft_artifacts(cfg) is False
    assert (tmp_path / "rt.yaml").read_text(encoding="utf-8").startswith("tft:")

    (tmp_path / "ref.parquet").write_bytes(b"x")
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck" / "m.pt").write_bytes(b"w")
    assert reg.recover_tft_artifacts(cfg) is True
    meta = json.loads((tmp_path / "ck" / "m.meta.json").read_text(encoding="utf-8"))
    assert meta["quantiles"] == [0.1, 0.5, 0.9]
    assert meta["target"] == "sales"
    assert meta["group_ids"] == ["Store"]


# -----------------------------------------------------------------------
# MLflow integration
# -----------------------------------------------------------------------


def test_fetch_latest_mlflow_metrics_empty_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    assert reg.fetch_latest_mlflow_metrics("exp", tracking_uri="file:///x") == {}

    _fake_mlflow(monkeypatch, experiments={}, runs_by_exp={})
    assert reg.fetch_latest_mlflow_metrics("exp", tracking_uri="file:///x") == {}

    exp = types.SimpleNamespace(experiment_id="7")
    _fake_mlflow(monkeypatch, experiments={"exp": exp}, runs_by_exp={"7": []})
    assert reg.fetch_latest_mlflow_metrics("exp", tracking_uri="file:///x") == {}


def test_fetch_latest_mlflow_metrics_canonicalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    exp = types.SimpleNamespace(experiment_id="7")
    calls = _fake_mlflow(
        monkeypatch,
        experiments={"exp": exp},
        runs_by_exp={"7": [_run({"val_smape": 12.5, "rmse": 3, "custom": 1})]},
    )

    metrics = reg.fetch_latest_mlflow_metrics("exp", tracking_uri="file:///mlruns")

    assert calls["tracking_uri"] == "file:///mlruns"
    assert calls["search"][0][1]["max_results"] == 1
    assert metrics["smape"] == 12.5
    assert metrics["val_smape"] == 12.5
    assert metrics["rmse"] == 3.0
    assert metrics["coverage_80pct"] == 0.0
    assert metrics["baseline_rmse"] == 0.0
    assert metrics["custom"] == 1.0


def test_fetch_latest_mlflow_metrics_reads_uri_from_config_and_handles_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_rl_config(monkeypatch, "http://mlflow.local:5000")
    exp = types.SimpleNamespace(experiment_id="1")
    calls = _fake_mlflow(
        monkeypatch,
        experiments={"exp": exp},
        runs_by_exp={},
        search_error=RuntimeError("server down"),
    )

    assert reg.fetch_latest_mlflow_metrics("exp") == {}
    assert calls["tracking_uri"] == "http://mlflow.local:5000"


def test_tracking_uri_reachable_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    assert reg._tracking_uri_reachable(None) is True
    assert reg._tracking_uri_reachable("") is True
    assert reg._tracking_uri_reachable("file:///tmp/mlruns") is True
    assert reg._tracking_uri_reachable("sqlite:///mlflow.db") is True
    assert reg._tracking_uri_reachable("http://") is False

    targets: list[tuple[str, int]] = []

    def _ok(address: tuple[str, int], timeout: float) -> contextlib.nullcontext:
        targets.append(address)
        return contextlib.nullcontext()

    monkeypatch.setattr(reg.socket, "create_connection", _ok)
    assert reg._tracking_uri_reachable("https://mlflow.example") is True
    assert reg._tracking_uri_reachable("http://mlflow.example:5000") is True
    assert targets == [("mlflow.example", 443), ("mlflow.example", 5000)]

    def _refuse(address: tuple[str, int], timeout: float) -> None:
        raise OSError("refused")

    monkeypatch.setattr(reg.socket, "create_connection", _refuse)
    assert reg._tracking_uri_reachable("http://mlflow.example") is False


def test_fetch_live_kpi_payload_unavailable_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    assert reg.fetch_live_kpi_payload() is None

    # No experiments at all.
    _fake_mlflow(monkeypatch, experiments={}, runs_by_exp={})
    assert reg.fetch_live_kpi_payload() is None

    # Query failure.
    exp = types.SimpleNamespace(experiment_id="1")
    _fake_mlflow(
        monkeypatch,
        experiments={"retail_tft_forecasting": exp},
        runs_by_exp={},
        search_error=OSError("boom"),
    )
    assert reg.fetch_live_kpi_payload() is None

    # Runs exist but both SMAPE and service level are zero/missing.
    _fake_mlflow(
        monkeypatch,
        experiments={"retail_rl_inventory": exp},
        runs_by_exp={"1": [_run({"total_cost": 5.0})]},
    )
    assert reg.fetch_live_kpi_payload() is None


def test_fetch_live_kpi_payload_unreachable_tracking_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_rl_config(monkeypatch, "http://mlflow.local:5000")
    monkeypatch.setattr(reg, "_tracking_uri_reachable", lambda uri: False)
    calls = _fake_mlflow(monkeypatch, experiments={}, runs_by_exp={})
    assert reg.fetch_live_kpi_payload() is None
    assert calls["tracking_uri"] is None


def test_fetch_live_kpi_payload_merges_forecast_and_rl_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_rl_config(monkeypatch, "http://mlflow.local:5000")
    monkeypatch.setattr(reg, "_tracking_uri_reachable", lambda uri: True)
    calls = _fake_mlflow(
        monkeypatch,
        experiments={
            "retail_tft_forecasting": types.SimpleNamespace(experiment_id="f"),
            "retail_rl_inventory": types.SimpleNamespace(experiment_id="r"),
        },
        runs_by_exp={
            "f": [
                _run(
                    {"val_smape": 10, "baseline_smape": 14, "rmse": 2, "coverage_80pct": 0.8},
                    run_id="tft-run-42",
                )
            ],
            "r": [_run({"mean_service_level": 0.95, "fill_rate": 0.9, "total_cost": 500})],
        },
    )

    payload = reg.fetch_live_kpi_payload()

    assert calls["tracking_uri"] == "http://mlflow.local:5000"
    assert payload is not None
    assert payload["smape"] == 10.0
    assert payload["baseline_smape"] == 14.0
    assert payload["smape_uplift_vs_baseline"] == 4.0
    assert payload["rmse"] == 2.0
    assert payload["baseline_rmse"] == 2.0
    assert payload["coverage_80pct"] == 0.8
    assert payload["service_level"] == 0.95
    assert payload["fill_rate"] == 0.9
    assert payload["total_cost"] == 500.0
    assert payload["model_version"] == "tft-run-42"


def test_fetch_live_kpi_payload_rl_only_uses_default_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_mlflow(
        monkeypatch,
        experiments={"retail_rl_inventory": types.SimpleNamespace(experiment_id="r")},
        runs_by_exp={"r": [_run({"service_level": 0.9})]},
    )
    payload = reg.fetch_live_kpi_payload()
    assert payload is not None
    assert payload["model_version"] == "mlflow_latest"
    assert payload["smape"] == 0.0
    assert payload["service_level"] == 0.9


# -----------------------------------------------------------------------
# RL simulation
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent",
    [None, object(), types.SimpleNamespace(predict=lambda obs, deterministic=True: 0)],
    ids=["no_agent", "agent_without_predict", "agent_without_env"],
)
def test_simulation_returns_empty_without_usable_agent(
    monkeypatch: pytest.MonkeyPatch, agent: Any
) -> None:
    """No placeholder KPIs are invented when there is no usable RL agent."""
    monkeypatch.setattr(reg, "get_rl_agent", lambda: agent)
    assert reg.run_rl_simulation_metrics(n_days=5, risk_aversion=0.3) == {}
    assert reg.get_latest_simulation() == {}


def test_simulation_without_tft_uses_agent_env_demand(
    monkeypatch: pytest.MonkeyPatch, fake_env: type[_FakeEnv]
) -> None:
    base_env = types.SimpleNamespace(
        demand_series=[5.0, 6.0, 7.0], forecast_fn="ffn", store_open_series=None
    )
    agent = _Agent(base_env)
    monkeypatch.setattr(reg, "get_rl_agent", lambda: agent)
    monkeypatch.setattr(reg, "get_tft_model", lambda: (None, None, None))

    result = reg.run_rl_simulation_metrics(n_days=10, risk_aversion=0.7, request_id="r1")

    assert result == {
        "service_level": 0.9,
        "fill_rate": 0.8,
        "total_cost": 102.0,
        "mc_runs": 1.0,
        "risk_aversion": pytest.approx(0.7),
        "model_version": "runtime",
    }
    env = fake_env.instances[-1]
    np.testing.assert_array_equal(env.demand_series, [5.0, 6.0, 7.0])
    assert env.kwargs["reward_type"] == "multi_objective"
    assert env.kwargs["forecast_fn"] == "ffn"
    assert agent.calls == 2  # terminated after second step
    assert reg.get_latest_simulation() == result
    assert reg._CACHE_LOADED_AT["simulation"] is not None


def test_simulation_without_tft_falls_back_to_reference_demand(
    monkeypatch: pytest.MonkeyPatch, fake_env: type[_FakeEnv]
) -> None:
    agent = _Agent(types.SimpleNamespace())
    requested: list[int] = []

    def _demand(min_length: int = 365) -> np.ndarray:
        requested.append(min_length)
        return np.full(3, 4.0)

    monkeypatch.setattr(reg, "get_rl_agent", lambda: agent)
    monkeypatch.setattr(reg, "get_tft_model", lambda: (None, None, None))
    monkeypatch.setattr(reg, "get_reference_demand_series", _demand)

    result = reg.run_rl_simulation_metrics(n_days=400)

    assert requested == [400]
    assert fake_env.instances[-1].kwargs["reward_type"] == "auto"
    assert result["risk_aversion"] == 0.5


@pytest.mark.parametrize(
    ("reference_df", "n_mc", "expected_store"),
    [
        (pd.DataFrame({"Store": [5, 2, 5]}), 2, 2),  # lowest store id in the reference data
        (pd.DataFrame({"sales": [1.0]}), 1, 1),  # no Store column -> store 1
    ],
    ids=["first_store", "default_store"],
)
def test_simulation_monte_carlo_with_tft_forecast(
    monkeypatch: pytest.MonkeyPatch,
    fake_env: type[_FakeEnv],
    reference_df: pd.DataFrame,
    n_mc: int,
    expected_store: int,
) -> None:
    seen: list[int] = []

    class _Model:
        def predict_for_store(self, store_id: int, horizon: int, **kwargs: Any) -> Any:
            seen.append(store_id)
            return FakeTensor(np.tile([[8.0, 10.0, 12.0]], (horizon, 1))), []

    agent = _Agent(types.SimpleNamespace(horizon=3, episode_length=4))
    monkeypatch.setattr(reg, "get_rl_agent", lambda: agent)
    monkeypatch.setattr(reg, "get_tft_model", lambda: (_Model(), object(), reference_df))
    monkeypatch.setattr(reg, "get_tft_quantile_indices", lambda: {"p10": 0, "p50": 1, "p90": 2})

    result = reg.run_rl_simulation_metrics(n_days=3, n_mc=n_mc)

    assert seen == [expected_store]
    assert result["mc_runs"] == float(n_mc)
    assert result["service_level"] == pytest.approx(0.9)
    assert result["fill_rate"] == pytest.approx(0.8)
    assert result["total_cost"] == pytest.approx(102.0)
    assert len(fake_env.instances) == n_mc
    assert all(len(env.demand_series) == 3 for env in fake_env.instances)
    assert reg.get_latest_simulation()["mc_runs"] == float(n_mc)


def test_simulation_forecast_failure_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Model:
        def predict_for_store(self, **kwargs: Any) -> Any:
            raise ValueError("bad store")

    monkeypatch.setattr(reg, "get_rl_agent", lambda: _Agent(types.SimpleNamespace()))
    monkeypatch.setattr(reg, "get_tft_model", lambda: (_Model(), object(), pd.DataFrame()))

    assert reg.run_rl_simulation_metrics(n_days=3, store_id=4) == {}
    assert reg.get_latest_simulation() == {}


def test_sim_env_pool_reuse_and_rebuild(fake_env: type[_FakeEnv]) -> None:
    kwargs = dict(
        config_path="cfg.yaml",
        risk_aversion=0.5,
        reward_type="auto",
        pool_size=2,
        default_days=0,
        env_class_id=1,
    )
    first = reg._get_or_create_sim_env_pool(**kwargs)
    assert len(first) == 2
    assert len(first[0].demand_series) == 1  # max(default_days, 1)
    assert reg._get_or_create_sim_env_pool(**kwargs) is first
    assert len(fake_env.instances) == 2

    rebuilt = reg._get_or_create_sim_env_pool(**{**kwargs, "risk_aversion": 0.9})
    assert rebuilt is not first
    assert rebuilt[0].kwargs["risk_aversion"] == 0.9
    assert len(fake_env.instances) == 4


def test_run_rl_simulation_metrics_async_shapes_results(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []

    def _sim(*args: Any) -> dict[str, float]:
        calls.append(args)
        store_id = args[2]
        return {"service_level": 0.9, "fill_rate": 0.8, "total_cost": 50.0 * store_id}

    monkeypatch.setattr(reg, "run_rl_simulation_metrics", _sim)

    out = asyncio.run(
        reg.run_rl_simulation_metrics_async([1, 2], n_days=7, risk_aversion=0.3, n_mc=4)
    )

    assert out == [
        {
            "store_id": float(store_id),
            "service_level": 0.9,
            "fill_rate": 0.8,
            "total_cost": 50.0 * store_id,
            "risk_aversion": 0.3,
        }
        for store_id in (1, 2)
    ]
    assert sorted(c[2] for c in calls) == [1, 2]
    assert all(c[0] == 7 and c[4] == 4 for c in calls)

    # Without an RL agent the simulation is empty and the batch reports it, not zeros.
    monkeypatch.setattr(reg, "run_rl_simulation_metrics", lambda *args: {})
    with pytest.raises(reg.SimulationUnavailableError, match="no RL agent loaded"):
        asyncio.run(reg.run_rl_simulation_metrics_async([1], n_days=7))


# -----------------------------------------------------------------------
# Reference demand series
# -----------------------------------------------------------------------


def test_reference_demand_from_registry_state_tiles_to_min_length() -> None:
    reg._atomic_swap(tft_reference_df=pd.DataFrame({"Sales": [1.0, 2.0, np.nan]}))
    np.testing.assert_array_equal(
        reg.get_reference_demand_series(min_length=5), [1.0, 2.0, 1.0, 2.0, 1.0]
    )


def test_reference_demand_from_parquet_truncates_to_min_length(tmp_path: Path) -> None:
    reg._atomic_swap(tft_reference_df=pd.DataFrame({"other": [1.0]}))
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame({"sales": [4.0, 5.0, 6.0]}).to_parquet(processed / "tft_reference.parquet")
    np.testing.assert_array_equal(reg.get_reference_demand_series(min_length=2), [4.0, 5.0])


def test_reference_demand_from_train_csv(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "tft_reference.parquet").write_bytes(b"not a parquet file")
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "train.csv").write_text("Store,Sales\n1,10\n2,20\n", encoding="utf-8")
    np.testing.assert_array_equal(reg.get_reference_demand_series(min_length=3), [10, 20, 10])


def test_reference_demand_raises_when_no_source(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "train.csv").write_text("Store,Customers\n1,10\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        reg.get_reference_demand_series()


# -----------------------------------------------------------------------
# Lazy model getters (shared TFT / RL behaviour)
# -----------------------------------------------------------------------


_LAZY_GETTERS = pytest.mark.parametrize(
    ("getter", "loader", "cache_key", "fields"),
    [
        (
            "get_tft_model",
            "load_tft_from_artifacts",
            "tft",
            ("tft_model", "tft_dataset", "tft_reference_df"),
        ),
        ("get_rl_agent", "load_rl_agent", "rl", ("rl_agent",)),
    ],
    ids=["tft", "rl"],
)


def _install_loader(
    monkeypatch: pytest.MonkeyPatch, loader: str, fields: tuple[str, ...], delay: float = 0.0
) -> tuple[list[dict[str, Any]], Any]:
    """Patch ``loader`` to set sentinel state; return (calls, expected getter result)."""
    sentinels = {name: object() for name in fields}
    calls: list[dict[str, Any]] = []

    def _load(**kwargs: Any) -> None:
        calls.append(kwargs)
        time.sleep(delay)
        reg._atomic_swap(**sentinels)

    monkeypatch.setattr(reg, loader, _load)
    monkeypatch.setattr(reg, "recover_tft_artifacts", lambda cfg: False)
    monkeypatch.setattr(reg, "get_reference_demand_series", lambda min_length=365: np.ones(2))
    expected = tuple(sentinels.values()) if len(fields) > 1 else sentinels[fields[0]]
    return calls, expected


@_LAZY_GETTERS
def test_concurrent_first_requests_load_model_once(
    monkeypatch: pytest.MonkeyPatch,
    getter: str,
    loader: str,
    cache_key: str,
    fields: tuple[str, ...],
) -> None:
    """Regression: the load lock is held for the whole lazy load, so racing callers load once."""
    calls, expected = _install_loader(monkeypatch, loader, fields, delay=0.05)
    n_threads = 8
    barrier = threading.Barrier(n_threads)

    def _call(_: int) -> Any:
        barrier.wait()
        return getattr(reg, getter)()

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        results = list(pool.map(_call, range(n_threads)))

    assert len(calls) == 1
    assert results == [expected] * n_threads
    # Later calls are served from the cached state without reloading.
    assert getattr(reg, getter)() == expected
    assert len(calls) == 1
    assert reg._CACHE_LOADED_AT[cache_key] is not None


@_LAZY_GETTERS
def test_lazy_getters_reload_after_ttl_expiry(
    monkeypatch: pytest.MonkeyPatch,
    getter: str,
    loader: str,
    cache_key: str,
    fields: tuple[str, ...],
) -> None:
    calls, expected = _install_loader(monkeypatch, loader, fields)
    reg._atomic_swap(**{name: "stale" for name in fields})
    reg._CACHE_LOADED_AT[cache_key] = time.time() - 10_000

    assert getattr(reg, getter)() == expected
    assert len(calls) == 1
    assert reg._CACHE_LOADED_AT[cache_key] == pytest.approx(time.time(), abs=5)


# -----------------------------------------------------------------------
# TFT loading
# -----------------------------------------------------------------------


class _FakeForecaster:
    def __init__(self, config_path: str, dataset: Any) -> None:
        self.config_path = config_path
        self.dataset = dataset
        self.built = False
        self.loaded_from: str | None = None

    def build(self) -> None:
        self.built = True

    @classmethod
    def load_any(cls, path: str, config_path: str, dataset: Any) -> _FakeForecaster:
        inst = cls(config_path, dataset)
        inst.loaded_from = path
        return inst


@pytest.fixture()
def fake_tft_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_module(
        monkeypatch,
        "src.models.tft_dataset",
        build_tft_dataset=lambda df, config_path: ("dataset", len(df), config_path),
    )
    _fake_module(monkeypatch, "src.models.tft_model", TFTForecaster=_FakeForecaster)


@pytest.mark.usefixtures("fake_tft_modules")
def test_load_tft_model_from_checkpoint_adds_store_alias(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        reg.load_tft_model(checkpoint_path=str(tmp_path / "missing.pt"))

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"w")
    df = pd.DataFrame({"store": [1, 2], "sales": [3.0, 4.0]})

    reg.load_tft_model(checkpoint_path=str(ckpt), dataset_df=df, config_path="cfg.yaml")

    model, dataset, reference = (
        reg._STATE.tft_model,
        reg._STATE.tft_dataset,
        reg._STATE.tft_reference_df,
    )
    assert isinstance(model, _FakeForecaster) and model.loaded_from == str(ckpt)
    assert dataset == ("dataset", 2, "cfg.yaml")
    assert list(reference["Store"]) == [1, 2]
    assert "Store" not in df.columns  # original frame untouched


@pytest.mark.usefixtures("fake_tft_modules")
def test_load_tft_model_builds_fresh_model() -> None:
    ref = pd.DataFrame({"Store": [7]})
    reg.load_tft_model(reference_df=ref, config_path="cfg.yaml")

    assert reg._STATE.tft_model.built is True
    assert reg._STATE.tft_dataset is None
    assert list(reg._STATE.tft_reference_df["store"]) == [7]


def test_load_tft_from_artifacts_missing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovered: list[dict[str, str]] = []
    monkeypatch.setattr(reg, "recover_tft_artifacts", lambda cfg: recovered.append(cfg))

    with pytest.raises(FileNotFoundError, match="Missing TFT checkpoint"):
        reg.load_tft_from_artifacts("c.pt", "r.parquet", "rt.yaml")
    assert recovered == [
        {"tft_checkpoint": "c.pt", "processed_data": "r.parquet", "runtime_config": "rt.yaml"}
    ]

    (tmp_path / "c.pt").write_bytes(b"w")
    with pytest.raises(FileNotFoundError, match="Missing TFT reference dataset"):
        reg.load_tft_from_artifacts("c.pt", "r.parquet", "rt.yaml")


def test_load_tft_from_artifacts_uses_fingerprint_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(reg, "load_tft_model", lambda **kwargs: captured.append(kwargs))
    (tmp_path / "c.pt").write_bytes(b"w")
    df = pd.DataFrame({"Store": [1, 2], "sales": [1.0, 2.0]})
    df.to_parquet(tmp_path / "r.parquet")
    (tmp_path / "alt.yaml").write_text("tft: {}\n", encoding="utf-8")
    (tmp_path / "fp.json").write_text(json.dumps({"config_path": "alt.yaml"}), encoding="utf-8")

    reg.load_tft_from_artifacts("c.pt", "r.parquet", "rt.yaml", fingerprint_path="fp.json")
    assert captured[-1]["config_path"] == "alt.yaml"
    assert captured[-1]["checkpoint_path"] == "c.pt"
    pd.testing.assert_frame_equal(captured[-1]["dataset_df"], df)

    (tmp_path / "fp.json").write_text("{not json", encoding="utf-8")
    reg.load_tft_from_artifacts("c.pt", "r.parquet", "rt.yaml", fingerprint_path="fp.json")
    assert captured[-1]["config_path"] == "rt.yaml"


def test_get_tft_model_lazy_loads_latest_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "checkpoints" / "tft").mkdir(parents=True)
    (tmp_path / "checkpoints" / "tft" / "alt.pt").write_bytes(b"w")
    recovered: list[Any] = []
    loads: list[dict[str, Any]] = []

    def _load(**kwargs: Any) -> None:
        loads.append(kwargs)
        reg._atomic_swap(tft_model="model", tft_dataset="ds", tft_reference_df="ref")

    monkeypatch.setattr(reg, "recover_tft_artifacts", lambda cfg: recovered.append(cfg))
    monkeypatch.setattr(reg, "load_tft_from_artifacts", _load)

    assert reg.get_tft_model() == ("model", "ds", "ref")
    assert Path(loads[0]["checkpoint_path"]).name == "alt.pt"
    assert loads[0]["reference_path"] == "data/processed/tft_reference.parquet"
    assert len(recovered) == 1  # reference parquet missing triggers one recovery


def test_get_tft_model_retries_once_when_load_leaves_model_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[Any] = []
    recovered: list[Any] = []
    monkeypatch.setattr(reg, "load_tft_from_artifacts", lambda **kw: loads.append(kw))
    monkeypatch.setattr(reg, "recover_tft_artifacts", lambda cfg: recovered.append(cfg))

    assert reg.get_tft_model() == (None, None, None)
    assert len(loads) == 2
    assert len(recovered) == 3  # pre-load recovery on both attempts + retry recovery
    assert reg._CACHE_LOADED_AT["tft"] is None


@pytest.mark.parametrize("readable", [True, False], ids=["readable", "corrupt"])
def test_get_tft_model_falls_back_to_reference_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readable: bool
) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    df = pd.DataFrame({"Store": [1], "sales": [3.0]})
    if readable:
        df.to_parquet(processed / "tft_reference.parquet")
    else:
        (processed / "tft_reference.parquet").write_bytes(b"corrupt")

    def _missing(**kwargs: Any) -> None:
        raise FileNotFoundError("no checkpoint")

    monkeypatch.setattr(reg, "recover_tft_artifacts", lambda cfg: False)
    monkeypatch.setattr(reg, "load_tft_from_artifacts", _missing)

    model, dataset, reference = reg.get_tft_model()

    assert model is None and dataset is None
    assert reg._STATE.tft_reference_df is reference
    if readable:
        pd.testing.assert_frame_equal(reference, df)
    else:
        assert reference is None


def test_get_tft_quantile_indices_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[float]] = []

    def _resolve(quantiles: list[float]) -> dict[str, int]:
        seen.append(quantiles)
        return {"p10": 0, "p50": 1, "p90": 2}

    monkeypatch.setattr(reg, "resolve_quantile_indices", _resolve)

    # 1) From loaded model config.
    model = types.SimpleNamespace(config={"quantiles": [0.1, 0.5, 0.9]})
    monkeypatch.setattr(reg, "get_tft_model", lambda: (model, None, None))
    assert reg.get_tft_quantile_indices() == {"p10": 0, "p50": 1, "p90": 2}
    assert seen[-1] == [0.1, 0.5, 0.9]

    # 2) No model and no runtime config -> hard-coded defaults.
    monkeypatch.setattr(reg, "get_tft_model", lambda: (None, None, None))
    assert reg.get_tft_quantile_indices() == {"p10": 1, "p50": 3, "p90": 5}

    # 3) Runtime config file present.
    Path("data/processed").mkdir(parents=True)
    Path("data/processed/tft_runtime_config.yaml").write_text("tft: {}\n", encoding="utf-8")
    monkeypatch.setattr(reg, "validate_config_schema", lambda *a, **k: {})
    monkeypatch.setattr(
        reg, "load_yaml_config", lambda *a, **k: {"tft": {"quantiles": [0.02, 0.1, 0.5, 0.9]}}
    )
    assert reg.get_tft_quantile_indices() == {"p10": 0, "p50": 1, "p90": 2}
    assert seen[-1] == [0.02, 0.1, 0.5, 0.9]


def test_warmup_forecast_cache_skips_without_model_or_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reg, "get_tft_model", lambda: (None, None, None))
    assert reg.warmup_forecast_cache() == 0

    monkeypatch.setattr(
        reg, "get_tft_model", lambda: (object(), object(), pd.DataFrame({"x": [1]}))
    )
    assert reg.warmup_forecast_cache() == 0

    monkeypatch.setattr(
        reg, "get_tft_model", lambda: (object(), object(), pd.DataFrame({"Store": [1]}))
    )
    assert reg.warmup_forecast_cache(top_n=0) == 0


def test_warmup_forecast_cache_populates_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Model:
        def predict_for_store(self, store_id: int, horizon: int, **kwargs: Any) -> Any:
            if store_id == 2:
                raise KeyError("unknown store")
            preds = np.array([[9.0, 10.0, 11.0]] * horizon) * store_id
            dates = list(pd.date_range("2025-01-01", periods=horizon))
            return FakeTensor(preds), dates

    ref = pd.DataFrame({"Store": [3, 1, 2, 2, 4]})
    monkeypatch.setattr(reg, "get_tft_model", lambda: (_Model(), object(), ref))
    monkeypatch.setattr(reg, "get_tft_quantile_indices", lambda: {"p10": 0, "p50": 1, "p90": 2})

    assert reg.warmup_forecast_cache(top_n=3, horizon=2) == 2

    cached = reg.get_cached_forecast(store_id=3, horizon=2)
    assert cached == {
        "store_id": 3,
        "horizon": 2,
        "dates": ["2025-01-01", "2025-01-02"],
        "p10": [27.0, 27.0],
        "p50": [30.0, 30.0],
        "p90": [33.0, 33.0],
        "model_version": "warmup_cache",
    }
    assert reg.get_cached_forecast(store_id=1, horizon=2) is not None
    assert reg.get_cached_forecast(store_id=2, horizon=2) is None
    assert reg.get_cached_forecast(store_id=4, horizon=2) is None  # beyond top_n


# -----------------------------------------------------------------------
# RL agent loading
# -----------------------------------------------------------------------


class _FakeRLAgent:
    def __init__(self, config_path: str, env: Any) -> None:
        self.config_path = config_path
        self.env = env
        self.built = False
        self.model: Any = None

    def build(self) -> None:
        self.built = True


@pytest.fixture()
def fake_rl_modules(monkeypatch: pytest.MonkeyPatch, fake_env: type[_FakeEnv]) -> None:
    _fake_module(monkeypatch, "src.rl.agent", RLInventoryAgent=_FakeRLAgent)


@pytest.mark.usefixtures("fake_rl_modules")
def test_load_rl_agent_builds_and_stores_agent() -> None:
    demand = np.full(4, 2.0)
    reg.load_rl_agent(config_path="rl.yaml", demand_series=demand)

    agent = reg._STATE.rl_agent
    assert isinstance(agent, _FakeRLAgent)
    assert agent.built is True
    assert agent.env.config_path == "rl.yaml"
    assert agent.env.demand_series is demand


@pytest.mark.usefixtures("fake_rl_modules")
@pytest.mark.parametrize("load_ok", [True, False], ids=["loads", "corrupt"])
def test_get_rl_agent_from_latest_ppo_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_ok: bool
) -> None:
    (tmp_path / "checkpoints" / "rl").mkdir(parents=True)
    (tmp_path / "checkpoints" / "rl" / "other.zip").write_bytes(b"z")

    class _PPO:
        @staticmethod
        def load(path: str) -> str:
            if not load_ok:
                raise RuntimeError("corrupt zip")
            return f"ppo:{Path(path).name}"

    _fake_module(monkeypatch, "stable_baselines3", PPO=_PPO)
    monkeypatch.setattr(reg, "get_reference_demand_series", lambda min_length=365: np.ones(3))

    agent = reg.get_rl_agent()

    if load_ok:
        assert isinstance(agent, _FakeRLAgent)
        assert agent.model == "ppo:other.zip"
        assert reg._CACHE_LOADED_AT["rl"] is not None
    else:
        assert agent is None
        assert reg._CACHE_LOADED_AT["rl"] is None


def test_get_rl_agent_without_any_data_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reg, "load_rl_agent", lambda **kw: pytest.fail("needs demand first"))
    # Empty cwd -> get_reference_demand_series raises FileNotFoundError.
    assert reg.get_rl_agent() is None


# -----------------------------------------------------------------------
# CL metrics
# -----------------------------------------------------------------------


def _write_cl_metrics(tmp_path: Path, content: str) -> Path:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    path = processed / "cl_metrics_latest.json"
    path.write_text(content, encoding="utf-8")
    return path


def test_get_cl_metrics_reads_artifact_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        reg, "fetch_latest_mlflow_metrics", lambda **kw: pytest.fail("mlflow not expected")
    )
    path = _write_cl_metrics(tmp_path, json.dumps({"bwt": -0.2, "strategy": "ewc"}))

    first = reg.get_cl_metrics()
    assert first == {"bwt": -0.2, "strategy": "ewc"}
    first["bwt"] = 123.0
    path.unlink()
    assert reg.get_cl_metrics() == {"bwt": -0.2, "strategy": "ewc"}


def test_get_cl_metrics_invalid_json_and_explicit_path(tmp_path: Path) -> None:
    _write_cl_metrics(tmp_path, "{broken")
    assert reg.get_cl_metrics() == {}

    reg.invalidate_cache("cl_metrics")
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps({"fwt": 0.1}), encoding="utf-8")
    assert reg.get_cl_metrics(path=str(custom)) == {"fwt": 0.1}


def test_get_cl_metrics_mlflow_fallback_and_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _mlflow(experiment_name: str) -> dict[str, float]:
        calls.append(experiment_name)
        return {"bwt": 0.05}

    monkeypatch.setattr(reg, "fetch_latest_mlflow_metrics", _mlflow)
    assert reg.get_cl_metrics() == {"bwt": 0.05}
    assert calls == ["retail_cl_training"]

    reg.invalidate_cache("cl_metrics")
    monkeypatch.setattr(reg, "fetch_latest_mlflow_metrics", lambda experiment_name: {})
    assert reg.get_cl_metrics() == {}
    assert reg._STATE.cl_metrics == {}
    assert reg._CACHE_LOADED_AT["cl_metrics"] is not None


def test_get_cl_metrics_reloads_after_ttl(tmp_path: Path) -> None:
    _write_cl_metrics(tmp_path, json.dumps({"bwt": 0.3}))
    reg._atomic_swap(cl_metrics={"bwt": -9.0})
    reg._CACHE_LOADED_AT["cl_metrics"] = time.time() - 10_000
    assert reg.get_cl_metrics() == {"bwt": 0.3}


# -----------------------------------------------------------------------
# Cache management + readiness
# -----------------------------------------------------------------------


def _populate_state() -> None:
    reg._atomic_swap(
        tft_model="m",
        tft_dataset="d",
        tft_reference_df="r",
        rl_agent="a",
        cl_metrics={"bwt": 0.0},
        latest_simulation={"service_level": 0.9},
        forecast_cache={"1:7": {}},
    )
    reg._CACHE_LOADED_AT.update({"tft": 1.0, "rl": 1.0, "cl_metrics": 1.0, "simulation": 1.0})


def _build_sim_env_pool() -> None:
    reg._get_or_create_sim_env_pool(
        config_path="c",
        risk_aversion=None,
        reward_type="auto",
        pool_size=1,
        default_days=2,
        env_class_id=1,
    )
    assert len(reg._SIM_ENV_POOL) == 1 and reg._SIM_ENV_POOL_SIGNATURE is not None


def test_invalidate_cache_per_key(fake_env: type[_FakeEnv]) -> None:
    _populate_state()

    reg.invalidate_cache("rl")
    assert reg._STATE.rl_agent is None and reg._CACHE_LOADED_AT["rl"] is None
    assert reg._STATE.tft_model == "m"

    reg.invalidate_cache("cl_metrics")
    assert reg._STATE.cl_metrics is None and reg._CACHE_LOADED_AT["cl_metrics"] is None

    _build_sim_env_pool()
    reg.invalidate_cache("simulation")
    assert reg._STATE.latest_simulation == {}
    assert reg._SIM_ENV_POOL == [] and reg._SIM_ENV_POOL_SIGNATURE is None
    assert reg._CACHE_LOADED_AT["simulation"] is None

    reg.invalidate_cache("tft")
    assert reg._STATE.tft_model is None and reg._STATE.tft_reference_df is None
    assert reg._STATE.forecast_cache == {}
    assert reg._CACHE_LOADED_AT["tft"] is None


def test_invalidate_cache_without_key_also_clears_sim_env_pool(fake_env: type[_FakeEnv]) -> None:
    """Regression: a full invalidate (used by clear_registry) drops the pooled sim envs too."""
    _populate_state()
    _build_sim_env_pool()

    reg.invalidate_cache()

    assert reg._SIM_ENV_POOL == []
    assert reg._SIM_ENV_POOL_SIGNATURE is None
    assert all(value is None for value in reg._CACHE_LOADED_AT.values())
    state = reg._STATE
    assert (state.tft_model, state.tft_dataset, state.tft_reference_df) == (None, None, None)
    assert state.rl_agent is None and state.cl_metrics is None
    assert state.latest_simulation == {} and state.forecast_cache == {}


def test_get_latest_simulation_returns_copy() -> None:
    reg._atomic_swap(latest_simulation={"service_level": 0.9})
    snapshot = reg.get_latest_simulation()
    snapshot["service_level"] = 0.0
    assert reg.get_latest_simulation() == {"service_level": 0.9}


def _create_tft_artifacts() -> None:
    Path("checkpoints/tft").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/tft/tft_latest.pt").write_bytes(b"w")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/processed/tft_reference.parquet").write_bytes(b"x")


def test_model_registry_ready_missing_tft_artifacts() -> None:
    assert reg.model_registry_ready() == (False, "TFT artifacts missing")


def test_model_registry_ready_recovers_then_requires_rl_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _recover(cfg: dict[str, Any]) -> bool:
        _create_tft_artifacts()
        return True

    monkeypatch.setattr(reg, "recover_tft_artifacts", _recover)
    assert reg.model_registry_ready() == (False, "RL config missing")


def test_model_registry_ready_ok_even_if_mlflow_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_tft_artifacts()
    _write_rl_config(monkeypatch, "http://mlflow.local:5000")
    probed: list[str] = []

    def _unreachable(uri: str) -> bool:
        probed.append(uri)
        return False

    monkeypatch.setattr(reg, "_tracking_uri_reachable", _unreachable)
    assert reg.model_registry_ready() == (True, "ok")
    assert probed == ["http://mlflow.local:5000"]

    def _bad_yaml(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("bad yaml")

    probed.clear()
    monkeypatch.setattr(reg, "load_yaml_config", _bad_yaml)
    assert reg.model_registry_ready() == (True, "ok")
    assert probed == []
