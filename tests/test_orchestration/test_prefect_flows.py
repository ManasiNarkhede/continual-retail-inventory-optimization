"""Tests for optional orchestration flow helpers."""

from __future__ import annotations

import pandas as pd

from src.orchestration.prefect_flows import build_prefect_flow, run_retail_pipeline


def test_run_retail_pipeline_returns_summary(monkeypatch) -> None:
    """Pipeline helper should return simple row/column summary payload."""
    train_df = pd.DataFrame(
        {"Store": [1], "Date": pd.to_datetime(["2015-01-01"]), "Sales": [100.0], "Open": [1]}
    )
    test_df = pd.DataFrame({"Store": [1], "Date": pd.to_datetime(["2015-01-02"]), "Open": [1]})
    feature_df = pd.DataFrame(
        {
            "Store": [1],
            "Date": pd.to_datetime(["2015-01-01"]),
            "sales": [100.0],
            "time_idx": [0],
            "rolling_mean_7d": [100.0],
            "rolling_std_7d": [0.0],
        }
    )

    monkeypatch.setattr(
        "src.orchestration.prefect_flows.load_data", lambda data_dir="data/raw": (train_df, test_df)
    )
    monkeypatch.setattr("src.orchestration.prefect_flows.preprocess_data", lambda df: df)
    monkeypatch.setattr("src.orchestration.prefect_flows.engineer_features", lambda df: feature_df)

    result = run_retail_pipeline()
    assert result["train_rows"] == 1
    assert result["feature_rows"] == 1


def test_build_prefect_flow_returns_none_without_prefect(monkeypatch) -> None:
    """Flow builder should degrade gracefully when Prefect is unavailable."""
    import builtins

    orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "prefect":
            raise ImportError("prefect not installed")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert build_prefect_flow() is None
