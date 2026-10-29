"""Unit tests for orchestration helpers and the optional Prefect flow builder (fake prefect)."""

from __future__ import annotations

import sys
import types

import pandas as pd

from src.orchestration import prefect_flows


def test_load_data_uses_rossmann_loader(monkeypatch) -> None:
    """load_data() constructs the loader with data_dir and returns its output."""
    seen: list[str] = []

    class _Loader:
        def __init__(self, data_dir: str) -> None:
            seen.append(data_dir)

        def load(self):  # noqa: ANN201
            return "train", "test"

    monkeypatch.setattr(prefect_flows, "RossmannDataLoader", _Loader)
    assert prefect_flows.load_data("some/dir") == ("train", "test")
    assert seen == ["some/dir"]


def test_preprocess_and_engineer_delegate(monkeypatch) -> None:
    """Preprocess and feature steps call their transformers on the input frame."""

    class _Pre:
        def fit_transform(self, df):  # noqa: ANN001, ANN201
            return df.assign(pre=1)

    class _Fe:
        def transform(self, df):  # noqa: ANN001, ANN201
            return df.assign(feat=df["a"] * 2)

    monkeypatch.setattr(prefect_flows, "RossmannPreprocessor", _Pre)
    monkeypatch.setattr(prefect_flows, "FeatureEngineer", _Fe)
    frame = pd.DataFrame({"a": [1, 2]})

    pre = prefect_flows.preprocess_data(frame)
    feat = prefect_flows.engineer_features(frame)

    assert pre["pre"].tolist() == [1, 1]
    assert feat["feat"].tolist() == [2, 4]


def test_run_retail_pipeline_counts_columns(monkeypatch) -> None:
    """Pipeline summary reports rows for each stage and feature column count."""
    train = pd.DataFrame({"a": [1, 2, 3]})
    test = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(prefect_flows, "load_data", lambda data_dir="data/raw": (train, test))
    monkeypatch.setattr(prefect_flows, "preprocess_data", lambda df: df.iloc[:2])
    monkeypatch.setattr(prefect_flows, "engineer_features", lambda df: df.assign(b=0, c=1))
    assert prefect_flows.run_retail_pipeline() == {
        "train_rows": 3,
        "test_rows": 1,
        "feature_rows": 2,
        "feature_columns": 3,
    }


def _fake_prefect(registry: dict) -> types.ModuleType:
    module = types.ModuleType("prefect")

    def task(fn):  # noqa: ANN001, ANN202
        registry.setdefault("tasks", []).append(fn.__name__)
        return fn

    def flow(name):  # noqa: ANN001, ANN202
        def _decorate(fn):  # noqa: ANN001, ANN202
            registry["flow_name"] = name
            return fn

        return _decorate

    module.task = task
    module.flow = flow
    return module


def test_build_prefect_flow_wires_tasks(monkeypatch) -> None:
    """With prefect available, the flow chains load -> preprocess -> engineer."""
    registry: dict = {}
    monkeypatch.setitem(sys.modules, "prefect", _fake_prefect(registry))
    calls: list = []
    train = pd.DataFrame({"a": [1, 2, 3, 4]})
    test = pd.DataFrame({"a": [1, 2]})

    def _load(data_dir="data/raw"):  # noqa: ANN001, ANN202
        calls.append(("load", data_dir))
        return train, test

    def _pre(df):  # noqa: ANN001, ANN202
        calls.append(("pre", len(df)))
        return df.iloc[:3]

    def _eng(df):  # noqa: ANN001, ANN202
        calls.append(("eng", len(df)))
        return df.iloc[:1]

    monkeypatch.setattr(prefect_flows, "load_data", _load)
    monkeypatch.setattr(prefect_flows, "preprocess_data", _pre)
    monkeypatch.setattr(prefect_flows, "engineer_features", _eng)

    pipeline = prefect_flows.build_prefect_flow()

    assert registry["flow_name"] == "retail-inventory-pipeline"
    assert registry["tasks"] == ["_load_data_task", "_preprocess_task", "_engineer_task"]
    assert pipeline(data_dir="x/raw") == {"train_rows": 4, "test_rows": 2, "feature_rows": 1}
    assert calls == [("load", "x/raw"), ("pre", 4), ("eng", 3)]


def test_build_prefect_flow_returns_none_without_prefect(monkeypatch) -> None:
    """Flow builder degrades gracefully to None when Prefect cannot be imported."""
    monkeypatch.setitem(sys.modules, "prefect", None)
    assert prefect_flows.build_prefect_flow() is None
