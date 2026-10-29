"""Unit tests for AvalancheExperienceBuilder using stubbed Avalanche modules.

Real Avalanche is replaced through ``sys.modules`` so these tests exercise the
builder's month-splitting logic deterministically and without importing the
heavy (and, on some Windows runtimes, crash-prone) Avalanche package tree.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from src.data.experience_builder import AvalancheExperienceBuilder, _load_benchmark_factory

_PACKAGES = ("avalanche", "avalanche.benchmarks", "avalanche.benchmarks.scenarios")
_UTILS = "avalanche.benchmarks.utils"
_GENERATORS = "avalanche.benchmarks.generators"
_SCENARIO = "avalanche.benchmarks.scenarios.dataset_scenario"


class _FakeAvalancheDataset:
    def __init__(self, datasets: list[Any]) -> None:
        self.datasets = datasets


class _FakeBenchmark:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _module(name: str, is_package: bool = False, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
    if is_package:
        module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_fake_avalanche(
    monkeypatch: pytest.MonkeyPatch,
    *,
    utils: bool = True,
    generators: bool = True,
    scenario: bool = False,
) -> None:
    """Install fake Avalanche packages; absent leaves make ``find_spec`` return None."""
    for name in _PACKAGES:
        monkeypatch.setitem(sys.modules, name, _module(name, is_package=True))
    leaves = {
        _UTILS: (utils, {"AvalancheDataset": _FakeAvalancheDataset}),
        _GENERATORS: (generators, {"dataset_benchmark": _FakeBenchmark}),
        _SCENARIO: (scenario, {"benchmark_from_datasets": _FakeBenchmark}),
    }
    for name, (present, attrs) in leaves.items():
        if present:
            monkeypatch.setitem(sys.modules, name, _module(name, **attrs))
        else:
            monkeypatch.delitem(sys.modules, name, raising=False)


def _daily_frame(start: str, days: int, target: str = "sales") -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Store": 1,
            target: [float(100 * d.month + d.day) for d in dates],
            "feat_a": [float(d.day) for d in dates],
            "feat_b": [float(d.month) for d in dates],
        }
    )


def _collect(dataset: Any) -> list[tuple[list[float], float]]:
    return [(dataset[i][0].tolist(), float(dataset[i][1])) for i in range(len(dataset))]


def _build(tmp_path: Path, df: pd.DataFrame, n: int, **kwargs: Any) -> dict[str, Any]:
    builder = AvalancheExperienceBuilder(n_experiences=n, output_dir=str(tmp_path))
    return builder.build(df=df, **kwargs).kwargs


def _parquets(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in tmp_path.glob("*.parquet"))


# -----------------------------------------------------------------------
# _load_benchmark_factory
# -----------------------------------------------------------------------


def test_factory_prefers_legacy_generators_layout(monkeypatch) -> None:
    _install_fake_avalanche(monkeypatch, generators=True, scenario=True)
    train, test = [TensorDataset(torch.zeros(1))], [TensorDataset(torch.ones(1))]

    benchmark = _load_benchmark_factory()(train, test)

    assert benchmark.kwargs == {"train_datasets": train, "test_datasets": test}


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        ({"utils": False}, "benchmark utilities are unavailable"),
        ({"generators": False, "scenario": False}, "Unsupported avalanche-lib layout"),
    ],
    ids=["utils-missing", "no-builder"],
)
def test_factory_raises_import_error(monkeypatch, layout: dict[str, bool], message: str) -> None:
    _install_fake_avalanche(monkeypatch, **layout)
    with pytest.raises(ImportError, match=message):
        _load_benchmark_factory()


def test_build_on_avalanche_05_wraps_datasets_for_benchmark_from_datasets(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: avalanche 0.5 gets ``train``/``test`` lists of AvalancheDatasets."""
    _install_fake_avalanche(monkeypatch, generators=False, scenario=True)
    torch.manual_seed(0)
    df = _daily_frame("2015-01-01", 90)  # Jan 31, Feb 28, Mar 31 days

    kwargs = _build(tmp_path, df, 3, feature_cols=["feat_a", "feat_b"])

    assert set(kwargs) == {"train", "test"}
    for wrapped in kwargs["train"] + kwargs["test"]:
        assert isinstance(wrapped, _FakeAvalancheDataset)
        assert len(wrapped.datasets) == 1
    # n_val = max(1, int(0.1 * n)): 31 -> 3, 28 -> 2, 31 -> 3
    assert [len(w.datasets[0]) for w in kwargs["train"]] == [28, 26, 28]
    assert [len(w.datasets[0]) for w in kwargs["test"]] == [3, 2, 3]


# -----------------------------------------------------------------------
# AvalancheExperienceBuilder.build (legacy factory)
# -----------------------------------------------------------------------


def test_init_creates_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b"
    builder = AvalancheExperienceBuilder(n_experiences=3, output_dir=str(out))
    assert out.is_dir()
    assert builder.n_experiences == 3
    assert builder.output_dir == out


def test_build_splits_first_n_months_into_train_and_validation(tmp_path: Path, monkeypatch) -> None:
    _install_fake_avalanche(monkeypatch)
    torch.manual_seed(0)
    days = pd.concat([_daily_frame(f"2015-0{m}-01", 10) for m in (1, 2, 3)], ignore_index=True)
    df = days.sample(frac=1.0, random_state=0)  # unsorted input, three months

    kwargs = _build(tmp_path, df, 2, feature_cols=["feat_a", "feat_b"])

    train_sets, test_sets = kwargs["train_datasets"], kwargs["test_datasets"]
    assert [len(d) for d in train_sets] == [9, 9]
    assert [len(d) for d in test_sets] == [1, 1]
    for month, train, test in zip((1, 2), train_sets, test_sets):
        expected = [([float(day), float(month)], float(100 * month + day)) for day in range(1, 11)]
        assert sorted(_collect(train) + _collect(test)) == expected

    assert _parquets(tmp_path) == ["experience_00_2015-01.parquet", "experience_01_2015-02.parquet"]
    january = pd.read_parquet(tmp_path / "experience_00_2015-01.parquet")
    assert len(january) == 10
    assert (january["Date"].dt.month == 1).all()
    assert january["Date"].is_monotonic_increasing


@pytest.mark.parametrize(
    ("column", "requested"), [("Sales", "sales"), ("sales", "Sales")], ids=["upper", "lower"]
)
def test_build_falls_back_to_alternate_sales_column(
    tmp_path: Path, monkeypatch, column: str, requested: str
) -> None:
    _install_fake_avalanche(monkeypatch)
    df = _daily_frame("2015-01-01", 3, target=column)

    kwargs = _build(tmp_path, df, 1, feature_cols=["feat_a"], target_col=requested)

    datasets = kwargs["train_datasets"] + kwargs["test_datasets"]
    assert sorted(target for d in datasets for _, target in _collect(d)) == [101.0, 102.0, 103.0]


def test_build_raises_for_missing_target(tmp_path: Path, monkeypatch) -> None:
    _install_fake_avalanche(monkeypatch)
    df = _daily_frame("2015-01-01", 3).drop(columns=["sales"])
    with pytest.raises(ValueError, match="Target column 'demand' not found"):
        _build(tmp_path, df, 1, feature_cols=["feat_a"], target_col="demand")


def test_build_drops_nan_rows_and_skips_empty_months(tmp_path: Path, monkeypatch) -> None:
    _install_fake_avalanche(monkeypatch)
    df = pd.concat(
        [
            _daily_frame("2015-01-01", 3),
            _daily_frame("2015-02-01", 4),
            _daily_frame("2015-03-01", 2),
        ],
        ignore_index=True,
    )
    df.loc[df["Date"].dt.month == 1, "feat_a"] = float("nan")  # January fully invalid
    df.loc[df["Date"] == pd.Timestamp("2015-02-04"), "sales"] = float("nan")

    kwargs = _build(tmp_path, df, 3, feature_cols=["feat_a"])

    train_sets, test_sets = kwargs["train_datasets"], kwargs["test_datasets"]
    assert [len(t) + len(v) for t, v in zip(train_sets, test_sets)] == [3, 2]
    assert _parquets(tmp_path) == ["experience_01_2015-02.parquet", "experience_02_2015-03.parquet"]
    february = pd.read_parquet(tmp_path / "experience_01_2015-02.parquet")
    assert len(february) == 3
    assert february["sales"].notna().all()


def test_build_single_row_month_reuses_dataset_for_train_and_test(
    tmp_path: Path, monkeypatch
) -> None:
    _install_fake_avalanche(monkeypatch)

    kwargs = _build(tmp_path, _daily_frame("2015-01-01", 1), 1, feature_cols=["feat_a", "feat_b"])

    (train,) = kwargs["train_datasets"]
    (test,) = kwargs["test_datasets"]
    assert isinstance(train, TensorDataset)
    assert train is test
    assert _collect(train) == [([1.0, 1.0], 101.0)]
