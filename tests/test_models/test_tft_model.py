"""Unit tests for the TFT forecaster wrapper, asymmetric loss and persistence paths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from pytorch_forecasting.metrics import QuantileLoss

from src.models import tft_model
from src.models.tft_model import AsymmetricQuantileLoss, TFTForecaster, resolve_quantile_indices

QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]


@pytest.fixture
def config_path(tmp_path: Path) -> str:
    """Write a minimal valid TFT config and return its path."""
    cfg = {
        "tft": {
            "learning_rate": 0.001,
            "hidden_size": 16,
            "lstm_layers": 1,
            "num_attention_heads": 2,
            "dropout": 0.1,
            "hidden_continuous_size": 8,
            "output_size": 7,
            "quantiles": QUANTILES,
            "max_encoder_length": 30,
            "max_prediction_length": 7,
            "target": "sales",
            "reduce_on_plateau_patience": 2,
            "weight_decay": 1e-4,
            "under_penalty_factor": 2.0,
            "static_categoricals": ["store_type"],
            "time_varying_known_reals": ["promo"],
        }
    }
    path = tmp_path / "tft_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


class _MlflowRecorder:
    """Capture mlflow param/artifact calls while pretending a run is active."""

    def __init__(self, monkeypatch, active: bool = True) -> None:  # noqa: ANN001
        self.params: dict[str, object] = {}
        self.artifacts: list[str] = []
        monkeypatch.setattr(tft_model.mlflow, "active_run", lambda: object() if active else None)
        monkeypatch.setattr(tft_model.mlflow, "log_param", self._log_param)
        monkeypatch.setattr(tft_model.mlflow, "log_artifact", self.artifacts.append)

    def _log_param(self, key: str, value: object) -> None:
        self.params[key] = value


# -----------------------------------------------------------------------
# AsymmetricQuantileLoss / resolve_quantile_indices
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quantiles", "factor", "pred", "target", "expected"),
    [
        # Under-prediction by 1: pinball 2*q*1, weights [3, 3, 1] -> [0.6, 3.0, 1.8].
        ([0.1, 0.5, 0.9], 3.0, 0.0, 1.0, [0.6, 3.0, 1.8]),
        # Over-prediction by 2: pinball 2*(1-q)*2, weights [2, 1] -> [6.0, 1.0].
        ([0.25, 0.75], 2.0, 2.0, 0.0, [6.0, 1.0]),
    ],
)
def test_asymmetric_loss_weights_quantiles_at_or_below_median(
    quantiles, factor, pred, target, expected  # noqa: ANN001
) -> None:
    """Quantiles <= 0.5 get the under-penalty weight; others keep the pinball loss."""
    loss = AsymmetricQuantileLoss(quantiles=quantiles, under_penalty_factor=factor)
    out = loss.loss(torch.full((2, 4, len(quantiles)), pred), torch.full((2, 4), target))
    assert out.shape == (2, 4, len(quantiles))
    assert torch.allclose(out, torch.tensor(expected).expand_as(out))
    assert loss.under_penalty_factor == factor


def test_asymmetric_loss_scalar_base_loss_uses_global_factor(monkeypatch) -> None:
    """When the base loss has no quantile dimension the whole loss is scaled."""
    monkeypatch.setattr(QuantileLoss, "loss", lambda self, y_pred, target: torch.tensor(2.0))
    loss = AsymmetricQuantileLoss(quantiles=[0.1, 0.9], under_penalty_factor=1.5)
    out = loss.loss(torch.zeros(1, 1, 2), torch.zeros(1, 1))
    assert float(out) == pytest.approx(3.0)


def test_resolve_quantile_indices_picks_nearest(config_path: str) -> None:
    """Indices resolve to the closest configured level and are exposed by the forecaster."""
    assert resolve_quantile_indices(QUANTILES) == {"p10": 1, "p50": 3, "p90": 5}
    assert resolve_quantile_indices([0.05, 0.45, 0.95], targets=(0.1, 0.5, 0.9)) == {
        "p10": 0,
        "p50": 1,
        "p90": 2,
    }
    assert TFTForecaster(config_path).quantile_indices == {"p10": 1, "p50": 3, "p90": 5}


# -----------------------------------------------------------------------
# build / predict
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda forecaster, path: forecaster.predict(object()),
        lambda forecaster, path: forecaster.predict_for_store(1, 7, object(), pd.DataFrame()),
        lambda forecaster, path: forecaster.save(str(path / "m.pt")),
        lambda forecaster, path: forecaster.save_model(str(path / "m")),
    ],
    ids=["predict", "predict_for_store", "save", "save_model"],
)
def test_methods_require_built_model(
    config_path: str, tmp_path: Path, call
) -> None:  # noqa: ANN001
    """Prediction and persistence fail before build()/load() and write nothing."""
    with pytest.raises(RuntimeError, match=r"build\(\)|No model"):
        call(TFTForecaster(config_path), tmp_path)
    assert list(tmp_path.glob("m*")) == []


def test_build_without_dataset_raises(config_path: str) -> None:
    """build() requires a dataset."""
    with pytest.raises(RuntimeError, match="Dataset is required"):
        TFTForecaster(config_path, dataset=None).build()


def test_build_passes_config_and_logs_to_active_run(config_path: str, monkeypatch) -> None:
    """build() forwards config hyperparameters and logs param count to MLflow."""
    recorder = _MlflowRecorder(monkeypatch)
    captured: dict[str, object] = {}
    linear = torch.nn.Linear(3, 2)  # 3*2 + 2 = 8 parameters

    def _from_dataset(dataset, **kwargs):  # noqa: ANN001
        captured["dataset"] = dataset
        captured.update(kwargs)
        return linear

    monkeypatch.setattr(tft_model.TemporalFusionTransformer, "from_dataset", _from_dataset)
    dataset = object()
    model = TFTForecaster(config_path, dataset).build()

    assert model is linear
    assert captured["dataset"] is dataset
    assert captured["hidden_size"] == 16
    assert captured["attention_head_size"] == 2
    assert captured["log_interval"] == -1
    assert isinstance(captured["loss"], AsymmetricQuantileLoss)
    assert captured["loss"].under_penalty_factor == 2.0
    assert list(captured["loss"].quantiles) == QUANTILES
    assert recorder.params == {"tft_param_count": 8, "under_penalty_factor": 2.0}


class _TupleModel:
    def __init__(self, result) -> None:  # noqa: ANN001
        self.result = result
        self.calls: list[dict] = []

    def predict(self, dataloader, **kwargs):  # noqa: ANN001, ANN201
        self.calls.append(kwargs)
        return self.result


class _Prediction:
    def __init__(self, output, index) -> None:  # noqa: ANN001
        self.output = output
        self.index = index


@pytest.mark.parametrize(
    ("result", "expected_pred", "expected_index"),
    [
        (("P", "I"), "P", "I"),
        (("P",), "P", None),
        (_Prediction("P", "I"), "P", "I"),
        ("P", "P", None),
    ],
)
def test_predict_unpacks_result_variants(
    config_path: str, result, expected_pred, expected_index  # noqa: ANN001
) -> None:
    """predict() normalises tuple, Prediction-like and bare return values."""
    forecaster = TFTForecaster(config_path)
    forecaster.model = _TupleModel(result)
    predictions, index = forecaster.predict(dataloader=object(), mode="quantiles")
    assert predictions == expected_pred
    assert index == expected_index
    assert forecaster.model.calls == [{"mode": "quantiles", "return_index": True}]


# -----------------------------------------------------------------------
# predict_for_store
# -----------------------------------------------------------------------


class _InferenceDataset:
    def __init__(self) -> None:
        self.loader_kwargs: dict = {}

    def to_dataloader(self, **kwargs):  # noqa: ANN201
        self.loader_kwargs = kwargs
        return "loader"


def _reference_df(store_col: str = "Store", with_date: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            store_col: [1, 1, 1, 2],
            "time_idx": [2, 0, 1, 0],
            "store_type": [3, 3, 3, 4],
            "promo": ["1", "0", "x", "1"],
            "sales": ["10", "20", "30", "40"],
        }
    )
    if with_date:
        frame["Date"] = pd.to_datetime(["2015-01-03", "2015-01-01", "2015-01-02", "2015-01-01"])
    return frame


def _patch_from_dataset(monkeypatch) -> dict:  # noqa: ANN001
    captured: dict = {"ds": _InferenceDataset()}

    def _from_dataset(training_dataset, data, **kwargs):  # noqa: ANN001
        captured["training_dataset"] = training_dataset
        captured["data"] = data
        captured["kwargs"] = kwargs
        return captured["ds"]

    monkeypatch.setattr(tft_model.TimeSeriesDataSet, "from_dataset", _from_dataset)
    return captured


def test_predict_for_store_returns_horizon_and_dates(config_path: str, monkeypatch) -> None:
    """Store rows are typed, sorted, predicted and dated from the last Date."""
    captured = _patch_from_dataset(monkeypatch)
    forecaster = TFTForecaster(config_path)
    preds = torch.arange(1 * 7 * 7, dtype=torch.float32).reshape(1, 7, 7)
    forecaster.model = _TupleModel(preds)
    training_ds = object()

    tensor, dates = forecaster.predict_for_store(
        store_id=1, horizon=3, training_dataset=training_ds, reference_df=_reference_df()
    )

    assert torch.equal(tensor, preds[0][:3])
    assert dates == [
        pd.Timestamp("2015-01-04"),
        pd.Timestamp("2015-01-05"),
        pd.Timestamp("2015-01-06"),
    ]
    data = captured["data"]
    assert list(data["time_idx"]) == [0, 1, 2]
    assert set(data["Store"]) == {1}
    assert data["store_type"].tolist() == ["3", "3", "3"]
    assert data["sales"].tolist() == [20.0, 30.0, 10.0]
    assert data["promo"].iloc[0] == 0.0 and data["promo"].iloc[2] == 1.0
    assert np.isnan(data["promo"].iloc[1])
    assert captured["training_dataset"] is training_ds
    assert captured["kwargs"] == {"predict": True, "stop_randomization": True}
    assert captured["ds"].loader_kwargs == {"train": False, "batch_size": 1, "num_workers": 0}
    assert forecaster.model.calls == [{"mode": "quantiles"}]
    assert forecaster._predict_warned is True


def test_predict_for_store_lowercase_store_and_numpy_predictions(
    config_path: str, monkeypatch
) -> None:
    """A lowercase store column and numpy 2-D output are accepted."""
    _patch_from_dataset(monkeypatch)
    forecaster = TFTForecaster(config_path)
    forecaster.model = _TupleModel(np.ones((5, 7)))
    tensor, dates = forecaster.predict_for_store(
        store_id=2,
        horizon=2,
        training_dataset=object(),
        reference_df=_reference_df(store_col="store", with_date=False),
    )
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (2, 7)
    assert len(dates) == 2
    assert dates[1] - dates[0] == pd.Timedelta(days=1)


def test_predict_for_store_rejects_bad_prediction_shape(config_path: str, monkeypatch) -> None:
    """A 1-D quantile output is rejected."""
    _patch_from_dataset(monkeypatch)
    forecaster = TFTForecaster(config_path)
    forecaster.model = _TupleModel(torch.ones(7))
    with pytest.raises(RuntimeError, match="Unexpected quantile"):
        forecaster.predict_for_store(1, 3, object(), _reference_df())


def test_predict_for_store_missing_store_or_column(config_path: str) -> None:
    """Unknown store ids and missing store columns raise KeyError."""
    forecaster = TFTForecaster(config_path)
    forecaster.model = _TupleModel(None)
    with pytest.raises(KeyError, match="Store 99 not found"):
        forecaster.predict_for_store(99, 3, object(), _reference_df())
    with pytest.raises(KeyError, match="Store column"):
        forecaster.predict_for_store(1, 3, object(), pd.DataFrame({"x": [1]}))


# -----------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------


def test_save_writes_state_dict_and_logs_artifact(
    config_path: str, tmp_path: Path, monkeypatch
) -> None:
    """save() persists a loadable state dict and logs it to an active run."""
    recorder = _MlflowRecorder(monkeypatch)
    forecaster = TFTForecaster(config_path)
    forecaster.model = torch.nn.Linear(2, 1)
    out = tmp_path / "nested" / "model.pt"

    forecaster.save(str(out))

    state = torch.load(out, map_location="cpu")
    assert torch.equal(state["weight"], forecaster.model.weight.detach())
    assert recorder.artifacts == [str(out)]


def test_save_model_serialises_hparams_and_logs(
    config_path: str, tmp_path: Path, monkeypatch
) -> None:
    """Bundle metadata contains JSON-safe hparams and both files are logged."""
    recorder = _MlflowRecorder(monkeypatch)
    model = torch.nn.Linear(2, 1)
    model.hparams = {"hidden": 16, "nested": {"a": (1, 2)}, "obj": Path("p"), "none": None}
    forecaster = TFTForecaster(config_path)
    forecaster.model = model
    base = tmp_path / "bundle" / "tft_latest"

    forecaster.save_model(str(base))

    meta = json.loads(base.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["model_hparams"] == {
        "hidden": 16,
        "nested": {"a": [1, 2]},
        "obj": str(Path("p")),
        "none": None,
    }
    assert meta["under_penalty_factor"] == 2.0
    assert meta["static_categoricals"] == ["store_type"]
    assert meta["group_ids"] == ["Store"]
    assert recorder.artifacts == [str(base.with_suffix(".pt")), str(base.with_suffix(".meta.json"))]


def test_bundle_paths_accepts_base_or_checkpoint(tmp_path: Path) -> None:
    """Both base and .pt paths resolve to the same bundle triple."""
    base = tmp_path / "tft_latest"
    expected = (base, base.with_suffix(".pt"), base.with_suffix(".meta.json"))
    assert TFTForecaster._bundle_paths(str(base)) == expected
    assert TFTForecaster._bundle_paths(str(base.with_suffix(".pt"))) == expected


def _saved_linear(path: Path) -> torch.nn.Linear:
    torch.manual_seed(0)
    source = torch.nn.Linear(2, 1)
    torch.save(source.state_dict(), path)
    return source


def test_load_any_checkpoint_only_builds_then_restores_weights(
    config_path: str, tmp_path: Path, monkeypatch
) -> None:
    """Without metadata, load_any() falls back to load(): build from dataset, then load weights."""
    _MlflowRecorder(monkeypatch, active=False)
    weights = tmp_path / "w.pt"
    source = _saved_linear(weights)
    monkeypatch.setattr(
        tft_model.TemporalFusionTransformer,
        "from_dataset",
        lambda dataset, **kwargs: torch.nn.Linear(2, 1),
    )

    loaded = TFTForecaster.load_any(str(weights), config_path, dataset=object())

    assert torch.equal(loaded.model.weight.cpu(), source.weight.detach())
    assert torch.equal(loaded.model.bias.cpu(), source.bias.detach())


def test_load_without_dataset_or_metadata_raises(config_path: str, tmp_path: Path) -> None:
    """Dataset-free load requires a metadata bundle."""
    weights = tmp_path / "w.pt"
    _saved_linear(weights)
    with pytest.raises(RuntimeError, match="metadata bundle"):
        TFTForecaster.load(str(weights), config_path)


def test_load_any_prefers_metadata_bundle(config_path: str, tmp_path: Path, monkeypatch) -> None:
    """With weights + metadata, load_any() reconstructs from metadata and restores weights."""
    base = tmp_path / "tft_latest"
    source = _saved_linear(base.with_suffix(".pt"))
    base.with_suffix(".meta.json").write_text("{}", encoding="utf-8")
    seen: list[Path] = []

    def _from_meta(meta_path: Path) -> torch.nn.Linear:
        seen.append(meta_path)
        return torch.nn.Linear(2, 1)

    monkeypatch.setattr(TFTForecaster, "_build_model_from_metadata", staticmethod(_from_meta))

    loaded = TFTForecaster.load_any(str(base), config_path)

    assert seen == [base.with_suffix(".meta.json")]
    assert torch.equal(loaded.model.weight.cpu(), source.weight.detach())


def test_load_model_missing_files_raise(config_path: str, tmp_path: Path) -> None:
    """load_model() reports a missing weights file, then a missing metadata file."""
    base = tmp_path / "tft_latest"
    with pytest.raises(FileNotFoundError, match="weights"):
        TFTForecaster.load_model(str(base), config_path)
    base.with_suffix(".pt").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="metadata"):
        TFTForecaster.load_model(str(base), config_path)


def test_load_any_raises_when_artifacts_missing(config_path: str, tmp_path: Path) -> None:
    """load_any() raises when neither checkpoint nor bundle exists."""
    with pytest.raises(FileNotFoundError, match="Unable to locate TFT model artifacts"):
        TFTForecaster.load_any(str(tmp_path / "missing_tft"), config_path)


def test_build_model_from_metadata_strips_loss_entries(tmp_path: Path, monkeypatch) -> None:
    """Metadata reconstruction drops serialised loss/metrics and rebuilds the loss."""
    captured: dict = {}

    class _FakeTFT:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            captured.update(kwargs)

    monkeypatch.setattr(tft_model, "TemporalFusionTransformer", _FakeTFT)
    meta = tmp_path / "m.meta.json"
    meta.write_text(
        json.dumps(
            {
                "quantiles": [0.2, 0.8],
                "under_penalty_factor": 4,
                "model_hparams": {"hidden_size": 8, "loss": "x", "logging_metrics": "y"},
            }
        ),
        encoding="utf-8",
    )

    model = TFTForecaster._build_model_from_metadata(meta)

    assert isinstance(model, _FakeTFT)
    assert set(captured) == {"hidden_size", "loss"}
    assert captured["hidden_size"] == 8
    assert list(captured["loss"].quantiles) == [0.2, 0.8]
    assert captured["loss"].under_penalty_factor == 4.0
