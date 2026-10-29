"""Unit tests for TFT interpretability computations and artefacts using lightweight fakes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytorch_forecasting
import torch

from src.models.tft_interpretability import TFTInterpreter


class _Dataset:
    time_varying_known_reals = ["promo", "dow"]
    time_varying_unknown_reals = ["sales"]
    static_categoricals = ["store_type"]
    static_reals = ["distance"]


class _Prediction:
    def __init__(self, output) -> None:  # noqa: ANN001
        self.output = output


class _InterpretModel:
    """Fake TFT exposing predict/interpret_output/forward with fixed outputs."""

    def __init__(
        self, encoder, static, predict_style: str = "tuple", attention=None
    ) -> None:  # noqa: ANN001
        self.encoder = encoder
        self.static = static
        self.predict_style = predict_style
        self.attention = attention
        self.eval_called = False
        self.raw_seen: list = []
        self.forward_inputs: list = []

    def eval(self):  # noqa: ANN201
        self.eval_called = True
        return self

    def predict(self, dataloader, mode, return_x):  # noqa: ANN001, ANN201
        assert mode == "raw" and return_x is True
        raw = {"raw": dataloader}
        if self.predict_style == "tuple":
            return raw, "x"
        if self.predict_style == "output":
            return _Prediction(raw)
        return raw

    def interpret_output(self, raw_output, reduction):  # noqa: ANN001, ANN201
        assert reduction == "sum"
        self.raw_seen.append(raw_output)
        return {"encoder_variables": self.encoder, "static_variables": self.static}

    def __call__(self, x):  # noqa: ANN001, ANN204
        self.forward_inputs.append(x)
        if self.attention is None:
            return {}
        return {"attention": self.attention[len(self.forward_inputs) - 1]}


class _Loader:
    def __init__(self, n_batches: int, batch_size: int) -> None:
        self.n_batches = n_batches
        self.batch_size = batch_size

    def __iter__(self):  # noqa: ANN204
        for i in range(self.n_batches):
            yield ({"batch": i}, None)


@pytest.mark.parametrize("predict_style", ["tuple", "output", "plain"])
def test_variable_importance_normalises_and_sorts(predict_style: str) -> None:
    """Encoder and static importances are normalised per group and sorted."""
    model = _InterpretModel(
        encoder=torch.tensor([1.0, 2.0, 1.0]),
        static=torch.tensor([3.0, 1.0]),
        predict_style=predict_style,
    )
    interpreter = TFTInterpreter(model, _Dataset())
    assert model.eval_called

    df = interpreter.variable_importance(dataloader="loader")

    assert model.raw_seen == [{"raw": "loader"}]
    assert list(df.columns) == ["variable", "importance", "type"]
    assert df["importance"].is_monotonic_decreasing
    lookup = {row.variable: (row.importance, row.type) for row in df.itertuples()}
    assert lookup == {
        "promo": (pytest.approx(0.25), "encoder"),
        "dow": (pytest.approx(0.5), "encoder"),
        "sales": (pytest.approx(0.25), "encoder"),
        "store_type": (pytest.approx(0.75), "static"),
        "distance": (pytest.approx(0.25), "static"),
    }
    assert df.iloc[0]["variable"] == "store_type"


def test_variable_importance_handles_numpy_and_zero_totals() -> None:
    """Non-tensor inputs are accepted and all-zero importances stay zero."""
    model = _InterpretModel(encoder=[0.0, 0.0, 0.0], static=[2.0, 2.0])
    df = TFTInterpreter(model, _Dataset()).variable_importance(dataloader=None)
    enc = df[df["type"] == "encoder"]
    static = df[df["type"] == "static"]
    assert enc["importance"].tolist() == [0.0, 0.0, 0.0]
    assert static["importance"].tolist() == [0.5, 0.5]


def test_attention_weights_averages_batch_means() -> None:
    """Attention is averaged over heads/batch, then over processed batches."""
    batch_a = torch.ones(2, 3, 2, 2)
    batch_b = torch.full((2, 3, 2, 2), 3.0)
    model = _InterpretModel(None, None, attention=[batch_a, batch_b])
    interpreter = TFTInterpreter(model, _Dataset())

    result = interpreter.attention_weights(_Loader(n_batches=2, batch_size=2), n_samples=0)

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, np.full((2, 2), 2.0))
    assert model.forward_inputs == [{"batch": 0}, {"batch": 1}]


def test_attention_weights_stops_after_n_samples() -> None:
    """Iteration stops once n_samples have been seen."""
    batches = [np.full((4, 1, 3, 3), float(i + 1)) for i in range(3)]
    model = _InterpretModel(None, None, attention=batches)
    result = TFTInterpreter(model, _Dataset()).attention_weights(
        _Loader(n_batches=3, batch_size=4), n_samples=4
    )
    assert len(model.forward_inputs) == 1
    np.testing.assert_allclose(result, np.ones((3, 3)))


def test_attention_weights_without_attention_returns_zero_matrix() -> None:
    """Batches lacking attention output yield a 1x1 zero matrix."""
    model = _InterpretModel(None, None, attention=None)
    result = TFTInterpreter(model, _Dataset()).attention_weights(_Loader(2, 1))
    assert result.shape == (1, 1)
    assert result[0, 0] == 0.0
    assert len(model.forward_inputs) == 2


def test_seasonal_attention_profile_reads_last_row_and_smooths() -> None:
    """Profile is last-row attention by lag, centred rolling mean with min_periods=1."""
    matrix = np.arange(16, dtype=float).reshape(4, 4)
    interpreter = TFTInterpreter(_InterpretModel(None, None), _Dataset())

    raw = interpreter.seasonal_attention_profile(matrix, window=1)
    smoothed = interpreter.seasonal_attention_profile(matrix, window=3)

    np.testing.assert_allclose(raw, [15.0, 14.0, 13.0, 12.0])
    np.testing.assert_allclose(smoothed, [14.5, 14.0, 13.0, 12.5])


def test_save_artefacts_writes_tables_figure_and_metadata(tmp_path: Path, monkeypatch) -> None:
    """save_artefacts persists importance CSV/JSON, attention NPY/PNG and a metadata index."""
    interpreter = TFTInterpreter(_InterpretModel(None, None), _Dataset())
    importance = pd.DataFrame(
        [
            {"variable": "promo", "importance": 0.7, "type": "encoder"},
            {"variable": "store_type", "importance": 0.3, "type": "static"},
        ]
    )
    attention = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)
    monkeypatch.setattr(interpreter, "variable_importance", lambda dataloader: importance)
    monkeypatch.setattr(interpreter, "attention_weights", lambda dataloader: attention)

    csv_path, png_path = interpreter.save_artefacts(dataloader=object(), output_dir=str(tmp_path))

    assert csv_path == tmp_path / "tft_variable_importance.csv"
    assert png_path == tmp_path / "tft_attention_map.png"
    assert png_path.stat().st_size > 0
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), importance)
    np.testing.assert_array_equal(np.load(tmp_path / "tft_attention_weights.npy"), attention)
    records = json.loads((tmp_path / "tft_variable_importance.json").read_text(encoding="utf-8"))
    assert [row["variable"] for row in records] == ["promo", "store_type"]
    meta = json.loads((tmp_path / "tft_interpretability_metadata.json").read_text(encoding="utf-8"))
    assert meta["importance_rows"] == 2
    assert meta["attention_shape"] == [2, 2]
    assert meta["files"]["attention_png"] == "tft_attention_map.png"


def _reference_df() -> pd.DataFrame:
    return pd.DataFrame({"Store": [1, 1, 2], "time_idx": [1, 0, 0], "sales": [1.0, 2.0, 3.0]})


def test_generate_store_figures_writes_pngs(tmp_path: Path, monkeypatch) -> None:
    """Store figures are built from an inference loader for the selected store."""
    captured: dict = {}

    class _InferenceDs:
        def to_dataloader(self, **kwargs):  # noqa: ANN201
            captured["loader_kwargs"] = kwargs
            return "store-loader"

    def _from_dataset(dataset, data, **kwargs):  # noqa: ANN001
        captured["dataset"] = dataset
        captured["data"] = data
        captured["kwargs"] = kwargs
        return _InferenceDs()

    monkeypatch.setattr(pytorch_forecasting.TimeSeriesDataSet, "from_dataset", _from_dataset)
    dataset = _Dataset()
    interpreter = TFTInterpreter(_InterpretModel(None, None), dataset)
    importance = pd.DataFrame(
        {"variable": ["a", "b"], "importance": [0.6, 0.4], "type": ["encoder", "static"]}
    )
    loaders: list = []

    def _importance(loader):  # noqa: ANN001, ANN202
        loaders.append(loader)
        return importance

    def _attention(loader, n_samples):  # noqa: ANN001, ANN202
        loaders.append((loader, n_samples))
        return np.eye(3)

    monkeypatch.setattr(interpreter, "variable_importance", _importance)
    monkeypatch.setattr(interpreter, "attention_weights", _attention)

    var_path, attn_path = interpreter.generate_store_figures(
        store_id=1, reference_df=_reference_df(), output_dir=str(tmp_path / "figs"), horizon=10
    )

    assert var_path == tmp_path / "figs" / "store_1_variable_importance.png"
    assert attn_path == tmp_path / "figs" / "store_1_attention_map.png"
    assert var_path.stat().st_size > 0 and attn_path.stat().st_size > 0
    assert captured["dataset"] is dataset
    assert captured["data"]["time_idx"].tolist() == [0, 1]
    assert set(captured["data"]["Store"]) == {1}
    assert captured["kwargs"] == {"predict": True, "stop_randomization": True}
    assert captured["loader_kwargs"] == {"train": False, "batch_size": 32, "num_workers": 0}
    assert loaders == ["store-loader", ("store-loader", 80)]


def test_generate_store_figures_unknown_store_raises(tmp_path: Path) -> None:
    """A store absent from the reference frame raises KeyError."""
    interpreter = TFTInterpreter(_InterpretModel(None, None), _Dataset())
    with pytest.raises(KeyError, match="Store 7"):
        interpreter.generate_store_figures(7, _reference_df(), output_dir=str(tmp_path))
