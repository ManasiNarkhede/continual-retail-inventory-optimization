"""Unit tests for TFTForecaster wrapper methods."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

from src.models.tft_model import TFTForecaster


@pytest.fixture
def config_path(tmp_path: Path) -> str:
    """Create temporary TFT config file.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        Config path string.
    """
    cfg = {
        "tft": {
            "learning_rate": 0.001,
            "hidden_size": 16,
            "lstm_layers": 1,
            "num_attention_heads": 2,
            "dropout": 0.1,
            "hidden_continuous_size": 8,
            "output_size": 7,
            "quantiles": [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
            "max_encoder_length": 30,
            "max_prediction_length": 7,
            "target": "sales",
            "reduce_on_plateau_patience": 2,
            "weight_decay": 1e-4,
        }
    }
    path = tmp_path / "tft_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_predict_raises_if_model_missing(config_path: str) -> None:
    """Ensure predict fails when build was not called.

    Args:
        config_path: Temporary config path.
    """
    dataset = MagicMock()
    forecaster = TFTForecaster(config_path, dataset)
    with pytest.raises(RuntimeError):
        forecaster.predict(MagicMock())


def test_save_raises_if_model_missing(config_path: str) -> None:
    """Ensure save fails when model is unavailable.

    Args:
        config_path: Temporary config path.
    """
    dataset = MagicMock()
    forecaster = TFTForecaster(config_path, dataset)
    with pytest.raises(RuntimeError):
        forecaster.save("model.pt")


def test_build_uses_from_dataset(config_path: str) -> None:
    """Ensure build delegates to TemporalFusionTransformer.from_dataset.

    Args:
        config_path: Temporary config path.
    """
    dataset = MagicMock()
    fake_model = MagicMock()
    fake_model.parameters.return_value = [torch.zeros(2, 2)]

    with patch(
        "src.models.tft_model.TemporalFusionTransformer.from_dataset", return_value=fake_model
    ):
        forecaster = TFTForecaster(config_path, dataset)
        model = forecaster.build()
        assert model is fake_model


def test_predict_for_store_requires_built_model(config_path: str) -> None:
    """Ensure predict_for_store raises when model is not initialized.

    Args:
        config_path: Temporary config path.
    """
    forecaster = TFTForecaster(config_path, MagicMock())
    with pytest.raises(RuntimeError):
        forecaster.predict_for_store(
            store_id=1,
            horizon=7,
            training_dataset=MagicMock(),
            reference_df=MagicMock(),
        )


def test_save_model_writes_bundle_metadata(config_path: str, tmp_path: Path) -> None:
    """save_model should write both weights and metadata bundle files."""
    dataset = MagicMock()
    forecaster = TFTForecaster(config_path, dataset)
    forecaster.model = MagicMock()

    base_path = tmp_path / "tft_latest"
    with patch("src.models.tft_model.torch.save") as mock_save:
        forecaster.save_model(str(base_path))

    mock_save.assert_called_once()
    saved_path = mock_save.call_args[0][1]
    assert Path(saved_path) == base_path.with_suffix(".pt")

    meta_path = base_path.with_suffix(".meta.json")
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "quantiles" in payload
    assert "max_prediction_length" in payload


def test_load_any_prefers_bundle_when_metadata_exists(config_path: str, tmp_path: Path) -> None:
    """load_any should route through load_model when metadata file exists."""
    base_path = tmp_path / "tft_latest"
    base_path.with_suffix(".pt").write_bytes(b"weights")
    base_path.with_suffix(".meta.json").write_text("{}", encoding="utf-8")

    sentinel = MagicMock()
    with (
        patch.object(TFTForecaster, "load_model", return_value=sentinel) as mock_load_model,
        patch.object(
            TFTForecaster,
            "load",
            return_value=MagicMock(),
        ) as mock_load,
    ):
        result = TFTForecaster.load_any(
            path_or_base=str(base_path.with_suffix(".pt")),
            config_path=config_path,
            dataset=MagicMock(),
        )

    assert result is sentinel
    mock_load_model.assert_called_once()
    mock_load.assert_not_called()


def test_load_any_falls_back_to_checkpoint_only(config_path: str, tmp_path: Path) -> None:
    """load_any should use checkpoint loader when metadata is missing."""
    base_path = tmp_path / "tft_latest"
    checkpoint_path = base_path.with_suffix(".pt")
    checkpoint_path.write_bytes(b"weights")

    sentinel = MagicMock()
    with patch.object(TFTForecaster, "load", return_value=sentinel) as mock_load:
        result = TFTForecaster.load_any(
            path_or_base=str(checkpoint_path),
            config_path=config_path,
            dataset=MagicMock(),
        )

    assert result is sentinel
    mock_load.assert_called_once()


def test_load_any_raises_when_artifacts_missing(config_path: str, tmp_path: Path) -> None:
    """load_any should raise when neither checkpoint nor bundle exists."""
    with pytest.raises(FileNotFoundError):
        TFTForecaster.load_any(
            path_or_base=str(tmp_path / "missing_tft"),
            config_path=config_path,
            dataset=MagicMock(),
        )
