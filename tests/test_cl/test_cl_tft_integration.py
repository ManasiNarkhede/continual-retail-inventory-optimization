"""Regression tests ensuring CL ablation uses TFT backbones."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.tft_model import TFTForecaster


def test_tft_forecaster_build_returns_torch_module(monkeypatch) -> None:
    """TFTForecaster.build should expose a real torch module for CL strategies."""

    class _StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

    monkeypatch.setattr(
        "src.models.tft_model.TemporalFusionTransformer.from_dataset",
        lambda *args, **kwargs: _StubModel(),
    )

    forecaster = TFTForecaster(config_path="configs/tft_config.yaml", dataset=object())
    built = forecaster.build()

    assert isinstance(built, nn.Module)
    assert forecaster.model is built
    params = list(built.parameters())
    assert params and torch.is_tensor(params[0])
