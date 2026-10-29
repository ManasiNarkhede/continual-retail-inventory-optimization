"""Unit tests for continual learning strategy wiring and TFT-aware loss."""

from __future__ import annotations

from pathlib import Path

import torch

from src.cl.strategy import AvalancheCLStrategy, _batch_loss
from src.models.tft_model import AsymmetricQuantileLoss


class _TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):  # noqa: ANN001
        return self.linear(x)


def _write_minimal_cl_config(tmp_path: Path) -> Path:
    path = tmp_path / "cl_config.yaml"
    path.write_text(
        """
continual_learning:
  strategy: Naive
  train_mb_size: 4
  train_epochs: 1
  eval_mb_size: 4
  ewc_lambda: 0.4
  replay_mem_size: 64
  device: cpu
  dynamic_prune_threshold: 0.08
  packnet_base_prune_ratio: 0.5
  enable_packnet_scheduler: false
""".strip(),
        encoding="utf-8",
    )
    return path


def test_strategy_uses_tft_backbone_module(tmp_path: Path, monkeypatch) -> None:
    """Strategy should bind to wrapped forecaster.model backbone."""

    monkeypatch.setattr(AvalancheCLStrategy, "_build_eval_plugin", lambda self: object())
    monkeypatch.setattr(AvalancheCLStrategy, "_build_strategy", lambda self: object())

    class _Forecaster:
        def __init__(self) -> None:
            self.model = _TinyModule()

    cfg_path = _write_minimal_cl_config(tmp_path)
    wrapped = _Forecaster()
    strategy = AvalancheCLStrategy(model=wrapped, config_path=str(cfg_path))
    try:
        assert strategy.model is wrapped.model
        assert strategy.backbone is wrapped.model
    finally:
        strategy.close()


def test_strategy_accepts_raw_torch_module(tmp_path: Path, monkeypatch) -> None:
    """Strategy should also accept a raw torch.nn.Module directly."""
    monkeypatch.setattr(AvalancheCLStrategy, "_build_eval_plugin", lambda self: object())
    monkeypatch.setattr(AvalancheCLStrategy, "_build_strategy", lambda self: object())
    cfg_path = _write_minimal_cl_config(tmp_path)
    model = _TinyModule()
    strategy = AvalancheCLStrategy(model=model, config_path=str(cfg_path))
    try:
        assert strategy.model is model
        assert strategy.backbone is model
    finally:
        strategy.close()


def test_batch_loss_handles_tft_quantile_output_with_asymmetric_loss() -> None:
    """Batch loss should route quantile tensor into AsymmetricQuantileLoss."""

    class _QuantileModel(torch.nn.Module):
        def forward(self, x):  # noqa: ANN001
            batch, horizon = x.shape
            prediction = torch.zeros((batch, horizon, 3), dtype=torch.float32)
            prediction[:, :, 0] = 10.0
            prediction[:, :, 1] = 12.0
            prediction[:, :, 2] = 14.0
            return {"prediction": prediction}

    model = _QuantileModel()
    criterion = AsymmetricQuantileLoss(quantiles=[0.1, 0.5, 0.9], under_penalty_factor=1.5)
    inputs = torch.ones((2, 4), dtype=torch.float32)
    targets = torch.full((2, 4), 11.0, dtype=torch.float32)

    loss = _batch_loss(
        model=model, criterion=criterion, batch=(inputs, (targets,)), device=torch.device("cpu")
    )

    assert torch.is_tensor(loss)
    assert float(loss.detach().cpu().mean()) >= 0.0


def test_strategy_builds_asymmetric_quantile_loss(tmp_path: Path, monkeypatch) -> None:
    """CL strategy should create TFT-native AsymmetricQuantileLoss criterion."""
    monkeypatch.setattr(AvalancheCLStrategy, "_build_eval_plugin", lambda self: object())

    captured = {}

    def _fake_build_strategy(self):  # noqa: ANN001
        captured["criterion"] = self._build_criterion()
        return object()

    monkeypatch.setattr(AvalancheCLStrategy, "_build_strategy", _fake_build_strategy)

    class _Forecaster:
        def __init__(self) -> None:
            self.model = _TinyModule()
            self.config = {"quantiles": [0.1, 0.5, 0.9], "under_penalty_factor": 1.7}

    strategy = AvalancheCLStrategy(
        model=_Forecaster(), config_path=str(_write_minimal_cl_config(tmp_path))
    )
    try:
        criterion = captured.get("criterion")
        assert isinstance(criterion, AsymmetricQuantileLoss)
        assert criterion.quantiles == [0.1, 0.5, 0.9]
        assert criterion.under_penalty_factor == 1.7
    finally:
        strategy.close()
