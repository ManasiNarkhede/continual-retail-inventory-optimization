"""Unit tests for the Avalanche CL strategy wrapper with stubbed Avalanche modules.

The ``avalanche.*`` modules imported lazily by ``AvalancheCLStrategy`` are replaced
through ``sys.modules`` so the wrapper logic (plugin/strategy wiring, isolated-loss
computation, intransigence bookkeeping, PackNet hooks) runs fast and deterministically.
"""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

import src.cl.packnet as packnet_module
import src.cl.strategy as strategy_module
from src.cl.strategy import (
    AvalancheCLStrategy,
    TFTAwareSmoothL1Loss,
    _batch_loss,
    _move_to_device,
    _prediction_tensor,
)
from src.models.tft_model import AsymmetricQuantileLoss

REPO_ROOT = Path(__file__).resolve().parents[2]
CPU = torch.device("cpu")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _pinball(pred: torch.Tensor, target: torch.Tensor, quantiles: list[float], under: float):
    """Hand-rolled asymmetric pinball loss (pytorch-forecasting scales by 2)."""
    total = []
    for idx, q in enumerate(quantiles):
        err = target - pred[..., idx]
        weight = under if q <= 0.5 else 1.0
        total.append(weight * 2.0 * torch.maximum((q - 1) * err, q * err))
    return float(torch.stack(total).mean())


def _loss_key(index: int) -> str:
    """Avalanche's per-experience eval loss metric name."""
    return f"Loss_Exp/eval_phase/test_stream/Task000/Exp{index:03d}"


class _QuantileNet(torch.nn.Module):
    """Tiny 2-layer net emitting ``(batch, horizon=1, 3 quantiles)``."""

    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Linear(4, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).unsqueeze(1)


class _FakeTextLogger:
    def __init__(self, file: Any) -> None:
        self.file = file
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _FakeInteractiveLogger:
    pass


class _FakeEvaluationPlugin:
    def __init__(self, *metrics: Any, loggers: list[Any]) -> None:
        self.metrics = metrics
        self.loggers = loggers


class _FakeStrategy:
    eval_fn: Callable[[int, Any], dict[str, Any]] = staticmethod(lambda idx, stream: {})

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.trained: list[Any] = []
        self.evaluated: list[Any] = []

    def train(self, experience: Any) -> None:
        self.trained.append(experience)

    def eval(self, stream: Any) -> dict[str, Any]:
        self.evaluated.append(list(stream))
        return type(self).eval_fn(len(self.evaluated) - 1, stream)


@pytest.fixture
def fake_avalanche(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, type]:
    """Run inside tmp_path with stub Avalanche modules installed and MLflow disabled."""
    schema_dir = tmp_path / "configs" / "schemas"
    schema_dir.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "configs" / "schemas" / "cl_schema.json", schema_dir)
    monkeypatch.chdir(tmp_path)

    classes = {name: type(name, (_FakeStrategy,), {}) for name in ("EWC", "Naive", "Replay")}
    metric_names = (
        "accuracy_metrics",
        "loss_metrics",
        "forgetting_metrics",
        "bwt_metrics",
        "forward_transfer_metrics",
    )
    modules = {
        "avalanche.evaluation.metrics": {
            n: (lambda name: lambda **kwargs: (name, kwargs))(n) for n in metric_names
        },
        "avalanche.logging": {
            "InteractiveLogger": _FakeInteractiveLogger,
            "TextLogger": _FakeTextLogger,
        },
        "avalanche.training.plugins": {"EvaluationPlugin": _FakeEvaluationPlugin},
        "avalanche.training.supervised": classes,
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    no_run = SimpleNamespace(active_run=lambda: None, log_metric=None)
    monkeypatch.setattr(strategy_module, "mlflow", no_run)
    monkeypatch.setattr(packnet_module, "mlflow", no_run)
    return classes


def _write_config(tmp_path: Path, **overrides: Any) -> str:
    values = {
        "strategy": "Naive",
        "train_mb_size": 8,
        "train_epochs": 1,
        "eval_mb_size": 4,
        "ewc_lambda": 0.4,
        "replay_mem_size": 64,
        "device": "cpu",
        "dynamic_prune_threshold": 0.08,
        "packnet_base_prune_ratio": 0.5,
        "enable_packnet_scheduler": False,
    }
    values.update(overrides)
    lines = ["continual_learning:"]
    for key, value in values.items():
        rendered = str(value).lower() if isinstance(value, bool) else value
        lines.append(f"  {key}: {rendered}")
    path = tmp_path / "cl_config.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _experience(seed: int, n: int = 4) -> SimpleNamespace:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=gen)
    y = torch.randn(n, 1, generator=gen)
    return SimpleNamespace(dataset=TensorDataset(x, y))


def _benchmark(n: int = 2) -> SimpleNamespace:
    train = [_experience(seed=10 + i) for i in range(n)]
    test = [f"test_exp_{i}" for i in range(n)]
    return SimpleNamespace(train_stream=train, test_stream=test)


def _expected_isolated_loss(state: dict[str, torch.Tensor], experience: SimpleNamespace) -> float:
    """Loss of the single (pre-step) batch seen by ``_compute_isolated_loss``."""
    model = _QuantileNet()
    model.load_state_dict(state)
    x, y = experience.dataset.tensors
    with torch.no_grad():
        return _pinball(model(x), y, [0.1, 0.5, 0.9], 1.5)


# -----------------------------------------------------------------------
# Pure helpers
# -----------------------------------------------------------------------


def test_move_to_device_handles_nested_containers() -> None:
    payload = {"a": torch.ones(2), "b": [torch.zeros(1), "label", (torch.arange(3), 7)]}
    moved = _move_to_device(payload, torch.device("meta"))

    assert moved["a"].device.type == "meta"
    assert isinstance(moved["b"], list)
    assert moved["b"][0].device.type == "meta"
    assert moved["b"][1] == "label"
    assert isinstance(moved["b"][2], tuple)
    assert moved["b"][2][0].device.type == "meta"
    assert moved["b"][2][1] == 7


PRED = torch.tensor([1.0])
OTHER = torch.tensor([2.0])


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (PRED, PRED),
        ({"other": OTHER, "prediction": PRED}, PRED),
        ({"meta": "x", "out": OTHER}, OTHER),
        (SimpleNamespace(prediction=PRED), PRED),
        (["x", OTHER, PRED], OTHER),
    ],
    ids=["tensor", "dict-prediction", "dict-first-tensor", "attribute", "sequence"],
)
def test_prediction_tensor_extracts_from_supported_containers(output: Any, expected: Any) -> None:
    assert _prediction_tensor(output) is expected


@pytest.mark.parametrize("bad", [{"a": 1}, ["x", 1], SimpleNamespace(prediction=[1.0]), 3.0])
def test_prediction_tensor_raises_without_tensor(bad: Any) -> None:
    with pytest.raises(TypeError, match="does not contain a tensor"):
        _prediction_tensor(bad)


@pytest.mark.parametrize("batch", [torch.ones(2, 2), (torch.ones(2, 2),), "batch"])
def test_batch_loss_rejects_malformed_batches(batch: Any) -> None:
    with pytest.raises(TypeError, match="at least inputs and targets"):
        _batch_loss(torch.nn.Linear(2, 1), torch.nn.MSELoss(), batch, CPU)


def test_batch_loss_uses_plain_criterion_for_non_quantile_loss() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[1.0], [1.0]])
    # predictions 2 and 4 -> squared errors 1 and 9 -> mean 5
    assert float(_batch_loss(model, torch.nn.MSELoss(), [x, y], CPU)) == pytest.approx(5.0)


def test_batch_loss_quantile_aligns_horizon_and_squeezes_target() -> None:
    criterion = AsymmetricQuantileLoss(quantiles=[0.1, 0.5, 0.9], under_penalty_factor=1.5)

    class _Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            pred = torch.zeros(x.shape[0], 3, 3)
            pred[..., 0], pred[..., 1], pred[..., 2] = 1.0, 2.0, 3.0
            return {"prediction": pred}

    y = torch.tensor([[[2.0], [9.0]], [[2.0], [9.0]]])  # (batch, horizon=2, 1)
    loss = _batch_loss(_Model(), criterion, (torch.zeros(2, 5), (y,)), CPU)

    pred = torch.tensor([1.0, 2.0, 3.0]).expand(2, 2, 3)
    assert float(loss) == pytest.approx(_pinball(pred, y.squeeze(-1), [0.1, 0.5, 0.9], 1.5))


@pytest.mark.parametrize("prediction", [[1.0, 3.0], [[1.0, 3.0]]], ids=["1d", "2d"])
def test_batch_loss_quantile_expands_low_rank_predictions(prediction: list[Any]) -> None:
    criterion = AsymmetricQuantileLoss(quantiles=[0.5], under_penalty_factor=1.0)

    class _Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tensor(prediction)

    target = np.array([2.0, 2.0], dtype=np.float32)  # non-tensor, 1-D target
    # q=0.5 -> 2 * 0.5 * |err| = |err| -> mean(|2-1|, |2-3|) = 1
    loss = _batch_loss(_Model(), criterion, (torch.zeros(1, 2), target), CPU)
    assert float(loss) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("prediction", "target", "expected"),
    [
        # median quantile [0, 3] vs [0.5, 0] -> SmoothL1(0.5)=0.125, SmoothL1(3)=2.5
        (
            torch.tensor([[[9.0, 0.0, -9.0], [9.0, 3.0, -9.0]]]),
            (torch.tensor([[0.5, 0.0]]),),
            1.3125,
        ),
        # dict output, trailing dims squeezed: diffs 0 and 2 -> (0 + 1.5) / 2
        ({"prediction": torch.tensor([[1.0], [2.0]])}, np.array([[1.0], [4.0]]), 0.75),
        # 6 predictions vs 4 targets -> first four compared: diffs 0, 0, 0, 2
        (
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 7.0, 7.0]]),
            torch.tensor([0.0, 0.0, 0.0, 2.0]),
            1.5 / 4,
        ),
    ],
    ids=["median-quantile", "dict-squeeze", "misaligned"],
)
def test_tft_aware_smooth_l1(prediction: Any, target: Any, expected: float) -> None:
    assert float(TFTAwareSmoothL1Loss()(prediction, target)) == pytest.approx(expected)


# -----------------------------------------------------------------------
# AvalancheCLStrategy wiring
# -----------------------------------------------------------------------


def test_init_builds_eval_plugin_and_log_file(tmp_path: Path, fake_avalanche) -> None:
    strategy = AvalancheCLStrategy(model=_QuantileNet(), config_path=_write_config(tmp_path))
    try:
        plugin = strategy.eval_plugin
        assert isinstance(plugin, _FakeEvaluationPlugin)
        assert [name for name, _ in plugin.metrics] == [
            "accuracy_metrics",
            "loss_metrics",
            "forgetting_metrics",
            "bwt_metrics",
            "forward_transfer_metrics",
        ]
        epoch_kwargs = {"minibatch": False, "epoch": True, "experience": True, "stream": True}
        assert plugin.metrics[0][1] == epoch_kwargs
        assert plugin.metrics[2][1] == {"experience": True, "stream": True}
        interactive, text_logger = plugin.loggers
        assert isinstance(interactive, _FakeInteractiveLogger)
        assert text_logger.file is strategy._log_file_handle
        assert (tmp_path / "logs" / "cl_log.txt").exists()
        assert strategy.dynamic_prune_threshold == pytest.approx(0.08)
        assert strategy.packnet_prune_base == pytest.approx(0.5)
        assert strategy.packnet_enabled is False
        assert strategy.intransigence_values == []
    finally:
        strategy.close()


@pytest.mark.parametrize(
    ("wrapped", "quantiles", "penalty"),
    [(False, [0.1, 0.5, 0.9], 1.5), (True, [0.2, 0.5, 0.8], 1.7)],
    ids=["raw-module", "forecaster"],
)
def test_init_binds_backbone_and_builds_criterion_from_config(
    tmp_path: Path, fake_avalanche, wrapped: bool, quantiles: list[float], penalty: float
) -> None:
    backbone = _QuantileNet()
    model: Any = backbone
    if wrapped:
        model = SimpleNamespace(
            model=backbone, config={"quantiles": quantiles, "under_penalty_factor": penalty}
        )
    strategy = AvalancheCLStrategy(model=model, config_path=_write_config(tmp_path))
    try:
        assert strategy.model is backbone and strategy.backbone is backbone
        criterion = strategy.strategy.kwargs["criterion"]
        assert isinstance(criterion, AsymmetricQuantileLoss)
        assert criterion.quantiles == quantiles
        assert criterion.under_penalty_factor == penalty
    finally:
        strategy.close()


@pytest.mark.parametrize(
    ("name", "extra"),
    [("Naive", {}), ("EWC", {"ewc_lambda": 0.4}), ("Replay", {"mem_size": 64})],
)
def test_build_strategy_passes_expected_kwargs(
    tmp_path: Path, fake_avalanche, name: str, extra: dict[str, Any]
) -> None:
    model = _QuantileNet()
    strategy = AvalancheCLStrategy(model=model, config_path=_write_config(tmp_path, strategy=name))
    try:
        built = strategy.strategy
        assert type(built) is fake_avalanche[name]
        kwargs = dict(built.kwargs)
        optimizer = kwargs.pop("optimizer")
        kwargs.pop("criterion")
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
        assert kwargs == {
            "model": model,
            "train_mb_size": 8,
            "train_epochs": 1,
            "eval_mb_size": 4,
            "evaluator": strategy.eval_plugin,
            "device": CPU,
            **extra,
        }
    finally:
        strategy.close()


def test_build_strategy_rejects_unknown_name(tmp_path: Path, fake_avalanche) -> None:
    strategy = AvalancheCLStrategy(model=_QuantileNet(), config_path=_write_config(tmp_path))
    try:
        strategy.config["strategy"] = "SI"
        with pytest.raises(ValueError, match="Unknown strategy 'SI'"):
            strategy._build_strategy()
    finally:
        strategy.close()


def test_close_is_safe_after_init_rejects_unsupported_model(tmp_path: Path, fake_avalanche):
    """Regression: close()/__del__ on a half-built instance must not raise."""
    strategy = AvalancheCLStrategy.__new__(AvalancheCLStrategy)
    with pytest.raises(TypeError, match="expects a TFTForecaster"):
        strategy.__init__(model=object(), config_path=_write_config(tmp_path))
    assert not hasattr(strategy, "_log_file_handle")
    assert not hasattr(strategy, "eval_plugin")

    strategy.close()


def test_close_releases_handle_and_tolerates_failing_loggers(tmp_path: Path, fake_avalanche):
    strategy = AvalancheCLStrategy(model=_QuantileNet(), config_path=_write_config(tmp_path))
    handle = strategy._log_file_handle
    text_logger = strategy.eval_plugin.loggers[1]

    class _BrokenLogger:
        def close(self) -> None:
            raise RuntimeError("boom")

    strategy.eval_plugin.loggers.append(_BrokenLogger())
    strategy.close()

    assert handle.closed
    assert strategy._log_file_handle is None
    assert text_logger.closed == 1

    class _BrokenHandle:
        def close(self) -> None:
            raise OSError("cannot close")

    strategy._log_file_handle = _BrokenHandle()
    strategy.close()  # must swallow handle and logger errors
    assert strategy._log_file_handle is None
    assert text_logger.closed == 2


def test_save_metrics_writes_json_payload(tmp_path: Path, fake_avalanche) -> None:
    strategy = AvalancheCLStrategy(
        model=_QuantileNet(), config_path=_write_config(tmp_path, strategy="EWC")
    )
    try:
        strategy.intransigence_values = [0.25, 0.0]
        out = tmp_path / "out" / "metrics.json"
        strategy.save_metrics(str(out))
        assert json.loads(out.read_text(encoding="utf-8")) == {
            "intransigence_per_experience": [0.25, 0.0],
            "strategy": "EWC",
        }
    finally:
        strategy.close()


# -----------------------------------------------------------------------
# Isolated loss
# -----------------------------------------------------------------------


def test_isolated_loss_trains_fresh_copy_from_initial_state(tmp_path: Path, fake_avalanche):
    torch.manual_seed(0)
    model = _QuantileNet()
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    strategy = AvalancheCLStrategy(model=model, config_path=_write_config(tmp_path))
    try:
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(5.0)
        experience = _experience(seed=1)

        # Single batch (mb_size 8 > 4 samples), one epoch -> loss before the only step.
        loss = strategy._compute_isolated_loss(experience)

        assert loss == pytest.approx(_expected_isolated_loss(initial_state, experience), rel=1e-5)
        assert all(torch.all(p == 5.0) for p in model.parameters())
    finally:
        strategy.close()


@pytest.mark.parametrize(
    "dataset",
    [TensorDataset(torch.randn(4, 2)), TensorDataset(torch.randn(4, 7), torch.randn(4, 1))],
    ids=["inputs-only", "wrong-width"],
)
def test_isolated_loss_skips_unusable_batches(
    tmp_path: Path, fake_avalanche, dataset: TensorDataset
) -> None:
    strategy = AvalancheCLStrategy(model=_QuantileNet(), config_path=_write_config(tmp_path))
    try:
        assert strategy._compute_isolated_loss(SimpleNamespace(dataset=dataset)) == 0.0
    finally:
        strategy.close()


# -----------------------------------------------------------------------
# train_all_experiences
# -----------------------------------------------------------------------


def test_train_all_experiences_flattens_and_logs_metrics(
    tmp_path: Path, fake_avalanche, monkeypatch
) -> None:
    logged: list[tuple[str, float, int]] = []
    monkeypatch.setattr(
        strategy_module,
        "mlflow",
        SimpleNamespace(
            active_run=lambda: object(),
            log_metric=lambda key, value, step: logged.append((key, value, step)),
        ),
    )
    # Joint loss 0 is below any isolated loss -> intransigence clamped to 0.
    fake_avalanche["Naive"].eval_fn = staticmethod(
        lambda idx, stream: {_loss_key(idx): 0.0, "Top1_Acc_Stream/eval": 0.5, "note": "text"}
    )
    strategy = AvalancheCLStrategy(model=_QuantileNet(), config_path=_write_config(tmp_path))
    try:
        benchmark = _benchmark()
        results = strategy.train_all_experiences(benchmark)

        flat_key = "Loss_Exp_eval_phase_test_stream_Task000_Exp001"
        assert results["experience_1"] == {
            "intransigence": 0.0,
            flat_key: 0.0,
            "Top1_Acc_Stream_eval": 0.5,
            "dynamic_pruned": 0.0,
            "adaptive_prune_ratio": 0.0,
        }
        assert strategy.intransigence_values == [0.0, 0.0]
        assert strategy.strategy.trained == benchmark.train_stream
        assert strategy.strategy.evaluated == [["test_exp_0"], ["test_exp_0", "test_exp_1"]]
        assert logged[-3:] == [
            (flat_key, 0.0, 1),
            ("Top1_Acc_Stream_eval", 0.5, 1),
            ("intransigence", 0.0, 1),
        ]
    finally:
        strategy.close()


def test_joint_loss_is_read_from_avalanche_loss_exp_metric(tmp_path: Path, fake_avalanche):
    """Regression: the joint loss comes from ``Loss_Exp/.../Exp{index:03d}``."""

    def _eval(idx: int, stream: Any) -> dict[str, Any]:
        results: dict[str, Any] = {_loss_key(j): 100.0 + j for j in range(idx + 1)}
        results["Loss_Stream/eval_phase/test_stream/Task000"] = 999.0
        return results

    fake_avalanche["Naive"].eval_fn = staticmethod(_eval)
    torch.manual_seed(0)
    model = _QuantileNet()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    strategy = AvalancheCLStrategy(model=model, config_path=_write_config(tmp_path))
    try:
        benchmark = _benchmark(n=3)
        strategy.train_all_experiences(benchmark)

        expected = [
            100.0 + i - _expected_isolated_loss(state, exp)
            for i, exp in enumerate(benchmark.train_stream)
        ]
        assert strategy.intransigence_values == pytest.approx(expected, rel=1e-5)
    finally:
        strategy.close()


@pytest.mark.parametrize(
    "eval_results",
    [
        {"Loss_Exp/experience_0": 50.0},  # legacy key format
        {_loss_key(1): 50.0},  # another experience
        {"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000": 50.0},  # not a loss
        {_loss_key(0): "50.0"},  # non-numeric value
    ],
    ids=["legacy-key", "other-experience", "accuracy", "non-numeric"],
)
def test_joint_loss_falls_back_to_isolated_loss(
    tmp_path: Path, fake_avalanche, eval_results: dict[str, Any]
) -> None:
    fake_avalanche["Naive"].eval_fn = staticmethod(lambda idx, stream: eval_results)
    strategy = AvalancheCLStrategy(model=_QuantileNet(), config_path=_write_config(tmp_path))
    try:
        results = strategy.train_all_experiences(_benchmark(n=1))
        assert results["experience_0"]["intransigence"] == 0.0
    finally:
        strategy.close()


def test_dynamic_packnet_pruning_computes_ratio_once(
    tmp_path: Path, fake_avalanche, monkeypatch
) -> None:
    """Regression: dynamic pruning reuses ``last_prune_ratio`` (one scheduler update)."""
    created: list[packnet_module.PackNetStrategy] = []

    class _RecordingPackNet(packnet_module.PackNetStrategy):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(packnet_module, "PackNetStrategy", _RecordingPackNet)
    fake_avalanche["Naive"].eval_fn = staticmethod(
        lambda idx, stream: {_loss_key(idx): 50.0 if idx == 0 else 0.0}
    )
    torch.manual_seed(0)
    model = _QuantileNet()
    config = _write_config(tmp_path, enable_packnet_scheduler=True)
    strategy = AvalancheCLStrategy(model=model, config_path=config)
    try:
        results = strategy.train_all_experiences(_benchmark())

        (packnet,) = created
        first, second = results["experience_0"], results["experience_1"]
        assert packnet.scheduler.intransigence_history == [first["intransigence"]]
        # I / I_max = 1 -> 0.5 * (1 - 0.8) = 0.1
        assert first["adaptive_prune_ratio"] == pytest.approx(0.1)
        assert first["dynamic_pruned"] == 1.0
        assert first["intransigence_critical_alert"] == 1.0
        assert (second["dynamic_pruned"], second["adaptive_prune_ratio"]) == (0.0, 0.0)
        assert second["intransigence_critical_alert"] == 0.0
        # floor(0.1 * n) pruned per tensor: only the 12-weight body.1.weight loses one.
        assert packnet.capacity_used() == pytest.approx(26 / 27)
        assert int((model.body[1].weight == 0).sum()) == 1
    finally:
        strategy.close()


@pytest.mark.parametrize(
    ("fail_at", "warning", "alert"),
    [
        ("init", "PackNet scheduler unavailable", None),
        ("prune", "Dynamic PackNet pruning failed", 1.0),
    ],
)
def test_train_all_experiences_tolerates_packnet_failures(
    tmp_path: Path,
    fake_avalanche,
    monkeypatch,
    caplog,
    fail_at: str,
    warning: str,
    alert: float | None,
) -> None:
    class _FailingPackNet:
        def __init__(self, model: Any, prune_ratio: float, adaptive_scheduler: Any) -> None:
            if fail_at == "init":
                raise RuntimeError("no packnet")
            self.scheduler = adaptive_scheduler

        def apply_masks(self) -> None:
            raise RuntimeError("mask failure")

        def prune_and_freeze(self, task_id: int, intransigence: float) -> None:
            raise RuntimeError("prune failure")

    monkeypatch.setattr(packnet_module, "PackNetStrategy", _FailingPackNet)
    fake_avalanche["Naive"].eval_fn = staticmethod(lambda idx, stream: {_loss_key(0): 50.0})
    config = _write_config(tmp_path, enable_packnet_scheduler=True)
    strategy = AvalancheCLStrategy(model=_QuantileNet(), config_path=config)
    try:
        with caplog.at_level("WARNING", logger="src.cl.strategy"):
            flat = strategy.train_all_experiences(_benchmark(n=1))["experience_0"]
        assert flat["intransigence"] > 0.08
        assert (flat["dynamic_pruned"], flat["adaptive_prune_ratio"]) == (0.0, 0.0)
        assert flat.get("intransigence_critical_alert") == alert
        assert warning in caplog.text
    finally:
        strategy.close()
