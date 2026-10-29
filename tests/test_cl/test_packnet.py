"""Unit tests for element-wise PackNet pruning and the adaptive pruning scheduler."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import src.cl.packnet as packnet_module
from src.cl.packnet import AdaptivePruningScheduler, PackNetStrategy


@pytest.fixture(autouse=True)
def _no_mlflow_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(packnet_module, "mlflow", SimpleNamespace(active_run=lambda: None))


def _linear(values: list[float]) -> nn.Linear:
    layer = nn.Linear(len(values), 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([values]))
    return layer


# -----------------------------------------------------------------------
# AdaptivePruningScheduler
# -----------------------------------------------------------------------


def test_compute_ratio_follows_formula_with_running_max() -> None:
    scheduler = AdaptivePruningScheduler(base_prune_ratio=0.5, sensitivity=0.8)
    # I / I_max = 1 -> 0.5 * (1 - 0.8) = 0.1
    assert scheduler.compute_ratio(0.2) == pytest.approx(0.1)
    # I_max stays 0.2, I / I_max = 0.5 -> 0.5 * (1 - 0.4) = 0.3
    assert scheduler.compute_ratio(0.1) == pytest.approx(0.3)
    # Negative intransigence is clamped to 0 -> base ratio
    assert scheduler.compute_ratio(-1.0) == pytest.approx(0.5)
    assert scheduler.intransigence_history == [0.2, 0.1, -1.0]


@pytest.mark.parametrize(
    ("kwargs", "intransigence", "expected"),
    [
        ({"base_prune_ratio": 1.0, "max_prune_ratio": 0.75}, 0.0, 0.75),  # 1.0 -> ceiling
        ({"base_prune_ratio": 0.5, "sensitivity": 1.0, "min_prune_ratio": 0.1}, 0.3, 0.1),  # 0
    ],
    ids=["ceiling", "floor"],
)
def test_compute_ratio_is_clamped(
    kwargs: dict[str, float], intransigence: float, expected: float
) -> None:
    assert AdaptivePruningScheduler(**kwargs).compute_ratio(intransigence) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(("used", "allowed"), [(0.5, True), (0.85, True), (0.9, False)])
def test_should_prune_respects_capacity_safety_threshold(used: float, allowed: bool) -> None:
    # Remaining capacity of exactly 0.15 is not below the threshold.
    assert AdaptivePruningScheduler(capacity_safety_threshold=0.15).should_prune(used) is allowed


# -----------------------------------------------------------------------
# PackNetStrategy
# -----------------------------------------------------------------------


def test_default_scheduler_uses_prune_ratio() -> None:
    packnet = PackNetStrategy(model=_linear([1.0, 2.0]), prune_ratio=0.3, retrain_epochs=2)
    assert packnet.scheduler.base_prune_ratio == pytest.approx(0.3)
    assert packnet.retrain_epochs == 2
    assert packnet.capacity_used() == 0.0
    assert packnet.assigned == {} and packnet.last_prune_ratio == 0.0


def test_prune_and_freeze_assigns_largest_weights_until_capacity_guard() -> None:
    model = _linear([1.0, -2.0, 3.0, -4.0])
    scheduler = AdaptivePruningScheduler(base_prune_ratio=0.5, capacity_safety_threshold=0.6)
    packnet = PackNetStrategy(model=model, adaptive_scheduler=scheduler)

    stats = packnet.prune_and_freeze(task_id=0, intransigence=0.0)

    # floor(0.5 * 4) = 2 smallest-magnitude weights pruned, the other two owned by task 0.
    assert stats == {"weight": pytest.approx(0.5)}
    assert packnet.last_prune_ratio == pytest.approx(0.5)
    assert torch.equal(model.weight.data, torch.tensor([[0.0, 0.0, 3.0, -4.0]]))
    assert torch.equal(packnet.masks[0]["weight"], torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    assert torch.equal(packnet.assigned["weight"], torch.tensor([[False, False, True, True]]))
    assert torch.equal(packnet.frozen_values["weight"], torch.tensor([[0.0, 0.0, 3.0, -4.0]]))
    assert packnet.layer_prune_history == {0: {"weight": pytest.approx(0.5)}}
    assert packnet.capacity_used() == pytest.approx(0.5)

    # Remaining capacity 0.5 < 0.6 safety threshold -> nothing is pruned or assigned.
    assert packnet.prune_and_freeze(task_id=1, intransigence=0.0) == {}
    assert 1 not in packnet.masks
    assert len(packnet.scheduler.intransigence_history) == 2
    assert torch.equal(model.weight.data, torch.tensor([[0.0, 0.0, 3.0, -4.0]]))


def test_prune_and_freeze_second_task_assigns_previously_free_weights() -> None:
    """Regression: later tasks claim free weights instead of skipping whole tensors."""
    model = _linear([float(v) for v in range(1, 11)])
    scheduler = AdaptivePruningScheduler(base_prune_ratio=0.5, sensitivity=0.0)
    packnet = PackNetStrategy(model=model, adaptive_scheduler=scheduler)

    packnet.prune_and_freeze(task_id=0)
    owned_0 = packnet.masks[0]["weight"].bool()
    assert int(owned_0.sum()) == 5  # weights 6..10
    assert packnet.capacity_used() == pytest.approx(0.5)

    # Simulate training of the (zeroed) free weights between tasks.
    with torch.no_grad():
        model.weight[~owned_0] = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1])

    stats = packnet.prune_and_freeze(task_id=1)

    owned_1 = packnet.masks[1]["weight"].bool()
    # floor(0.5 * 5 free) = 2 pruned (0.2, 0.1), 3 assigned (0.5, 0.4, 0.3).
    assert stats == {"weight": pytest.approx(0.4)}
    assert int(owned_1.sum()) == 3
    assert not bool((owned_0 & owned_1).any())
    assert packnet.capacity_used() == pytest.approx(0.8)
    expected = torch.tensor([[0.5, 0.4, 0.3, 0.0, 0.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
    assert torch.equal(model.weight.data, expected)
    assert torch.equal(packnet.frozen_values["weight"], expected)
    assert torch.equal(packnet.assigned["weight"], owned_0 | owned_1)


def test_prune_and_freeze_skips_scalar_non_trainable_and_fully_assigned_params() -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(2.0))
            self.fixed = nn.Parameter(torch.tensor([1.0, 2.0]), requires_grad=False)
            self.bias = nn.Parameter(torch.tensor([2.0]))

    model = _Model()
    packnet = PackNetStrategy(model=model, prune_ratio=0.5)

    # floor(0.5 * 1) = 0 -> the single-element bias is kept whole and owned by task 0.
    assert packnet.prune_and_freeze(task_id=0) == {"bias": 0.0}
    assert torch.equal(packnet.masks[0]["bias"], torch.tensor([1.0]))
    # Only the bias is assignable, so capacity is full and the guard skips task 1.
    assert packnet.capacity_used() == 1.0
    assert packnet.prune_and_freeze(task_id=1) == {}
    assert 1 not in packnet.masks and 1 not in packnet.layer_prune_history
    assert float(model.scale) == 2.0
    assert torch.equal(model.fixed.data, torch.tensor([1.0, 2.0]))
    assert torch.equal(model.bias.data, torch.tensor([2.0]))


def test_prune_and_freeze_logs_layer_ratios_to_active_mlflow_run(monkeypatch) -> None:
    logged: list[tuple[str, float, int]] = []
    fake_mlflow = SimpleNamespace(
        active_run=lambda: object(),
        log_metric=lambda name, value, step: logged.append((name, value, step)),
    )
    monkeypatch.setattr(packnet_module, "mlflow", fake_mlflow)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))

    PackNetStrategy(model=model, prune_ratio=0.5).prune_and_freeze(task_id=3)

    # floor(0.5 * n) pruned per tensor: 12 -> 6, 4 -> 2, 4 -> 2, 1 -> 0.
    assert logged == [
        ("packnet_layer_prune_ratio_0_weight", 0.5, 3),
        ("packnet_layer_prune_ratio_0_bias", 0.5, 3),
        ("packnet_layer_prune_ratio_2_weight", 0.5, 3),
        ("packnet_layer_prune_ratio_2_bias", 0.0, 3),
    ]


def test_apply_masks_restores_frozen_values_and_leaves_free_weights() -> None:
    model = _linear([1.0, -2.0, 3.0, -4.0])
    packnet = PackNetStrategy(model=model, prune_ratio=0.5)
    packnet.prune_and_freeze(task_id=0)
    with torch.no_grad():
        model.weight.fill_(5.0)

    packnet.apply_masks()

    assert torch.equal(model.weight.data, torch.tensor([[5.0, 5.0, 3.0, -4.0]]))


def test_train_task_updates_only_free_weights() -> None:
    model = nn.Linear(3, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -1.0, 2.0]]))
        model.bias.fill_(0.25)
    packnet = PackNetStrategy(model=model, prune_ratio=0.5)
    packnet.prune_and_freeze(task_id=0)  # weight 0.5 pruned (free); -1, 2 and bias owned
    free = ~packnet.assigned["weight"]
    assert free.tolist() == [[True, False, False]]
    before: dict[str, Any] = {n: p.detach().clone() for n, p in model.named_parameters()}

    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]])
    y = torch.tensor([1.0, 0.0])
    # predictions: -2 + 6 + 0.25 = 4.25 and 2 + 0.25 = 2.25 -> MSE = (3.25^2 + 2.25^2) / 2
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = packnet.train_task(loader, optimizer, nn.MSELoss(), task_id=1, epochs=1)

    assert loss == pytest.approx((3.25**2 + 2.25**2) / 2)
    assert torch.equal(model.weight.data[~free], before["weight"][~free])
    assert torch.equal(model.bias.data, before["bias"])
    # d(MSE)/dw0 = mean(2 * err * x0) = (2*3.25*1 + 2*2.25*2) / 2 = 7.75 -> w0 = -0.775
    assert float(model.weight.data[0, 0]) == pytest.approx(-0.775)
