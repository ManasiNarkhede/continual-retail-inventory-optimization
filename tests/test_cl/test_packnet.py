"""Unit tests for PackNet strategy helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.cl.packnet import AdaptivePruningScheduler, PackNetStrategy


def test_packnet_capacity_bounds() -> None:
    """Capacity usage should stay within [0, 1]."""
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
    packnet = PackNetStrategy(model=model, prune_ratio=0.5)
    packnet.masks[0] = {
        name: torch.ones_like(param) for name, param in model.named_parameters() if param.dim() >= 1
    }
    value = packnet.capacity_used()
    assert 0.0 <= value <= 1.0


def test_adaptive_pruning_scheduler_ratio_changes_with_intransigence() -> None:
    """Adaptive pruning ratio should reduce as intransigence increases."""
    scheduler = AdaptivePruningScheduler(base_prune_ratio=0.5, sensitivity=0.8)
    low = scheduler.compute_ratio(0.01)
    high = scheduler.compute_ratio(0.50)
    assert high <= low
