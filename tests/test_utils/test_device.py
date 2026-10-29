"""Tests for device auto-detection helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from src.utils.device import get_device


def test_get_device_auto_prefers_cuda(monkeypatch) -> None:
    """Auto mode should choose CUDA when available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.backends, "mps", SimpleNamespace(is_available=lambda: True), raising=False
    )
    assert get_device("auto").type == "cuda"


def test_get_device_auto_uses_mps_when_cuda_unavailable(monkeypatch) -> None:
    """Auto mode should pick MPS when CUDA is unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends, "mps", SimpleNamespace(is_available=lambda: True), raising=False
    )
    assert get_device("auto").type == "mps"


def test_get_device_mps_falls_back_to_cpu(monkeypatch) -> None:
    """Explicit MPS request should fallback to CPU when unavailable."""
    monkeypatch.setattr(
        torch.backends, "mps", SimpleNamespace(is_available=lambda: False), raising=False
    )
    assert get_device("mps").type == "cpu"


def test_get_device_cuda_falls_back_to_cpu(monkeypatch) -> None:
    """Explicit CUDA request should fallback to CPU when unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device("cuda").type == "cpu"
