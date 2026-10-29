"""Device resolution helpers for CPU/CUDA/MPS runtimes."""

from __future__ import annotations

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Resolve best available torch device.

    Args:
        preferred: Requested device string among ``auto``, ``cpu``, ``cuda``, ``mps``.

    Returns:
        Torch device object with availability fallback.
    """
    request = (preferred or "auto").strip().lower()
    if request == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request == "mps":
        has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
        return torch.device("mps" if has_mps else "cpu")
    if request == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")
