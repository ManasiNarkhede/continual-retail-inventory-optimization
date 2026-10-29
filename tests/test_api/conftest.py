"""Shared API test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.limiter import limiter
from src.api.main import app
from src.api.model_registry import clear_registry


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    """Reset in-memory rate limiter state around each API test."""
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture()
def isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run in an empty cwd (no repo data/checkpoints) with a clean registry singleton."""
    monkeypatch.chdir(tmp_path)
    clear_registry()
    yield
    clear_registry()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class FakeTensor:
    """Minimal torch-like tensor wrapper over a numpy array."""

    def __init__(self, values: Any) -> None:
        self._arr = np.asarray(values, dtype=float)

    def __getitem__(self, key: Any) -> FakeTensor:
        return FakeTensor(self._arr[key])

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr
