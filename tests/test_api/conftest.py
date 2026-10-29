"""Shared API test fixtures."""

from __future__ import annotations

import pytest

from src.api.limiter import limiter


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    """Reset in-memory rate limiter state around each API test."""
    limiter.reset()
    yield
    limiter.reset()
