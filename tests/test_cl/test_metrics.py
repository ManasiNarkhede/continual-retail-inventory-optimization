"""Unit tests for continual learning metrics."""

from __future__ import annotations

import numpy as np

from src.cl.metrics import compute_bwt, compute_fwt, compute_plasticity, compute_stability


def test_compute_bwt_returns_float() -> None:
    """Validate BWT output type and finite value."""
    matrix = np.array(
        [
            [0.80, 0.00, 0.00],
            [0.75, 0.82, 0.00],
            [0.72, 0.80, 0.84],
        ]
    )
    value = compute_bwt(matrix)
    assert isinstance(value, float)
    assert np.isfinite(value)


def test_compute_fwt_returns_float() -> None:
    """Validate FWT output type and finite value."""
    matrix = np.array(
        [
            [0.80, 0.50, 0.40],
            [0.75, 0.82, 0.55],
            [0.72, 0.80, 0.84],
        ]
    )
    baseline = np.array([0.40, 0.45, 0.50])
    value = compute_fwt(matrix, baseline)
    assert isinstance(value, float)
    assert np.isfinite(value)


def test_compute_plasticity_matches_diagonal_mean() -> None:
    """Validate plasticity against direct diagonal computation."""
    matrix = np.array([[0.8, 0.0], [0.7, 0.9]])
    value = compute_plasticity(matrix)
    assert np.isclose(value, np.mean([0.8, 0.9]))


def test_compute_stability_single_task_is_one() -> None:
    """Validate stability edge case when only one task exists."""
    matrix = np.array([[0.8]])
    assert compute_stability(matrix) == 1.0
