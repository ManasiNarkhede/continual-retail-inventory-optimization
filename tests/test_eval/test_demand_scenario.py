"""Tests for demand scenario Monte Carlo helpers."""

from __future__ import annotations

import numpy as np

from src.eval.demand_scenario import build_scenario_batch, triangular_split_sample


def test_triangular_split_sample_length_and_non_negative() -> None:
    p10 = np.array([10.0, 11.0, 12.0], dtype=float)
    p50 = np.array([12.0, 13.0, 14.0], dtype=float)
    p90 = np.array([15.0, 16.0, 17.0], dtype=float)

    out = triangular_split_sample(p10=p10, p50=p50, p90=p90, days=7, rng=np.random.default_rng(1))
    assert out.shape == (7,)
    assert np.all(out >= 0.0)


def test_build_scenario_batch_count() -> None:
    p10 = np.array([10.0, 11.0, 12.0], dtype=float)
    p50 = np.array([12.0, 13.0, 14.0], dtype=float)
    p90 = np.array([15.0, 16.0, 17.0], dtype=float)

    batch = build_scenario_batch(p10=p10, p50=p50, p90=p90, n_days=5, n_scenarios=4, seed=7)
    assert len(batch) == 4
    assert all(item.shape == (5,) for item in batch)
