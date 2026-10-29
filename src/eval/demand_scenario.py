"""Demand scenario generators used by Monte Carlo simulation paths."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def triangular_split_sample(
    p10: npt.NDArray[np.float64],
    p50: npt.NDArray[np.float64],
    p90: npt.NDArray[np.float64],
    days: int,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Sample demand path from p10/p50/p90 quantiles using split-triangle logic."""
    generator = rng or np.random.default_rng()
    n_days = max(int(days), 1)
    if len(p50) == 0:
        return np.zeros(n_days, dtype=np.float64)

    samples: list[float] = []
    for t in range(n_days):
        idx = min(t, len(p50) - 1)
        low = float(p10[idx])
        mid = float(p50[idx])
        high = float(p90[idx])
        u = float(generator.random())
        if u < 0.25:
            value = low + (mid - low) * np.sqrt(max(u / 0.25, 0.0))
        else:
            value = high - (high - mid) * np.sqrt(max((1.0 - u) / 0.75, 0.0))
        samples.append(max(0.0, float(value)))
    return np.asarray(samples, dtype=np.float64)


def build_scenario_batch(
    p10: npt.NDArray[np.float64],
    p50: npt.NDArray[np.float64],
    p90: npt.NDArray[np.float64],
    n_days: int,
    n_scenarios: int,
    seed: int | None = None,
) -> list[npt.NDArray[np.float64]]:
    """Generate multiple Monte Carlo demand scenarios."""
    generator = np.random.default_rng(seed)
    return [
        triangular_split_sample(p10=p10, p50=p50, p90=p90, days=n_days, rng=generator)
        for _ in range(max(int(n_scenarios), 1))
    ]
