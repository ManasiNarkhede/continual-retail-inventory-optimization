"""Hand-computed unit tests for continual learning metric helpers."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import src.cl.metrics as cl_metrics
from src.cl.metrics import (
    compute_bwt,
    compute_fwt,
    compute_plasticity,
    compute_stability,
    plot_plasticity_stability_tradeoff,
)

MATRIX = np.array(
    [
        [0.80, 0.50, 0.40],
        [0.75, 0.82, 0.55],
        [0.72, 0.80, 0.84],
    ]
)
SINGLE = np.array([[0.9]])
fwt = partial(compute_fwt, baseline=np.array([0.40, 0.45, 0.50]))


@pytest.mark.parametrize(
    ("metric", "matrix", "expected"),
    [
        # mean(R[2,0] - R[0,0], R[2,1] - R[1,1]) = mean(-0.08, -0.02)
        (compute_bwt, MATRIX, -0.05),
        # mean(R[0,1] - b[1], R[1,2] - b[2]) = mean(0.05, 0.05)
        (fwt, MATRIX, 0.05),
        (compute_plasticity, MATRIX, (0.80 + 0.82 + 0.84) / 3),
        # retained = mean(0.72, 0.80) = 0.76, original = mean(0.80, 0.82) = 0.81
        (compute_stability, MATRIX, 0.76 / 0.81),
        (compute_stability, np.array([[0.0, 0.0], [0.5, 0.3]]), 0.0),  # zero original
        (compute_bwt, SINGLE, 0.0),
        (fwt, SINGLE, 0.0),
        (compute_plasticity, SINGLE, 0.9),
        (compute_stability, SINGLE, 1.0),
    ],
    ids=[
        "bwt",
        "fwt",
        "plasticity",
        "stability",
        "stability-zero-original",
        "bwt-single-task",
        "fwt-single-task",
        "plasticity-single-task",
        "stability-single-task",
    ],
)
def test_metric_matches_hand_computation(
    metric: Callable[[np.ndarray], float], matrix: np.ndarray, expected: float
) -> None:
    value = metric(matrix)
    assert isinstance(value, float)
    assert value == pytest.approx(expected)


def test_plot_tradeoff_writes_png_with_expected_series(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    real_savefig = cl_metrics.plt.savefig

    def _capturing_savefig(*args, **kwargs):  # noqa: ANN002, ANN003
        ax = cl_metrics.plt.gca()
        captured["lines"] = [
            (line.get_label(), list(line.get_xdata()), list(line.get_ydata()))
            for line in ax.get_lines()
        ]
        captured["xlabel"] = ax.get_xlabel()
        captured["title"] = ax.get_title()
        captured["dpi"] = kwargs.get("dpi")
        return real_savefig(*args, **kwargs)

    monkeypatch.setattr(cl_metrics.plt, "savefig", _capturing_savefig)

    target = tmp_path / "nested" / "dir" / "tradeoff.png"
    result = plot_plasticity_stability_tradeoff(
        thresholds=[0.05, 0.15, 0.25],
        plasticity_values=[0.9, 0.8, 0.7],
        stability_values=[0.6, 0.7, 0.8],
        output_path=str(target),
    )

    assert result == target
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert captured["lines"] == [
        ("Plasticity", [0.05, 0.15, 0.25], [0.9, 0.8, 0.7]),
        ("Stability", [0.05, 0.15, 0.25], [0.6, 0.7, 0.8]),
        ("Capacity guard 15%", [0.15, 0.15], [0, 1]),
    ]
    assert captured["xlabel"] == "capacity_safety_threshold"
    assert captured["title"] == "Plasticity vs Stability Trade-off"
    assert captured["dpi"] == 180
    assert cl_metrics.plt.get_fignums() == []  # figure closed after saving
