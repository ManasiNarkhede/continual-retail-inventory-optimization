"""Continual learning metric computation utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_bwt(results_matrix: np.ndarray) -> float:
    """Compute Backward Transfer (forgetting metric).

    Args:
        results_matrix: T x T matrix where R[j, i] is performance on task i
            after training task j.

    Returns:
        Backward transfer score.
    """
    task_count = results_matrix.shape[0]
    if task_count < 2:
        return 0.0
    return float(
        np.mean(
            [
                results_matrix[task_count - 1, i] - results_matrix[i, i]
                for i in range(task_count - 1)
            ]
        )
    )


def compute_fwt(results_matrix: np.ndarray, baseline: np.ndarray) -> float:
    """Compute Forward Transfer.

    Args:
        results_matrix: T x T CL results matrix.
        baseline: Baseline vector with shape (T,).

    Returns:
        Forward transfer score.
    """
    task_count = results_matrix.shape[0]
    if task_count < 2:
        return 0.0
    return float(np.mean([results_matrix[i - 1, i] - baseline[i] for i in range(1, task_count)]))


def compute_plasticity(results_matrix: np.ndarray) -> float:
    """Compute plasticity as average diagonal performance.

    Args:
        results_matrix: T x T CL results matrix.

    Returns:
        Plasticity score.
    """
    task_count = results_matrix.shape[0]
    return float(np.mean([results_matrix[i, i] for i in range(task_count)]))


def compute_stability(results_matrix: np.ndarray) -> float:
    """Compute stability ratio from retained versus original performance.

    Args:
        results_matrix: T x T CL results matrix.

    Returns:
        Stability ratio.
    """
    task_count = results_matrix.shape[0]
    if task_count < 2:
        return 1.0
    retained = np.mean([results_matrix[task_count - 1, i] for i in range(task_count - 1)])
    original = np.mean([results_matrix[i, i] for i in range(task_count - 1)])
    return float(retained / original) if original > 0 else 0.0


def plot_plasticity_stability_tradeoff(
    thresholds: list[float],
    plasticity_values: list[float],
    stability_values: list[float],
    output_path: str = "outputs/ablation/plasticity_stability_tradeoff.png",
) -> Path:
    """Generate plasticity-stability trade-off plot.

    Args:
        thresholds: Capacity safety thresholds used in experiments.
        plasticity_values: Plasticity values aligned with thresholds.
        stability_values: Stability values aligned with thresholds.
        output_path: Destination PNG path.

    Returns:
        Path to saved figure.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, plasticity_values, marker="o", label="Plasticity", color="#0f766e")
    plt.plot(thresholds, stability_values, marker="s", label="Stability", color="#d97706")
    plt.axvline(0.15, linestyle="--", color="#334155", linewidth=1.0, label="Capacity guard 15%")
    plt.xlabel("capacity_safety_threshold")
    plt.ylabel("Score")
    plt.title("Plasticity vs Stability Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out
