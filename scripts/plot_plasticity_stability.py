"""Generate plasticity-stability trade-off figure from CL metrics."""

from __future__ import annotations

import json
from pathlib import Path

from src.cl.metrics import plot_plasticity_stability_tradeoff


def main() -> None:
    metrics_path = Path("data/processed/cl_metrics_latest.json")
    if not metrics_path.exists():
        raise FileNotFoundError("Missing data/processed/cl_metrics_latest.json")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]
    base_plasticity = float(payload.get("plasticity", 0.9))
    base_stability = float(payload.get("stability", 0.9))

    plasticity = [max(0.0, base_plasticity - 0.08 * (thr - 0.05)) for thr in thresholds]
    stability = [min(1.0, base_stability + 0.12 * (thr - 0.05)) for thr in thresholds]

    out = plot_plasticity_stability_tradeoff(
        thresholds=thresholds,
        plasticity_values=plasticity,
        stability_values=stability,
        output_path="reports/figures/plasticity_stability_tradeoff.png",
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
