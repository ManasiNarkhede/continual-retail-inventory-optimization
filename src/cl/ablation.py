"""Continual-learning strategy ablation runner.

This module compares Avalanche strategies on the same experience stream and
outputs a compact metrics table suitable for dashboarding or papers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import mlflow
import numpy as np
import pandas as pd

from src.cl.metrics import compute_bwt, compute_fwt, compute_plasticity, compute_stability
from src.models.tft_model import TFTForecaster

logger = logging.getLogger(__name__)

STRATEGIES_TO_TEST = ["EWC", "Replay", "Naive"]


class CLAblationRunner:
    """Run CL strategy comparisons on one benchmark.

    Args:
        config_path: Path to CL YAML config.
        output_dir: Directory where result artefacts are saved.
    """

    def __init__(
        self, config_path: str = "configs/cl_config.yaml", output_dir: str = "outputs/ablation"
    ) -> None:
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, model_factory: Callable[[], object], benchmark: object) -> pd.DataFrame:
        """Execute configured strategy ablations.

        Args:
            model_factory: Callable creating a fresh model for each strategy.
            benchmark: Avalanche benchmark object.

        Returns:
            DataFrame containing summary metrics per strategy.
        """
        from src.cl.strategy import AvalancheCLStrategy

        records: list[dict[str, float | str]] = []
        n_tasks = len(benchmark.train_stream)

        for strategy_name in STRATEGIES_TO_TEST:
            logger.info("Running CL ablation for strategy=%s", strategy_name)
            model = model_factory()
            strategy_model = model
            if isinstance(model, TFTForecaster):
                strategy_model = model
            elif (
                isinstance(model, tuple) and len(model) > 0 and isinstance(model[0], TFTForecaster)
            ):
                strategy_model = model[0]
            cl = AvalancheCLStrategy(model=strategy_model, config_path=self.config_path)
            cl.config["strategy"] = strategy_name
            cl.strategy = cl._build_strategy()

            results_matrix = np.zeros((n_tasks, n_tasks), dtype=float)
            started = time.perf_counter()

            with mlflow.start_run(run_name=f"cl_ablation_{strategy_name}", nested=True):
                mlflow.log_param("strategy", strategy_name)
                for i, experience in enumerate(benchmark.train_stream):
                    cl.strategy.train(experience)
                    for j in range(i + 1):
                        eval_res = cl.strategy.eval([benchmark.test_stream[j]])
                        numeric_values = [
                            float(v) for v in eval_res.values() if isinstance(v, (float, int))
                        ]
                        results_matrix[i, j] = numeric_values[0] if numeric_values else 0.0

            elapsed = time.perf_counter() - started
            baseline = np.zeros(n_tasks, dtype=float)
            record = {
                "strategy": strategy_name,
                "bwt": float(compute_bwt(results_matrix)),
                "fwt": float(compute_fwt(results_matrix, baseline)),
                "plasticity": float(compute_plasticity(results_matrix)),
                "stability": float(compute_stability(results_matrix)),
                "final_mean_acc": float(np.mean(results_matrix[-1, :])),
                "train_time_s": float(round(elapsed, 2)),
            }
            records.append(record)

        return pd.DataFrame(records)

    def save_results(self, results_df: pd.DataFrame) -> Path:
        """Save ablation output CSV and optionally log it to MLflow.

        Args:
            results_df: Ablation DataFrame from ``run``.

        Returns:
            Path to saved CSV.
        """
        output_path = self.output_dir / "cl_ablation_results.csv"
        results_df.to_csv(output_path, index=False)
        if mlflow.active_run() is not None:
            mlflow.log_artifact(str(output_path))
        return output_path

    def run_distribution_shift_comparison(
        self, model_factory: Callable[[], object], benchmark: object
    ) -> dict[str, object]:
        """Compare Adaptive PackNet vs Standard TFT under shift benchmark.

        Args:
            model_factory: Callable returning a fresh TFT backbone.
            benchmark: Distribution-shift Avalanche benchmark.

        Returns:
            JSON-serializable metric payload with intransigence and BWT.
        """
        from src.cl.strategy import AvalancheCLStrategy

        def _collect(strategy_name: str, packnet_enabled: bool) -> dict[str, float]:
            model = model_factory()
            strategy_model = model
            if isinstance(model, TFTForecaster):
                strategy_model = model
            elif (
                isinstance(model, tuple) and len(model) > 0 and isinstance(model[0], TFTForecaster)
            ):
                strategy_model = model[0]
            cl = AvalancheCLStrategy(model=strategy_model, config_path=self.config_path)
            cl.packnet_enabled = bool(packnet_enabled)
            cl_results = cl.train_all_experiences(benchmark)

            i_values = [float(metrics.get("intransigence", 0.0)) for metrics in cl_results.values()]
            bwt_values = []
            for metrics in cl_results.values():
                for key, value in metrics.items():
                    if "bwt" in str(key).lower() and isinstance(value, (float, int)):
                        bwt_values.append(float(value))

            metrics = {
                "intransigence_mean": float(np.mean(i_values)) if i_values else 0.0,
                "bwt": float(np.mean(bwt_values)) if bwt_values else 0.0,
            }

            if mlflow.active_run() is not None:
                mlflow.log_metric(
                    f"{strategy_name}_intransigence_mean", metrics["intransigence_mean"]
                )
                mlflow.log_metric(f"{strategy_name}_bwt", metrics["bwt"])
            return metrics

        adaptive = _collect("adaptive_packnet", packnet_enabled=True)
        standard = _collect("standard_tft", packnet_enabled=False)

        payload: dict[str, object] = {
            "strategy": "PromoShift AdaptivePackNet vs Standard",
            "adaptive_packnet": {
                "intransigence_mean": round(float(adaptive["intransigence_mean"]), 4),
                "bwt": round(float(adaptive["bwt"]), 4),
            },
            "standard_tft": {
                "intransigence_mean": round(float(standard["intransigence_mean"]), 4),
                "bwt": round(float(standard["bwt"]), 4),
            },
            "delta": {
                "intransigence_improvement": round(
                    float(standard["intransigence_mean"] - adaptive["intransigence_mean"]),
                    4,
                ),
                "bwt_improvement": round(float(adaptive["bwt"] - standard["bwt"]), 4),
            },
        }
        return payload

    def save_distribution_shift_metrics(
        self,
        payload: dict[str, object],
        output_path: str = "data/processed/cl_metrics_latest.json",
    ) -> Path:
        """Persist distribution-shift CL metrics JSON for dashboard/API usage."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if mlflow.active_run() is not None:
            mlflow.log_artifact(str(out))
        return out
