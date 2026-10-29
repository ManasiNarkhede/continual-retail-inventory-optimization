"""Reward sensitivity sweep utilities for RL risk-vs-reward analysis."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.eval.harness import EvaluationHarness


def run_reward_sensitivity_sweep(
    val_df: pd.DataFrame,
    output_dir: str = "outputs/evaluation",
    severities: list[float] | None = None,
) -> Path:
    """Run stockout severity sweep and log Pareto frontier metrics.

    Args:
        val_df: Validation dataframe containing demand columns.
        output_dir: Output directory path.
        severities: Optional severity grid.

    Returns:
        Path to saved sweep CSV.
    """
    harness = EvaluationHarness(output_dir=output_dir, log_to_mlflow=True)
    if severities is None:
        severities = [round(value, 3) for value in np.linspace(0.01, 0.1, 10)]

    result_df = harness.stress_test_stockout_severity(
        val_df=val_df,
        forecaster=True,
        rl_agent=True,
        severities=severities,
        output_filename="risk_reward_pareto.csv",
    )

    # Pareto approximation: non-dominated points for (cost minimize, service maximize)
    pareto_mask = []
    rows = result_df.to_dict(orient="records")
    for i, candidate in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            better_or_equal = float(other["total_cost"]) <= float(
                candidate["total_cost"]
            ) and float(other["service_level"]) >= float(candidate["service_level"])
            strictly_better = float(other["total_cost"]) < float(candidate["total_cost"]) or float(
                other["service_level"]
            ) > float(candidate["service_level"])
            if better_or_equal and strictly_better:
                dominated = True
                break
        pareto_mask.append(not dominated)

    result_df["pareto_optimal"] = pareto_mask
    out_path = Path(output_dir) / "risk_reward_pareto.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    if mlflow.active_run() is None:
        mlflow.set_experiment("retail_rl_inventory")
        with mlflow.start_run(run_name="reward_sensitivity_pareto"):
            for idx, row in result_df.iterrows():
                mlflow.log_metric("pareto_total_cost", float(row["total_cost"]), step=int(idx))
                mlflow.log_metric(
                    "pareto_service_level", float(row["service_level"]), step=int(idx)
                )
                mlflow.log_metric(
                    "pareto_stockout_severity", float(row["stockout_severity"]), step=int(idx)
                )
            mlflow.log_artifact(str(out_path))

    return out_path
