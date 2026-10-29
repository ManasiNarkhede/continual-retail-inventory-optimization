"""Run research-grade results experiments and persist outputs.

This script executes:
1) Distribution-shift CL ablation (promo vs non-promo)
2) RL stockout-severity stress tests
3) TFT interpretability figure generation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.cl.ablation import CLAblationRunner
from src.data.feature_engineer import FeatureEngineer
from src.data.loader import RossmannDataLoader
from src.data.preprocessor import RossmannPreprocessor
from src.eval.harness import EvaluationHarness
from src.models.tft_dataset import build_tft_dataset
from src.models.tft_interpretability import TFTInterpreter
from src.models.tft_model import TFTForecaster


def _prepare_df() -> pd.DataFrame:
    loader = RossmannDataLoader("data/raw")
    train_df, _ = loader.load()
    pre = RossmannPreprocessor()
    train_df = pre.fit_transform(train_df)
    return FeatureEngineer().transform(train_df)


def _promo_shift_benchmark(df: pd.DataFrame):
    from avalanche.benchmarks.generators import dataset_benchmark
    from avalanche.benchmarks.utils import AvalancheDataset

    runtime_cfg = "data/processed/tft_runtime_config.yaml"
    if not Path(runtime_cfg).exists():
        runtime_cfg = "configs/tft_config.yaml"

    base_ds = build_tft_dataset(df, config_path=runtime_cfg)

    promo_col = "Promo" if "Promo" in df.columns else "promo"
    if promo_col not in df.columns:
        raise ValueError("Promo/promo column is required for distribution-shift benchmark.")

    promo_df = df[df[promo_col].astype(float) > 0.0].copy()
    nonpromo_df = df[df[promo_col].astype(float) <= 0.0].copy()

    if promo_df.empty or nonpromo_df.empty:
        raise ValueError("Promo split produced empty partition(s).")

    nonpromo_ds = build_tft_dataset(
        nonpromo_df, config_path=runtime_cfg, training=False, training_dataset=base_ds
    )
    promo_ds = build_tft_dataset(
        promo_df, config_path=runtime_cfg, training=False, training_dataset=base_ds
    )

    train_stream = [AvalancheDataset(nonpromo_ds), AvalancheDataset(promo_ds)]
    test_stream = [AvalancheDataset(nonpromo_ds), AvalancheDataset(promo_ds)]
    return (
        dataset_benchmark(train_datasets=train_stream, test_datasets=test_stream),
        base_ds,
        runtime_cfg,
    )


def run_distribution_shift(df: pd.DataFrame, cl_config: str) -> dict:
    benchmark, train_ds, runtime_cfg = _promo_shift_benchmark(df)
    runner = CLAblationRunner(config_path=cl_config)

    payload = runner.run_distribution_shift_comparison(
        model_factory=lambda: TFTForecaster(config_path=runtime_cfg, dataset=train_ds).build(),
        benchmark=benchmark,
    )
    runner.save_distribution_shift_metrics(
        payload, output_path="data/processed/cl_metrics_latest.json"
    )
    return payload


def run_stress_test(df: pd.DataFrame) -> Path:
    harness = EvaluationHarness(output_dir="outputs/evaluation", log_to_mlflow=True)
    result_df = harness.stress_test_stockout_severity(
        val_df=df,
        forecaster=True,
        rl_agent=True,
        severities=[round(v, 3) for v in np.linspace(0.01, 0.1, 10)],
        output_filename="rl_stress_test.csv",
    )
    out = Path("outputs/evaluation/rl_stress_test.csv")
    result_df.to_csv(out, index=False)
    return out


def run_reward_sensitivity(df: pd.DataFrame) -> Path:
    """Run risk-vs-reward Pareto sweep and persist CSV."""
    from src.eval.reward_sensitivity import run_reward_sensitivity_sweep

    return run_reward_sensitivity_sweep(val_df=df, output_dir="outputs/evaluation")


def run_interpretability(store_id: int) -> tuple[Path, Path]:
    runtime_cfg = Path("data/processed/tft_runtime_config.yaml")
    if not runtime_cfg.exists():
        runtime_cfg = Path("configs/tft_config.yaml")

    ref_df = pd.read_parquet("data/processed/tft_reference.parquet")
    train_ds = build_tft_dataset(ref_df, config_path=str(runtime_cfg))
    forecaster = TFTForecaster.load_any(
        path_or_base="checkpoints/tft/tft_latest.pt",
        config_path=str(runtime_cfg),
        dataset=train_ds,
    )
    interpreter = TFTInterpreter(forecaster.model, train_ds)
    var_path, attn_path = interpreter.generate_store_figures(
        store_id=store_id,
        reference_df=ref_df,
        output_dir="reports/figures",
    )
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(var_path))
        mlflow.log_artifact(str(attn_path))
    return var_path, attn_path


def main(cl_config: str, store_id: int) -> None:
    mlflow.set_experiment("retail_results_validation")
    df = _prepare_df()

    with mlflow.start_run(run_name="results_section_experiments"):
        shift = run_distribution_shift(df, cl_config=cl_config)
        stress_path = run_stress_test(df)
        pareto_path = run_reward_sensitivity(df)
        var_path, attn_path = run_interpretability(store_id=store_id)

        print("Distribution shift metrics written to data/processed/cl_metrics_latest.json")
        print(json.dumps(shift, indent=2))
        print(f"Stress test CSV: {stress_path}")
        print(f"Risk-Reward Pareto CSV: {pareto_path}")
        print(f"Variable importance figure: {var_path}")
        print(f"Attention map figure: {attn_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cl-config", default="configs/cl_config.yaml")
    parser.add_argument("--store-id", type=int, default=1)
    args = parser.parse_args()
    main(cl_config=args.cl_config, store_id=args.store_id)
