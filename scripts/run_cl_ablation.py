"""Run CL strategy ablation over Rossmann experiences.

Usage:
    python scripts/run_cl_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from src.cl.ablation import CLAblationRunner
from src.data.feature_engineer import FeatureEngineer
from src.data.loader import RossmannDataLoader
from src.data.preprocessor import RossmannPreprocessor
from src.models.tft_dataset import build_tft_dataset
from src.models.tft_model import TFTForecaster


def _build_tft_experiences(
    df,
    config_path: str,
    n_experiences: int = 6,
):
    """Build Avalanche benchmark where each experience is a TFT store partition."""
    import numpy as np
    from avalanche.benchmarks.generators import dataset_benchmark
    from avalanche.benchmarks.utils import AvalancheDataset

    data = df.copy().sort_values(["Store", "Date"])
    full_train_ds = build_tft_dataset(data, config_path=config_path)

    store_ids = sorted(data["Store"].dropna().astype(int).unique().tolist())
    if not store_ids:
        raise RuntimeError("No stores found to build CL experiences.")

    n_chunks = max(1, min(int(n_experiences), len(store_ids)))
    store_chunks = np.array_split(np.asarray(store_ids, dtype=int), n_chunks)

    train_datasets = []
    test_datasets = []
    for chunk in store_chunks:
        if chunk.size == 0:
            continue

        exp_df = data[data["Store"].isin(chunk.tolist())].copy()
        if exp_df.empty:
            continue

        exp_ds = build_tft_dataset(
            exp_df,
            config_path=config_path,
            training=False,
            training_dataset=full_train_ds,
        )
        avl_ds = AvalancheDataset(exp_ds)
        train_datasets.append(avl_ds)
        test_datasets.append(avl_ds)

    if not train_datasets:
        raise RuntimeError("No CL experiences were built from TFT dataset.")

    return (
        dataset_benchmark(train_datasets=train_datasets, test_datasets=test_datasets),
        full_train_ds,
    )


def _build_promo_shift_benchmark(train_df):
    """Create two-experience benchmark: non-promo then promo."""
    from avalanche.benchmarks.generators import dataset_benchmark
    from avalanche.benchmarks.utils import AvalancheDataset

    runtime_cfg = "data/processed/tft_runtime_config.yaml"
    if not Path(runtime_cfg).exists():
        runtime_cfg = "configs/tft_config.yaml"

    base_ds = build_tft_dataset(train_df, config_path=runtime_cfg)
    promo_col = "Promo" if "Promo" in train_df.columns else "promo"
    nonpromo_df = train_df[train_df[promo_col].astype(float) <= 0.0].copy()
    promo_df = train_df[train_df[promo_col].astype(float) > 0.0].copy()

    nonpromo_ds = build_tft_dataset(
        nonpromo_df, config_path=runtime_cfg, training=False, training_dataset=base_ds
    )
    promo_ds = build_tft_dataset(
        promo_df, config_path=runtime_cfg, training=False, training_dataset=base_ds
    )

    benchmark = dataset_benchmark(
        train_datasets=[AvalancheDataset(nonpromo_ds), AvalancheDataset(promo_ds)],
        test_datasets=[AvalancheDataset(nonpromo_ds), AvalancheDataset(promo_ds)],
    )
    return benchmark, runtime_cfg, base_ds


def main() -> None:
    """Execute CL ablation pipeline and persist outputs."""
    loader = RossmannDataLoader("data/raw")
    train_df, _ = loader.load()
    pre = RossmannPreprocessor()
    train_df = pre.fit_transform(train_df)
    engineer = FeatureEngineer()
    train_df = engineer.transform(train_df)

    runtime_cfg = "data/processed/tft_runtime_config.yaml"
    if not Path(runtime_cfg).exists():
        runtime_cfg = "configs/tft_config.yaml"

    benchmark, training_ds = _build_tft_experiences(
        train_df, config_path=runtime_cfg, n_experiences=6
    )
    shift_benchmark, runtime_cfg, base_ds = _build_promo_shift_benchmark(train_df)

    mlflow.set_experiment("retail_cl_ablation")
    with mlflow.start_run(run_name="cl_ablation_master"):
        runner = CLAblationRunner()
        results = runner.run(
            model_factory=lambda: TFTForecaster(
                config_path=runtime_cfg, dataset=training_ds
            ).build(),
            benchmark=benchmark,
        )
        csv_path = runner.save_results(results)
        print(results.to_string(index=False))
        print(f"Saved: {csv_path}")

        shift_payload = runner.run_distribution_shift_comparison(
            model_factory=lambda: TFTForecaster(config_path=runtime_cfg, dataset=base_ds).build(),
            benchmark=shift_benchmark,
        )
        json_path = runner.save_distribution_shift_metrics(shift_payload)
        print(f"Shift metrics saved: {json_path}")


if __name__ == "__main__":
    main()
