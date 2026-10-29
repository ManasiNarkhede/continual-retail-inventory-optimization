"""Avalanche experience builder for continual learning.

Splits the Rossmann dataset into monthly temporal experiences.
Each experience equals one month of data across all stores.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, random_split

BenchmarkFactory = Callable[[list[Dataset], list[Dataset]], Any]


def _load_benchmark_factory() -> BenchmarkFactory:
    """Return a ``(train, test) -> benchmark`` builder for the installed Avalanche version.

    The returned benchmark always exposes ``train_stream`` and ``test_stream``.

    Raises:
        ImportError: If required Avalanche symbols are unavailable.
    """
    generators_spec = importlib.util.find_spec("avalanche.benchmarks.generators")
    scenario_spec = importlib.util.find_spec("avalanche.benchmarks.scenarios.dataset_scenario")
    utils_spec = importlib.util.find_spec("avalanche.benchmarks.utils")

    if utils_spec is None:
        raise ImportError("avalanche-lib benchmark utilities are unavailable in this environment.")

    if generators_spec is not None:
        # avalanche-lib < 0.5 accepts plain tensor datasets and subsets.
        from avalanche.benchmarks.generators import dataset_benchmark

        def legacy_factory(train: list[Dataset], test: list[Dataset]) -> Any:
            return dataset_benchmark(train_datasets=train, test_datasets=test)

        return legacy_factory

    if scenario_spec is not None:
        # avalanche-lib >= 0.5 requires AvalancheDatasets; stream names come from the keywords.
        from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
        from avalanche.benchmarks.utils import AvalancheDataset

        def factory(train: list[Dataset], test: list[Dataset]) -> Any:
            return benchmark_from_datasets(
                train=[AvalancheDataset([ds]) for ds in train],
                test=[AvalancheDataset([ds]) for ds in test],
            )

        return factory

    raise ImportError(
        "Unsupported avalanche-lib layout for benchmark builders. "
        "Expected one of: avalanche.benchmarks.generators.dataset_benchmark or "
        "avalanche.benchmarks.scenarios.dataset_scenario.benchmark_from_datasets."
    )


class AvalancheExperienceBuilder:
    """Build Avalanche benchmark from monthly Rossmann slices.

    Args:
        n_experiences: Number of monthly experiences to create.
        output_dir: Directory where monthly parquet slices are saved.
    """

    def __init__(self, n_experiences: int = 6, output_dir: str = "data/experiences") -> None:
        """Initialize benchmark builder.

        Args:
            n_experiences: Number of monthly experiences to include.
            output_dir: Directory where experience parquet files are written.
        """
        self.n_experiences = n_experiences
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "sales",
    ) -> Any:
        """Split dataframe by month and return an Avalanche benchmark.

        Args:
            df: Feature-engineered dataframe with Date column.
            feature_cols: List of input feature columns.
            target_col: Target column name.

        Returns:
            Avalanche benchmark with train and test streams.

        Raises:
            ImportError: If avalanche-lib benchmark utilities are unavailable.
        """
        build_benchmark = _load_benchmark_factory()

        data = df.copy().sort_values("Date")
        if target_col not in data.columns:
            alt_target = "Sales" if target_col == "sales" else "sales"
            if alt_target in data.columns:
                target_col = alt_target
            else:
                raise ValueError(f"Target column '{target_col}' not found in dataframe.")
        months = sorted(data["Date"].dt.to_period("M").unique())
        selected_months = months[: self.n_experiences]

        train_datasets = []
        test_datasets = []

        for index, month in enumerate(selected_months):
            month_mask = data["Date"].dt.to_period("M") == month
            month_df = data.loc[month_mask].dropna(subset=feature_cols + [target_col])
            if month_df.empty:
                continue

            features = torch.tensor(month_df[feature_cols].values, dtype=torch.float32)
            targets = torch.tensor(month_df[target_col].values, dtype=torch.float32)

            base_dataset = TensorDataset(features, targets)
            n_val = max(1, int(0.1 * len(base_dataset)))
            n_train = len(base_dataset) - n_val
            if n_train <= 0:
                n_train = len(base_dataset)
                n_val = 0

            if n_val > 0:
                train_split, val_split = random_split(base_dataset, [n_train, n_val])
            else:
                train_split = base_dataset
                val_split = base_dataset

            train_datasets.append(train_split)
            test_datasets.append(val_split)

            parquet_path = self.output_dir / f"experience_{index:02d}_{month}.parquet"
            month_df.to_parquet(parquet_path, index=False)

        return build_benchmark(train_datasets, test_datasets)
