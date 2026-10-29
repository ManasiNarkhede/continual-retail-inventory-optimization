"""CLI script for RL reward sensitivity and Pareto frontier logging."""

from __future__ import annotations

from src.data.feature_engineer import FeatureEngineer
from src.data.loader import RossmannDataLoader
from src.data.preprocessor import RossmannPreprocessor
from src.eval.reward_sensitivity import run_reward_sensitivity_sweep


def main() -> None:
    loader = RossmannDataLoader("data/raw")
    train_df, _ = loader.load()
    train_df = RossmannPreprocessor().fit_transform(train_df)
    train_df = FeatureEngineer().transform(train_df)

    out = run_reward_sensitivity_sweep(train_df)
    print(f"Saved reward sensitivity sweep: {out}")


if __name__ == "__main__":
    main()
