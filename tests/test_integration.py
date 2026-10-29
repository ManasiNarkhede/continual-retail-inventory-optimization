"""End-to-end integration test: data -> TFT -> RL decision."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_full_pipeline_integration() -> None:
    """Exercise core pipeline components with real artifacts when available."""
    if not all(Path("data/raw", name).exists() for name in ("train.csv", "test.csv", "store.csv")):
        pytest.skip("Rossmann raw CSVs not present in data/raw")
    _ = pytest.importorskip("pandas")
    _ = pytest.importorskip("torch")
    _ = pytest.importorskip("stable_baselines3")

    from src.api.model_registry import get_tft_model, load_rl_agent
    from src.data.feature_engineer import FeatureEngineer
    from src.data.loader import RossmannDataLoader
    from src.data.preprocessor import RossmannPreprocessor
    from src.rl.inventory_env import RetailInventoryEnv

    loader = RossmannDataLoader("data/raw")
    train_df, _test_df = loader.load()
    train_df = RossmannPreprocessor().fit_transform(train_df)
    train_df = FeatureEngineer().transform(train_df)
    assert not train_df.empty

    model, dataset, reference_df = get_tft_model()
    if model is None or dataset is None or reference_df is None:
        pytest.skip("TFT artifacts not available for integration test")

    stores = sorted(reference_df["Store"].dropna().astype(int).unique().tolist())
    assert stores
    store_id = int(stores[0])
    pred, _dates = model.predict_for_store(
        store_id=store_id, horizon=7, training_dataset=dataset, reference_df=reference_df
    )
    q_idx = model.quantile_indices
    p50 = pred[:, q_idx.get("p50", 3)].detach().cpu().numpy().tolist()
    assert len(p50) == 7

    demand = (
        train_df[train_df["Store"] == store_id].sort_values("Date")["sales"].to_numpy(dtype=float)
    )
    # Fallback to synthetic demand when the reference store is not present in raw CSVs
    # (this can happen when reference parquet was built from a different data subset)
    if demand.size == 0:
        demand = np.ones(365, dtype=float) * 50.0
    env = RetailInventoryEnv(config_path="configs/rl_config.yaml", demand_series=demand)
    load_rl_agent(config_path="configs/rl_config.yaml", demand_series=demand)

    obs, _ = env.reset()
    assert obs.shape[0] == env.horizon + env._AUX_FEATURES
