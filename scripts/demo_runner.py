"""End-to-end demonstration script.

Runs a lightweight demo pipeline and endpoint checks.

Usage:
    python scripts/demo_runner.py
    python scripts/demo_runner.py --full
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _synthetic_df() -> pd.DataFrame:
    """Build compact synthetic dataset for demo fallback."""
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=200)
    rows: list[dict[str, object]] = []
    for store in [1, 2]:
        for date in dates:
            rows.append(
                {
                    "Store": store,
                    "Date": date,
                    "Sales": int(np.random.normal(5000 + store * 200, 800)),
                    "Promo": int(np.random.random() < 0.3),
                    "DayOfWeek": date.dayofweek,
                    "SchoolHoliday": 0,
                    "StateHoliday": 0,
                    "Open": 1,
                    "Customers": 500,
                    "StoreType": "a",
                    "Assortment": "a",
                    "CompetitionDistance": 500,
                    "CompetitionOpenSinceMonth": 1.0,
                    "CompetitionOpenSinceYear": 2014.0,
                    "Promo2": 0,
                    "Promo2SinceWeek": 0.0,
                    "Promo2SinceYear": 0.0,
                    "PromoInterval": "None",
                }
            )
    df = pd.DataFrame(rows)
    from src.data.feature_engineer import FeatureEngineer
    from src.data.preprocessor import RossmannPreprocessor

    pre = RossmannPreprocessor()
    fe = FeatureEngineer()
    return fe.transform(pre.fit_transform(df))


def demo_data_pipeline(full: bool = False) -> pd.DataFrame:
    """Load Rossmann data or fallback to synthetic.

    Args:
        full: If True, attempt full raw-data pipeline. If False, force
            lightweight synthetic pipeline for fast demo runtime.
    """
    print("\n[1/4] Data pipeline")
    started = time.time()
    if not full:
        df = _synthetic_df()
        print(f"  Synthetic rows: {len(df):,} in {time.time() - started:.1f}s")
        return df

    try:
        from src.data.feature_engineer import FeatureEngineer
        from src.data.loader import RossmannDataLoader
        from src.data.preprocessor import RossmannPreprocessor

        loader = RossmannDataLoader("data/raw")
        train_df, _ = loader.load()
        pre = RossmannPreprocessor()
        fe = FeatureEngineer()
        train_df = fe.transform(pre.fit_transform(train_df))
        print(f"  Loaded {len(train_df):,} rows in {time.time() - started:.1f}s")
        return train_df
    except Exception:
        print("  Falling back to synthetic demo data")
        df = _synthetic_df()
        print(f"  Synthetic rows: {len(df):,} in {time.time() - started:.1f}s")
        return df


def demo_rl_evaluation(train_df: pd.DataFrame) -> float:
    """Run minimal RL rollout without full training."""
    print("\n[2/4] RL evaluation")
    started = time.time()
    from src.rl.inventory_env import RetailInventoryEnv
    from src.rl.reward import ShapedReward

    store = int(train_df["Store"].iloc[0])
    sales_col = "Sales" if "Sales" in train_df.columns else "sales"
    demand = (
        train_df[train_df["Store"] == store].sort_values("Date")[sales_col].to_numpy(dtype=float)
    )
    reward_fn = ShapedReward(target_stock=120.0)
    env = RetailInventoryEnv(
        config_path="configs/rl_config.yaml", demand_series=demand, reward_fn=reward_fn
    )

    service: list[float] = []
    for _ in range(3):
        obs, _ = env.reset()
        done = False
        while not done:
            action = np.array([0.0], dtype=float)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        service.append(env.service_level())
    mean_service = float(np.mean(service))
    print(f"  Mean service level: {mean_service:.2%} in {time.time() - started:.1f}s")
    return mean_service


def demo_api() -> bool:
    """Check key API endpoints with TestClient."""
    print("\n[3/4] API checks")
    started = time.time()
    from fastapi.testclient import TestClient

    from src.api.main import app

    client = TestClient(app)
    checks = [
        ("GET /health", client.get("/health")),
        ("GET /api/metrics/kpis", client.get("/api/metrics/kpis")),
        ("GET /api/metrics/cl", client.get("/api/metrics/cl")),
        ("GET /api/alerts/active", client.get("/api/alerts/active")),
    ]
    ok = True
    for name, response in checks:
        good = response.status_code in {200, 503}
        ok = ok and good
        print(f"  {'OK' if good else 'FAIL'} {name}: {response.status_code}")
    print(f"  API checks completed in {time.time() - started:.1f}s")
    return ok


def main(full: bool = False) -> int:
    """Run demo pipeline.

    Args:
        full: Reserved flag for future full-pipeline mode.

    Returns:
        Process exit code.
    """
    print("=" * 60)
    print("DEMO RUNNER - Retail Inventory Optimization")
    print("=" * 60)
    started = time.time()

    data = demo_data_pipeline(full=full)
    demo_rl_evaluation(data)
    api_ok = demo_api()

    print("\n[4/4] Summary")
    print(f"  Total demo time: {time.time() - started:.1f}s")
    print(f"  API status: {'OK' if api_ok else 'FAIL'}")
    print("=" * 60)
    return 0 if api_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full training pipeline")
    args = parser.parse_args()
    Path("outputs").mkdir(parents=True, exist_ok=True)
    sys.exit(main(full=args.full))
