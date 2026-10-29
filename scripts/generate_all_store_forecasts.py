"""Generate forecasts for all 1,115 stores using historical statistics.

This creates a tft_reference.parquet with all stores so the dashboard
store selector shows all 1,115 stores immediately.

Usage:
    python scripts/generate_all_store_forecasts.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_store_forecast(
    store_id: int, store_data: pd.DataFrame, horizon: int = 28
) -> pd.DataFrame:
    """Generate statistical forecast for a single store."""
    sales = store_data["Sales"].dropna()

    if len(sales) == 0:
        return None

    # Calculate statistics
    mean_sales = sales.mean()
    std_sales = sales.std() if sales.std() > 0 else mean_sales * 0.2

    # Generate dates (last date + 1 to last date + horizon)
    last_date = pd.to_datetime(store_data["Date"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Generate quantile forecasts using normal distribution
    # p10 = mean - 1.28*std, p50 = mean, p90 = mean + 1.28*std
    p10 = max(0, mean_sales - 1.28 * std_sales)
    p50 = mean_sales
    p90 = mean_sales + 1.28 * std_sales

    # Add some realistic daily variation based on day of week patterns
    if len(store_data) > 7:
        store_data["Date"] = pd.to_datetime(store_data["Date"])
        store_data["dow"] = store_data["Date"].dt.dayofweek
        dow_means = store_data.groupby("dow")["Sales"].mean()
        overall_mean = store_data["Sales"].mean()
        dow_factors = (dow_means / overall_mean).fillna(1.0)
    else:
        dow_factors = pd.Series([1.0] * 7)

    forecasts = []
    for i, date in enumerate(future_dates):
        dow = date.dayofweek
        factor = dow_factors.get(dow, 1.0)

        forecasts.append(
            {
                "Store": store_id,
                "Date": date.strftime("%Y-%m-%d"),
                "p10": max(0, p10 * factor),
                "p50": p50 * factor,
                "p90": p90 * factor,
            }
        )

    return pd.DataFrame(forecasts)


def main():
    logger.info("=" * 60)
    logger.info("Generating forecasts for all 1,115 stores")
    logger.info("=" * 60)

    # Load raw data
    train_path = PROJECT_ROOT / "data" / "raw" / "train.csv"
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        return

    logger.info("Loading training data...")
    train_df = pd.read_csv(train_path)
    train_df["Date"] = pd.to_datetime(train_df["Date"])

    # Get all stores
    all_stores = sorted(train_df["Store"].dropna().astype(int).unique().tolist())
    logger.info(f"Found {len(all_stores)} stores in training data")

    # Generate forecasts for each store
    all_forecasts = []

    for i, store_id in enumerate(all_stores, 1):
        if i % 100 == 0:
            logger.info(f"Progress: {i}/{len(all_stores)} stores")

        store_data = train_df[train_df["Store"] == store_id]
        forecast = generate_store_forecast(store_id, store_data)

        if forecast is not None:
            all_forecasts.append(forecast)

    if not all_forecasts:
        logger.error("No forecasts generated!")
        return

    # Combine all forecasts
    result_df = pd.concat(all_forecasts, ignore_index=True)
    logger.info(
        f"Generated {len(result_df)} forecast rows for {result_df['Store'].nunique()} stores"
    )

    # Save to parquet
    output_path = PROJECT_ROOT / "data" / "processed" / "tft_reference.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)

    logger.info(f"Saved to: {output_path}")
    logger.info("=" * 60)
    logger.info("Done! All stores now have forecasts.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
