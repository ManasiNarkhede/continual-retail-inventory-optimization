"""Generate reference parquet with all 1115 stores including synthetic data for missing ones."""

from pathlib import Path

import numpy as np
import pandas as pd

# Load data
train = pd.read_csv("data/raw/train.csv", parse_dates=["Date"])
store = pd.read_csv("data/raw/store.csv")

print(f"Train stores: {train['Store'].nunique()}")
print(f"Store.csv stores: {store['Store'].nunique()}")

# Feature engineering for stores WITH training data
print("Processing stores with training data...")
train_fe = train.copy()
train_fe["year"] = train_fe["Date"].dt.year
train_fe["month"] = train_fe["Date"].dt.month
train_fe["week_of_year"] = train_fe["Date"].dt.isocalendar().week.astype(int)
train_fe["day_of_month"] = train_fe["Date"].dt.day
train_fe["day_of_week"] = train_fe["Date"].dt.dayofweek
train_fe["is_weekend"] = train_fe["day_of_week"].isin([5, 6]).astype(int)
train_fe["quarter"] = train_fe["Date"].dt.quarter

# Cyclical encoding
train_fe["dow_sin"] = np.sin(2 * np.pi * train_fe["day_of_week"] / 7)
train_fe["dow_cos"] = np.cos(2 * np.pi * train_fe["day_of_week"] / 7)
train_fe["month_sin"] = np.sin(2 * np.pi * train_fe["month"] / 12)
train_fe["month_cos"] = np.cos(2 * np.pi * train_fe["month"] / 12)

# Lagged features
train_fe = train_fe.sort_values(["Store", "Date"])
train_fe["sales_7d_lag"] = train_fe.groupby("Store")["Sales"].shift(7)
train_fe["sales_14d_lag"] = train_fe.groupby("Store")["Sales"].shift(14)
train_fe["sales_28d_lag"] = train_fe.groupby("Store")["Sales"].shift(28)
train_fe["rolling_mean_7d"] = train_fe.groupby("Store")["Sales"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean()
)
train_fe["rolling_mean_28d"] = train_fe.groupby("Store")["Sales"].transform(
    lambda x: x.shift(1).rolling(28, min_periods=1).mean()
)
train_fe["rolling_std_7d"] = train_fe.groupby("Store")["Sales"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).std()
)
train_fe["ewma_sales_7d"] = train_fe.groupby("Store")["Sales"].transform(
    lambda x: x.shift(1).ewm(span=7, min_periods=1).mean()
)
train_fe["ewma_sales_28d"] = train_fe.groupby("Store")["Sales"].transform(
    lambda x: x.shift(1).ewm(span=28, min_periods=1).mean()
)
train_fe["days_to_holiday"] = train_fe.groupby("Store")["Date"].transform(
    lambda x: (x.max() - x).dt.days
)
train_fe["days_after_holiday"] = train_fe.groupby("Store")["Date"].transform(
    lambda x: (x - x.min()).dt.days
)
train_fe["competition_open_since_months"] = (train_fe["year"] - 2000) * 12 + train_fe["month"]
train_fe["time_idx"] = (train_fe["Date"] - train_fe["Date"].min()).dt.days

# Rename columns to match expected format
renames = {
    "Sales": "sales",
    "Customers": "customers",
    "Open": "open",
    "Promo": "promo",
    "StateHoliday": "state_holiday",
    "SchoolHoliday": "school_holiday",
    "Store": "Store",
    "DayOfWeek": "DayOfWeek",
    "Date": "Date",
}
train_fe = train_fe.rename(columns=renames)

# Merge with store metadata
store_renames = {
    "StoreType": "store_type",
    "Assortment": "assortment",
    "CompetitionDistance": "competition_distance",
    "CompetitionOpenSinceMonth": "competition_open_since_month",
    "CompetitionOpenSinceYear": "competition_open_since_year",
    "Promo2": "promo2",
    "Promo2SinceWeek": "promo2_since_week",
    "Promo2SinceYear": "promo2_since_year",
    "PromoInterval": "promo_interval",
}
store = store.rename(columns=store_renames)

# For stores with training data: merge
train_with_stores = train_fe.merge(store, on="Store", how="left")

# For stores WITHOUT training data: create synthetic rows
missing_stores = list(set(store["Store"]) - set(train["Store"]))
print(f"Creating synthetic data for {len(missing_stores)} missing stores...")

synthetic_rows = []
date_range = pd.date_range(train["Date"].min(), train["Date"].max(), freq="D")

for store_id in missing_stores:
    store_meta = store[store["Store"] == store_id].iloc[0].to_dict()
    for i, date in enumerate(date_range):
        dow = date.dayofweek
        month = date.month
        row = {
            "Store": store_id,
            "DayOfWeek": dow + 1,
            "Date": date,
            "sales": np.random.normal(5000, 1500),  # Synthetic mean sales
            "customers": np.random.normal(500, 200),
            "open": 1 if dow < 6 else 0,
            "promo": np.random.choice([0, 1], p=[0.6, 0.4]),
            "state_holiday": 0,
            "school_holiday": 0,
            "year": date.year,
            "month": month,
            "week_of_year": date.isocalendar().week,
            "day_of_month": date.day,
            "day_of_week": dow,
            "is_weekend": 1 if dow in [5, 6] else 0,
            "quarter": (month - 1) // 3 + 1,
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "sales_7d_lag": np.random.normal(5000, 1500),
            "sales_14d_lag": np.random.normal(5000, 1500),
            "sales_28d_lag": np.random.normal(5000, 1500),
            "rolling_mean_7d": np.random.normal(5000, 1500),
            "rolling_mean_28d": np.random.normal(5000, 1500),
            "rolling_std_7d": 1500,
            "ewma_sales_7d": np.random.normal(5000, 1500),
            "ewma_sales_28d": np.random.normal(5000, 1500),
            "days_to_holiday": (date_range.max() - date).days,
            "days_after_holiday": (date - date_range.min()).days,
            "competition_open_since_months": (date.year - 2000) * 12 + month,
            "time_idx": (date - date_range.min()).days,
            **{k: v for k, v in store_meta.items() if k != "Store"},
        }
        synthetic_rows.append(row)

synthetic_df = pd.DataFrame(synthetic_rows)

# Combine real + synthetic
final_df = pd.concat([train_with_stores, synthetic_df], ignore_index=True)

# Clean up NaN values
final_df = final_df.dropna(subset=["sales"])
final_df = final_df.fillna(
    {
        "customers": 500,
        "competition_distance": train_with_stores["competition_distance"].median(),
        "competition_open_since_month": 1,
        "competition_open_since_year": 2000,
        "promo2": 0,
        "promo2_since_week": 0,
        "promo2_since_year": 0,
        "promo_interval": "Jan,Apr,Jul,Oct",
        "rolling_std_7d": 1500,
    }
)

# Ensure Store column exists (lowercase for compatibility)
if "Store" in final_df.columns and "store" not in final_df.columns:
    final_df["store"] = final_df["Store"]

print(f"Final reference shape: {final_df.shape}")
print(f"Unique stores: {final_df['Store'].nunique()}")
print(f"Store range: {final_df['Store'].min()} - {final_df['Store'].max()}")

# Save
out_path = Path("data/processed/tft_reference.parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)
final_df.to_parquet(out_path, index=False)
print(f"Saved to {out_path}")
