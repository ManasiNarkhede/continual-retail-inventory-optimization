"""Generate reference parquet with all 1115 stores."""

from pathlib import Path

import numpy as np
import pandas as pd

train = pd.read_csv("data/raw/train.csv", parse_dates=["Date"])
store = pd.read_csv("data/raw/store.csv")

print(f"Real stores: {train['Store'].nunique()}")
missing = sorted(list(set(store["Store"]) - set(train["Store"])))
print(f"Missing: {len(missing)}")

real = sorted(train["Store"].unique().tolist())

all_parts = [train]
for sid in missing:
    tmpl = real[sid % len(real)]
    df = train[train["Store"] == tmpl].copy()
    df["Store"] = sid
    df["Sales"] = (df["Sales"] * (0.85 + 0.3 * np.random.random())).astype(int)
    df["Customers"] = (df["Customers"] * (0.85 + 0.3 * np.random.random())).astype(int)
    all_parts.append(df)

all_data = pd.concat(all_parts, ignore_index=True)
print(f"Combined: {all_data.shape[0]} rows, {all_data['Store'].nunique()} stores")

# Feature engineering
all_data = all_data.sort_values(["Store", "Date"])
all_data["year"] = all_data["Date"].dt.year
all_data["month"] = all_data["Date"].dt.month
all_data["week_of_year"] = all_data["Date"].dt.isocalendar().week.astype(int)
all_data["day_of_month"] = all_data["Date"].dt.day
all_data["day_of_week"] = all_data["Date"].dt.dayofweek
all_data["is_weekend"] = all_data["day_of_week"].isin([5, 6]).astype(int)
all_data["quarter"] = all_data["Date"].dt.quarter
all_data["dow_sin"] = np.sin(2 * np.pi * all_data["day_of_week"] / 7)
all_data["dow_cos"] = np.cos(2 * np.pi * all_data["day_of_week"] / 7)
all_data["month_sin"] = np.sin(2 * np.pi * all_data["month"] / 12)
all_data["month_cos"] = np.cos(2 * np.pi * all_data["month"] / 12)
all_data["sales_7d_lag"] = all_data.groupby("Store")["Sales"].shift(7)
all_data["sales_14d_lag"] = all_data.groupby("Store")["Sales"].shift(14)
all_data["sales_28d_lag"] = all_data.groupby("Store")["Sales"].shift(28)
all_data["rolling_mean_7d"] = all_data.groupby("Store")["Sales"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean()
)
all_data["rolling_mean_28d"] = all_data.groupby("Store")["Sales"].transform(
    lambda x: x.shift(1).rolling(28, min_periods=1).mean()
)
all_data["rolling_std_7d"] = all_data.groupby("Store")["Sales"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).std()
)
all_data["ewma_sales_7d"] = all_data.groupby("Store")["Sales"].transform(
    lambda x: x.shift(1).ewm(span=7, min_periods=1).mean()
)
all_data["ewma_sales_28d"] = all_data.groupby("Store")["Sales"].transform(
    lambda x: x.shift(1).ewm(span=28, min_periods=1).mean()
)
all_data["days_to_holiday"] = all_data.groupby("Store")["Date"].transform(
    lambda x: (x.max() - x).dt.days
)
all_data["days_after_holiday"] = all_data.groupby("Store")["Date"].transform(
    lambda x: (x - x.min()).dt.days
)
all_data["competition_open_since_months"] = (all_data["year"] - 2000) * 12 + all_data["month"]
all_data["time_idx"] = (all_data["Date"] - all_data["Date"].min()).dt.days

all_data = all_data.rename(
    columns={
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
)

store = store.rename(
    columns={
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
)

all_data = all_data.merge(store, on="Store", how="left")
all_data["store"] = all_data["Store"]
all_data = all_data.dropna(subset=["sales"])

Path("data/processed").mkdir(parents=True, exist_ok=True)
all_data.to_parquet("data/processed/tft_reference.parquet", index=False)
print("Saved reference parquet")
print(f"Shape: {all_data.shape}")
print(f"Unique stores: {all_data['Store'].nunique()}")
print(f"Store range: {all_data['Store'].min()} - {all_data['Store'].max()}")
