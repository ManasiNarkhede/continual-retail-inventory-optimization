"""Per-store cost analysis for inventory performance diagnostics.

Usage:
    python scripts/cost_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_cost_analysis(output_dir: str = "outputs/evaluation") -> Path:
    """Compute and plot per-store average demand-cost proxy.

    Args:
        output_dir: Destination directory for CSV and plot outputs.

    Returns:
        Path to generated CSV summary.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = Path("data/raw/train.csv")
    if train_path.exists():
        df = pd.read_csv(train_path, usecols=["Store", "Sales", "Open"])  # type: ignore[arg-type]
        df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
    else:
        rng = np.random.default_rng(42)
        rows = []
        for store in range(1, 51):
            for _ in range(120):
                rows.append(
                    {"Store": store, "Sales": float(max(100.0, rng.normal(6000, 1200))), "Open": 1}
                )
        df = pd.DataFrame(rows)

    summary = (
        df.groupby("Store")["Sales"]
        .agg(mean_sales="mean", std_sales="std", n_days="count")
        .reset_index()
        .assign(
            stockout_proxy_cost=lambda s: s["std_sales"].fillna(0.0) * 0.8,
            holding_proxy_cost=lambda s: s["mean_sales"] * 0.05,
        )
    )
    summary["total_proxy_cost"] = summary["stockout_proxy_cost"] + summary["holding_proxy_cost"]
    summary = summary.sort_values("total_proxy_cost", ascending=False)

    csv_path = out / "store_cost_breakdown.csv"
    summary.to_csv(csv_path, index=False)

    top = summary.head(20).sort_values("total_proxy_cost")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Store"].astype(str), top["holding_proxy_cost"], label="Holding proxy")
    ax.barh(
        top["Store"].astype(str),
        top["stockout_proxy_cost"],
        left=top["holding_proxy_cost"],
        label="Stockout proxy",
    )
    ax.set_title("Per-store cost breakdown (top 20 by proxy cost)")
    ax.set_xlabel("Proxy cost")
    ax.set_ylabel("Store")
    ax.legend()
    fig.tight_layout()
    fig_path = out / "store_cost_breakdown.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")
    return csv_path


if __name__ == "__main__":
    run_cost_analysis()
