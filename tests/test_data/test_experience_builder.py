"""Unit tests for AvalancheExperienceBuilder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="Avalanche imports are unstable on current Windows runtime")

from src.data.experience_builder import AvalancheExperienceBuilder


def test_experience_builder_creates_benchmark_and_parquets(tmp_path: Path) -> None:
    """Ensure benchmark and experience parquet outputs are generated."""
    rows = []
    for month in [1, 2]:
        for day in range(1, 6):
            rows.append(
                {
                    "Date": pd.Timestamp(year=2015, month=month, day=day),
                    "Store": 1,
                    "sales": float(100 + day + month),
                    "feat_a": float(day),
                    "feat_b": float(month),
                }
            )
    df = pd.DataFrame(rows)

    output_dir = tmp_path / "experiences"
    builder = AvalancheExperienceBuilder(n_experiences=2, output_dir=str(output_dir))
    benchmark = builder.build(df=df, feature_cols=["feat_a", "feat_b"], target_col="sales")

    assert len(benchmark.train_stream) > 0
    parquet_files = list(output_dir.glob("*.parquet"))
    assert parquet_files
