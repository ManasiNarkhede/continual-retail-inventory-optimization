"""Unit tests for RossmannDataLoader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import RossmannDataLoader


def _write_csv(path: Path, content: str) -> None:
    """Write CSV content to disk.

    Args:
        path: Destination file path.
        content: CSV text content.
    """
    path.write_text(content, encoding="utf-8")


def test_loader_reads_and_merges(tmp_path: Path) -> None:
    """Validate loader merges train and test with store metadata.

    Args:
        tmp_path: Temporary directory fixture.
    """
    _write_csv(
        tmp_path / "train.csv",
        "Store,DayOfWeek,Date,Sales,Customers,Open,Promo,StateHoliday,SchoolHoliday\n"
        "1,1,2015-01-01,100,20,1,1,0,0\n"
        "1,7,2015-01-04,0,0,0,0,0,0\n",
    )
    _write_csv(
        tmp_path / "test.csv",
        "Store,DayOfWeek,Date,Open,Promo,StateHoliday,SchoolHoliday\n" "1,2,2015-01-02,1,0,0,0\n",
    )
    _write_csv(
        tmp_path / "store.csv",
        "Store,StoreType,Assortment,CompetitionDistance\n1,a,a,100\n",
    )

    loader = RossmannDataLoader(tmp_path)
    train_df, test_df = loader.load()
    assert len(train_df) == 1  # closed day dropped by default
    assert not test_df.empty
    assert "StoreType" in train_df.columns
    assert pd.api.types.is_datetime64_any_dtype(train_df["Date"])
    assert loader.load(drop_closed=False)[0]["Sales"].tolist() == [100, 0]


def test_loader_missing_file_raises(tmp_path: Path) -> None:
    """Ensure missing source files raise FileNotFoundError.

    Args:
        tmp_path: Temporary directory fixture.
    """
    loader = RossmannDataLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_loader_schema_rejects_missing_sales_column(tmp_path: Path) -> None:
    """Loader should fail fast when Sales is missing from train.csv."""
    _write_csv(
        tmp_path / "train.csv",
        "Store,DayOfWeek,Date,Customers,Open,Promo,StateHoliday,SchoolHoliday\n"
        "1,1,2015-01-01,20,1,1,0,0\n",
    )
    _write_csv(
        tmp_path / "test.csv",
        "Store,DayOfWeek,Date,Open,Promo,StateHoliday,SchoolHoliday\n" "1,2,2015-01-02,1,0,0,0\n",
    )
    _write_csv(
        tmp_path / "store.csv",
        "Store,StoreType,Assortment,CompetitionDistance\n1,a,a,100\n",
    )

    loader = RossmannDataLoader(tmp_path)
    with pytest.raises(ValueError):
        loader.load()
