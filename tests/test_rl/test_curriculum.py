"""Unit tests for curriculum scheduler."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rl.curriculum import CurriculumScheduler


def _make_train_df() -> pd.DataFrame:
    stores = np.repeat(np.arange(1, 9), 40)
    dates = pd.date_range("2015-01-01", periods=40).tolist() * 8
    sales = np.random.uniform(100.0, 500.0, size=len(stores))
    return pd.DataFrame({"Store": stores, "Date": pd.to_datetime(dates), "Sales": sales})


def _make_train_df_lowercase_sales() -> pd.DataFrame:
    stores = np.repeat(np.arange(1, 5), 20)
    dates = pd.date_range("2015-01-01", periods=20).tolist() * 4
    sales = np.random.uniform(50.0, 250.0, size=len(stores))
    return pd.DataFrame({"Store": stores, "Date": pd.to_datetime(dates), "sales": sales})


def test_curriculum_has_levels() -> None:
    """Scheduler should create at least one level."""
    scheduler = CurriculumScheduler(_make_train_df(), n_levels=4)
    assert len(scheduler.levels) >= 1


def test_curriculum_record_episode_returns_bool() -> None:
    """record_episode should return boolean progression indicator."""
    scheduler = CurriculumScheduler(_make_train_df(), n_levels=3)
    advanced = scheduler.record_episode(service_level=0.99)
    assert isinstance(advanced, bool)


def test_curriculum_accepts_lowercase_sales_column() -> None:
    """Curriculum scheduler should also work with normalized 'sales' column."""
    scheduler = CurriculumScheduler(_make_train_df_lowercase_sales(), n_levels=2)
    assert len(scheduler.levels) >= 1
