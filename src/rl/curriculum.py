"""Curriculum scheduler for RL inventory training.

The scheduler orders stores by demand variability and advances through levels
as performance thresholds are met.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CurriculumLevel:
    """Single curriculum level definition.

    Attributes:
        level_id: 1-based level index.
        store_ids: Store IDs assigned to this level.
        description: Human-readable level summary.
        service_level_threshold: Minimum service level to progress.
        min_episodes: Minimum episodes to run at this level.
    """

    level_id: int
    store_ids: list[int]
    description: str
    service_level_threshold: float = 0.9
    min_episodes: int = 50


class CurriculumScheduler:
    """Manage progression across RL curriculum levels.

    Args:
        train_df: Dataframe containing ``Store`` and ``Sales``.
        n_levels: Number of curriculum levels.
    """

    def __init__(self, train_df: pd.DataFrame, n_levels: int = 4) -> None:
        self.n_levels = int(max(1, n_levels))
        self.levels = self._build_levels(train_df)
        self.current_level_idx = 0
        self.episodes_at_level = 0

    def _build_levels(self, df: pd.DataFrame) -> list[CurriculumLevel]:
        """Create levels by sorting stores by coefficient of variation."""
        sales_col = "Sales" if "Sales" in df.columns else "sales"
        if sales_col not in df.columns:
            raise ValueError("CurriculumScheduler requires a sales column ('Sales' or 'sales').")

        store_stats = (
            df.groupby("Store")[sales_col]
            .agg(mean="mean", std="std")
            .assign(cv=lambda item: item["std"] / item["mean"].clip(lower=1e-8))
            .sort_values("cv")
        )
        stores = store_stats.index.tolist()
        if not stores:
            return [CurriculumLevel(level_id=1, store_ids=[1], description="Fallback level")]

        per_level = max(1, len(stores) // self.n_levels)
        thresholds = [0.88, 0.90, 0.92, 0.95]
        levels: list[CurriculumLevel] = []

        for idx in range(self.n_levels):
            start = idx * per_level
            end = (idx + 1) * per_level if idx < self.n_levels - 1 else len(stores)
            level_stores = stores[start:end]
            if not level_stores:
                continue

            cv_min = float(store_stats.loc[level_stores[0], "cv"])
            cv_max = float(store_stats.loc[level_stores[-1], "cv"])
            levels.append(
                CurriculumLevel(
                    level_id=idx + 1,
                    store_ids=[int(s) for s in level_stores],
                    description=f"Level {idx + 1}: CV range [{cv_min:.2f}, {cv_max:.2f}]",
                    service_level_threshold=thresholds[min(idx, len(thresholds) - 1)],
                )
            )

        return levels or [
            CurriculumLevel(level_id=1, store_ids=[int(stores[0])], description="Fallback level")
        ]

    def current_level(self) -> CurriculumLevel:
        """Return active curriculum level."""
        return self.levels[self.current_level_idx]

    def current_store_ids(self) -> list[int]:
        """Return store IDs for the active level."""
        return self.current_level().store_ids

    def record_episode(self, service_level: float) -> bool:
        """Record an episode and advance level if conditions are met.

        Args:
            service_level: Service level achieved in the episode.

        Returns:
            ``True`` when the scheduler advances to a new level.
        """
        self.episodes_at_level += 1
        level = self.current_level()
        can_advance = self.current_level_idx < len(self.levels) - 1
        if (
            can_advance
            and self.episodes_at_level >= level.min_episodes
            and float(service_level) >= level.service_level_threshold
        ):
            self.current_level_idx += 1
            self.episodes_at_level = 0
            logger.info("Curriculum advanced to %s", self.current_level().description)
            return True
        return False

    def is_complete(self) -> bool:
        """Return whether the final level is reached."""
        return self.current_level_idx >= len(self.levels) - 1
