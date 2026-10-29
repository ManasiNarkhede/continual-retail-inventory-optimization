"""Tests for model registry artifact recovery helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api import model_registry


def test_recover_bundle_metadata_creates_sidecar(tmp_path: Path) -> None:
    """Missing .meta.json should be generated from config + checkpoint."""
    checkpoint = tmp_path / "tft_latest.pt"
    checkpoint.write_bytes(b"weights")

    cfg_path = tmp_path / "tft_runtime_config.yaml"
    cfg_path.write_text(
        """
tft:
  quantiles: [0.1, 0.5, 0.9]
  target: sales
  max_encoder_length: 30
  max_prediction_length: 7
  static_categoricals: [StoreType]
  static_reals: [competition_distance]
  time_varying_known_reals: [promo]
  time_varying_unknown_reals: [sales]
""".strip(),
        encoding="utf-8",
    )

    ok = model_registry._recover_bundle_metadata(checkpoint_path=checkpoint, config_path=cfg_path)
    assert ok is True

    meta_path = checkpoint.with_suffix(".meta.json")
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["quantiles"] == [0.1, 0.5, 0.9]
    assert payload["target"] == "sales"


def test_recover_runtime_config_from_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime config should be copied from base config when absent."""
    runtime_cfg = tmp_path / "data" / "processed" / "tft_runtime_config.yaml"
    base_cfg = tmp_path / "configs" / "tft_config.yaml"
    base_cfg.parent.mkdir(parents=True, exist_ok=True)
    base_cfg.write_text("tft:\n  target: sales\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    ok = model_registry._recover_runtime_config(runtime_cfg)
    assert ok is True
    assert runtime_cfg.exists()


def test_cache_invalidation_resets_ttl_tracking() -> None:
    """invalidate_cache should clear internal TTL timestamps."""
    model_registry._CACHE_LOADED_AT.update(
        {"tft": 1.0, "rl": 1.0, "cl_metrics": 1.0, "simulation": 1.0}
    )
    model_registry.invalidate_cache()
    assert all(value is None for value in model_registry._CACHE_LOADED_AT.values())
