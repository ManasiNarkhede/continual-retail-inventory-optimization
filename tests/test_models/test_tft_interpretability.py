"""Tests for TFT interpretability artifact persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.tft_interpretability import TFTInterpreter


class _Model:
    def eval(self):
        return self


class _Dataset:
    time_varying_known_reals = []
    time_varying_unknown_reals = []
    static_categoricals = []
    static_reals = []


def test_save_artefacts_writes_csv_and_png(tmp_path: Path, monkeypatch) -> None:
    """save_artefacts should persist variable importance and attention map files."""
    interpreter = TFTInterpreter(model=_Model(), dataset=_Dataset())

    importance = pd.DataFrame(
        [
            {"variable": "promo", "importance": 0.7, "type": "encoder"},
            {"variable": "store_type", "importance": 0.3, "type": "static"},
        ]
    )
    attention = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=float)

    monkeypatch.setattr(interpreter, "variable_importance", lambda dataloader: importance)
    monkeypatch.setattr(interpreter, "attention_weights", lambda dataloader: attention)

    csv_path, png_path = interpreter.save_artefacts(dataloader=object(), output_dir=str(tmp_path))

    assert csv_path.exists()
    assert png_path.exists()
    loaded = pd.read_csv(csv_path)
    assert list(loaded.columns) == ["variable", "importance", "type"]
    assert len(loaded) == 2
    assert png_path.suffix == ".png"
    assert (tmp_path / "tft_variable_importance.json").exists()
    assert (tmp_path / "tft_attention_weights.npy").exists()
    assert (tmp_path / "tft_interpretability_metadata.json").exists()
