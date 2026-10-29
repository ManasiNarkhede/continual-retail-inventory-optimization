"""Tests ensuring HPO modules avoid import-time side effects."""

from __future__ import annotations

import importlib
import sys


def test_hpo_imports_do_not_create_optuna_dir(tmp_path, monkeypatch) -> None:
    """Importing HPO modules should not create optuna_studies directory."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "optuna_studies"
    assert not target.exists()

    for module in ["src.hpo.tft_hpo", "src.hpo.rl_hpo"]:
        if module in sys.modules:
            del sys.modules[module]
        importlib.import_module(module)

    assert not target.exists()
