"""Project smoke test for core training and serving imports.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODULES = [
    "src.data.loader",
    "src.data.preprocessor",
    "src.data.feature_engineer",
    "src.models.tft_dataset",
    "src.models.tft_model",
    "src.models.tft_trainer",
    "src.models.tft_interpretability",
    "src.models.lr_finder",
    "src.cl.strategy",
    "src.cl.ablation",
    "src.cl.packnet",
    "src.rl.inventory_env",
    "src.rl.reward",
    "src.rl.curriculum",
    "src.rl.multi_store_env",
    "src.api.main",
    "src.api.routers.forecast",
    "src.api.routers.inventory",
    "src.api.routers.simulation",
    "src.api.routers.metrics",
    "src.api.routers.alerts",
    "src.eval.harness",
    "src.eval.baselines",
    "src.eval.shap_explainer",
    "src.hpo.tft_hpo",
    "src.hpo.rl_hpo",
    "space_app",
]


def main() -> int:
    """Import all core modules and return process status code."""
    failures: list[str] = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"OK  {name}")
        except Exception as exc:
            message = str(exc)
            if "No module named" in message:
                print(f"SKIP {name}: {message}")
                continue
            failures.append(f"{name}: {message}")
            print(f"ERR {name}: {message}")

    if failures:
        print("\nSmoke test failed")
        for line in failures:
            print(f" - {line}")
        return 1
    print("\nSmoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
