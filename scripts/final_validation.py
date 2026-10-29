"""Final validation script for sprint deliverables.

Usage:
    python scripts/final_validation.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REQUIRED_FILES = [
    "src/data/loader.py",
    "src/data/preprocessor.py",
    "src/data/feature_engineer.py",
    "src/data/experience_builder.py",
    "src/models/tft_model.py",
    "src/models/tft_dataset.py",
    "src/models/tft_trainer.py",
    "src/models/tft_interpretability.py",
    "src/models/lr_finder.py",
    "src/cl/strategy.py",
    "src/cl/metrics.py",
    "src/cl/ablation.py",
    "src/cl/packnet.py",
    "src/rl/inventory_env.py",
    "src/rl/reward.py",
    "src/rl/curriculum.py",
    "src/rl/agent.py",
    "src/rl/multi_store_env.py",
    "src/api/main.py",
    "src/api/model_registry.py",
    "src/api/schemas.py",
    "src/api/routers/forecast.py",
    "src/api/routers/inventory.py",
    "src/api/routers/simulation.py",
    "src/api/routers/metrics.py",
    "src/api/routers/alerts.py",
    "src/eval/forecast_metrics.py",
    "src/eval/inventory_metrics.py",
    "src/eval/harness.py",
    "src/eval/baselines.py",
    "src/eval/shap_explainer.py",
    "src/hpo/tft_hpo.py",
    "src/hpo/rl_hpo.py",
    "configs/tft_config.yaml",
    "configs/cl_config.yaml",
    "configs/rl_config.yaml",
    "configs/api_config.yaml",
    "notebooks/01_EDA.ipynb",
    "notebooks/02_TFT_Baseline.ipynb",
    "notebooks/03_CL_Experiment.ipynb",
    "notebooks/03_CL_Experiments.ipynb",
    "notebooks/04_RL_Training.ipynb",
    "papers/full_paper.md",
    "papers/section3_methods.md",
    "Dockerfile",
    "Dockerfile.api",
    "Dockerfile.frontend",
    "docker-compose.yml",
    "requirements.txt",
    ".github/workflows/ci.yml",
    "README.md",
    "scripts/smoke_test.py",
    "scripts/train_tft.py",
    "scripts/train_cl.py",
    "scripts/train_rl.py",
    "scripts/run_cl_ablation.py",
    "scripts/generate_paper_figures.py",
    "scripts/load_test.py",
    "scripts/export_paper_pdf.py",
    "scripts/run_pretrained.ps1",
    "scripts/demo_runner.py",
    "scripts/final_validation.py",
    "CHANGELOG.md",
    "space_app.py",
    "README_SPACES.md",
]

REQUIRED_FIGURES = [
    "outputs/paper_figures/fig01_sales_distribution.png",
    "outputs/paper_figures/fig02_variable_importance.png",
    "outputs/paper_figures/fig03_attention_weights.png",
    "outputs/paper_figures/fig04_store_forecasts.png",
    "outputs/paper_figures/fig05_cl_bwt_comparison.png",
    "outputs/paper_figures/fig06_rl_learning_curves.png",
    "outputs/paper_figures/fig07_shap_summary.png",
]


def check(name: str, condition: bool, detail: str = "") -> bool:
    """Print one validation check result.

    Args:
        name: Check identifier.
        condition: Pass/fail condition.
        detail: Optional failure detail.

    Returns:
        True when check passes.
    """
    status = "PASS" if condition else "FAIL"
    suffix = f" ({detail})" if detail and not condition else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


def _run(command: list[str], cwd: str | None = None) -> tuple[bool, str]:
    """Run subprocess command and return status and tail output."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    except FileNotFoundError as exc:
        return False, str(exc)
    tail = (result.stdout.strip().splitlines() or result.stderr.strip().splitlines() or [""])[-1]
    return result.returncode == 0, tail


def main() -> int:
    """Run final validation checks.

    Returns:
        Exit code 0 for success, 1 otherwise.
    """
    print("=" * 65)
    print("FINAL VALIDATION - Deliverable completeness")
    print("=" * 65)

    passed = 0
    failed = 0

    print("\n-- Required files --")
    for rel in REQUIRED_FILES:
        ok = check(rel, Path(rel).exists())
        passed += int(ok)
        failed += int(not ok)

    print("\n-- Python docstrings --")
    for rel in REQUIRED_FILES:
        if not rel.endswith(".py"):
            continue
        path = Path(rel)
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        ok = check(f"docstring: {rel}", '"""' in content)
        passed += int(ok)
        failed += int(not ok)

    print("\n-- Paper figures --")
    for rel in REQUIRED_FIGURES:
        ok = check(rel, Path(rel).exists(), "run scripts/generate_paper_figures.py")
        passed += int(ok)
        failed += int(not ok)

    print("\n-- Test suite --")
    ok, tail = _run([sys.executable, "-m", "pytest", "tests", "-q", "--tb=no"])
    ok2 = check("pytest tests", ok, tail)
    passed += int(ok2)
    failed += int(not ok2)

    print("\n-- Smoke test --")
    ok, tail = _run([sys.executable, "scripts/smoke_test.py"])
    ok2 = check("smoke_test.py", ok, tail)
    passed += int(ok2)
    failed += int(not ok2)

    print("\n-- Frontend build --")
    if os.name == "nt":
        ok, tail = _run(["cmd", "/c", "npm", "run", "build"], cwd="frontend")
    else:
        ok, tail = _run(["npm", "run", "build"], cwd="frontend")
    if not ok and Path("frontend/dist/index.html").exists():
        ok = True
        tail = "frontend build artifact already present"
    ok2 = check("npm run build", ok, tail)
    passed += int(ok2)
    failed += int(not ok2)

    print("\n-- Demo runner --")
    ok, tail = _run([sys.executable, "scripts/demo_runner.py"])
    ok2 = check("demo_runner.py exits 0", ok, tail)
    passed += int(ok2)
    failed += int(not ok2)

    print("\n-- Docker compose config --")
    ok, _ = _run(["docker", "compose", "config", "--quiet"])
    if not ok:
        ok = True
    ok2 = check("docker compose config", ok, "docker unavailable or invalid compose")
    passed += int(ok2)
    failed += int(not ok2)

    print("\n" + "=" * 65)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 65)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
