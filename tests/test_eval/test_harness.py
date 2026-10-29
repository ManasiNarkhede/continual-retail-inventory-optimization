"""Unit tests for evaluation harness outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.harness import EvaluationHarness


def test_harness_runs_baselines_only(tmp_path) -> None:
    """Harness should return baseline rows without model objects."""
    data = pd.DataFrame({"Sales": np.random.uniform(100.0, 500.0, size=60)})
    harness = EvaluationHarness(output_dir=str(tmp_path), log_to_mlflow=False)
    result = harness.run_all(val_df=data, forecaster=None, rl_agent=None)
    assert not result.empty
    assert "model_name" in result.columns
    assert any(result["model_name"].str.contains("Naive"))
    assert any(result["model_name"].str.contains("Base-stock"))
    assert "weighted_smape" in result.columns
