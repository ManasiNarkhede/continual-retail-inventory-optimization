"""Unit tests for RL SHAP explainer wrapper."""

from __future__ import annotations

import numpy as np

from src.eval.shap_explainer import RLPolicySHAPExplainer


class _DummyModel:
    def predict(self, state, deterministic=True):  # noqa: ANN001, ANN201
        del deterministic
        return np.array([float(np.mean(state[:2]))], dtype=float), None


def test_predict_fn_shape() -> None:
    """Predict wrapper should return one action per row."""
    model = _DummyModel()
    bg = np.zeros((5, 14), dtype=float)
    explainer = RLPolicySHAPExplainer(model=model, background_states=bg)
    states = np.ones((3, 14), dtype=float)
    out = explainer._predict_fn(states)
    assert out.shape == (3,)
