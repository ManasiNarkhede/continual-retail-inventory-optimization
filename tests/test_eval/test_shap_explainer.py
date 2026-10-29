"""Unit tests for the RL SHAP explainer using an injected fake ``shap`` module."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.eval.shap_explainer import STATE_FEATURE_NAMES, RLPolicySHAPExplainer


class _Model:
    """Policy whose action is the sum of the first two state features."""

    def __init__(self) -> None:
        self.dtypes: list = []

    def predict(self, state, deterministic=True):  # noqa: ANN001, ANN201
        assert deterministic is True
        self.dtypes.append(state.dtype)
        return np.array([state[0] + state[1]]), None


def _fake_shap() -> types.ModuleType:
    module = types.ModuleType("shap")
    module.calls = {}

    def sample(data, n):  # noqa: ANN001, ANN202
        module.calls["sample"] = (data, n)
        return data[:n]

    class KernelExplainer:
        def __init__(self, fn, background) -> None:  # noqa: ANN001
            module.calls["kernel"] = (fn, background)
            self.fn = fn

        def shap_values(self, states, nsamples):  # noqa: ANN001, ANN201
            module.calls["shap_values"] = (states, nsamples)
            return [self.fn(states)]

    def summary_plot(values, states, feature_names, show):  # noqa: ANN001, ANN202
        module.calls["summary_plot"] = (values, states, feature_names, show)
        plt.bar(range(len(feature_names)), np.abs(np.asarray(values)).mean(axis=0))

    module.sample = sample
    module.KernelExplainer = KernelExplainer
    module.summary_plot = summary_plot
    return module


def test_default_feature_names_and_float_background() -> None:
    """Defaults fall back to the 14 state feature names and cast background to float."""
    explainer = RLPolicySHAPExplainer(_Model(), background_states=[[1, 2], [3, 4]])
    assert explainer.feature_names == STATE_FEATURE_NAMES
    assert len(explainer.feature_names) == 14
    assert explainer.background_states.dtype == float
    custom = RLPolicySHAPExplainer(_Model(), np.zeros((1, 2)), feature_names=["a", "b"])
    assert custom.feature_names == ["a", "b"]


def test_predict_fn_values_and_float32_inputs() -> None:
    """Predictions are computed per row with float32 states."""
    model = _Model()
    explainer = RLPolicySHAPExplainer(model, np.zeros((2, 3)))
    out = explainer._predict_fn(np.array([[1.0, 2.0, 9.0], [0.5, -1.0, 9.0]]))
    np.testing.assert_allclose(out, [3.0, -0.5])
    assert model.dtypes == [np.float32, np.float32]


@pytest.mark.parametrize(
    ("background_rows", "n_samples", "expected_background"),
    [(6, 4, 4), (3, 100, 3)],
    ids=["sampled", "capped-at-background-size"],
)
def test_explain_uses_kernel_explainer(
    monkeypatch, background_rows: int, n_samples: int, expected_background: int
) -> None:
    """explain() samples min(n_samples, rows) background rows and wraps _predict_fn."""
    fake = _fake_shap()
    monkeypatch.setitem(sys.modules, "shap", fake)
    background = np.arange(background_rows * 2, dtype=float).reshape(background_rows, 2)
    explainer = RLPolicySHAPExplainer(_Model(), background, feature_names=["a", "b"])

    values = explainer.explain([[1.0, 1.0], [2.0, 3.0]], n_samples=n_samples)

    data, n = fake.calls["sample"]
    assert n == expected_background
    np.testing.assert_array_equal(data, background)
    fn, bg = fake.calls["kernel"]
    assert fn == explainer._predict_fn
    assert bg.shape == (expected_background, 2)
    passed_states, nsamples = fake.calls["shap_values"]
    assert nsamples == n_samples and passed_states.dtype == float
    assert isinstance(values, np.ndarray)
    np.testing.assert_allclose(values, [[2.0, 5.0]])


def test_explain_without_shap_raises_import_error(monkeypatch) -> None:
    """A missing shap package surfaces as an ImportError with install hint."""
    monkeypatch.setitem(sys.modules, "shap", None)
    explainer = RLPolicySHAPExplainer(_Model(), np.zeros((2, 2)))
    with pytest.raises(ImportError, match="Install shap"):
        explainer.explain(np.zeros((1, 2)))


def test_plot_summary_writes_figure(tmp_path: Path, monkeypatch) -> None:
    """plot_summary() renders through shap.summary_plot and saves the PNG."""
    fake = _fake_shap()
    monkeypatch.setitem(sys.modules, "shap", fake)
    monkeypatch.chdir(tmp_path)
    explainer = RLPolicySHAPExplainer(_Model(), np.zeros((2, 2)), feature_names=["a", "b"])
    values = np.array([[0.1, -0.2], [0.3, 0.4]])

    explainer.plot_summary(values, [[1, 2], [3, 4]])

    out = tmp_path / "outputs" / "paper_figures" / "fig07_shap_summary.png"
    assert out.exists() and out.stat().st_size > 0
    passed_values, passed_states, names, show = fake.calls["summary_plot"]
    assert passed_values is values
    assert passed_states.dtype == float
    assert names == ["a", "b"]
    assert show is False


def test_plot_summary_skips_without_shap(tmp_path: Path, monkeypatch) -> None:
    """Without shap the plot is skipped and nothing is written."""
    monkeypatch.setitem(sys.modules, "shap", None)
    monkeypatch.chdir(tmp_path)
    explainer = RLPolicySHAPExplainer(_Model(), np.zeros((2, 2)))
    assert explainer.plot_summary(np.zeros((1, 2)), np.zeros((1, 2))) is None
    assert not (tmp_path / "outputs").exists()
