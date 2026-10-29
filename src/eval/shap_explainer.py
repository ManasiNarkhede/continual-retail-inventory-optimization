"""SHAP explainability wrapper for RL policy decisions."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

STATE_FEATURE_NAMES = [
    "stock_norm",
    "forecast_d1",
    "forecast_d2",
    "forecast_d3",
    "forecast_d4",
    "forecast_d5",
    "forecast_d6",
    "forecast_d7",
    "lead_time_norm",
    "holding_cost_norm",
    "stockout_cost_norm",
    "dow_sin",
    "dow_cos",
    "promo_flag",
]


class RLPolicySHAPExplainer:
    """Compute SHAP values for SB3 policies.

    Args:
        model: Trained PPO/SAC model with ``predict``.
        background_states: Background state matrix.
        feature_names: Optional custom feature names.
    """

    def __init__(
        self, model, background_states: np.ndarray, feature_names: list[str] | None = None
    ) -> None:
        self.model = model
        self.background_states = np.asarray(background_states, dtype=float)
        self.feature_names = feature_names or STATE_FEATURE_NAMES

    def _predict_fn(self, states: np.ndarray) -> np.ndarray:
        """Run model prediction over a batch of states.

        Args:
            states: State matrix.

        Returns:
            One-dimensional action array.
        """
        outputs: list[float] = []
        for state in np.asarray(states, dtype=float):
            action, _ = self.model.predict(state.astype(np.float32), deterministic=True)
            outputs.append(float(action[0]))
        return np.asarray(outputs, dtype=float)

    def explain(self, test_states: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Compute SHAP values using KernelExplainer.

        Args:
            test_states: States to explain.
            n_samples: Number of SHAP samples.

        Returns:
            SHAP values array.

        Raises:
            ImportError: If SHAP is not installed.
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError("Install shap to use RLPolicySHAPExplainer") from exc

        background = shap.sample(
            self.background_states, min(int(n_samples), len(self.background_states))
        )
        explainer = shap.KernelExplainer(self._predict_fn, background)
        values = explainer.shap_values(
            np.asarray(test_states, dtype=float), nsamples=int(n_samples)
        )
        logger.info("Computed SHAP values for %d states", len(np.asarray(test_states)))
        return np.asarray(values)

    def plot_summary(self, shap_values: np.ndarray, test_states: np.ndarray) -> None:
        """Render SHAP summary plot to paper figure output.

        Args:
            shap_values: SHAP matrix from ``explain``.
            test_states: Input states aligned to SHAP outputs.
        """
        try:
            import matplotlib
            import shap

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("shap/matplotlib unavailable; skipping SHAP summary plot")
            return

        Path("outputs/paper_figures").mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            np.asarray(test_states, dtype=float),
            feature_names=self.feature_names,
            show=False,
        )
        plt.title("Figure 7 - SHAP feature importance for PPO order policy")
        plt.tight_layout()
        plt.savefig("outputs/paper_figures/fig07_shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP summary figure")
