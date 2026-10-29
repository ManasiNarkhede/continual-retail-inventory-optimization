"""Interpretability utilities for trained TFT models.

This module extracts variable-importance and attention summaries used for
analysis, dashboards, and paper figures.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

plt.switch_backend("Agg")


class TFTInterpreter:
    """Extract interpretability artefacts from a trained TFT model.

    Args:
        model: Trained TemporalFusionTransformer instance.
        dataset: TimeSeriesDataSet used to train the model.
    """

    def __init__(self, model: TemporalFusionTransformer, dataset: TimeSeriesDataSet) -> None:
        self.model = model
        self.dataset = dataset
        self.model.eval()

    def variable_importance(self, dataloader: DataLoader) -> pd.DataFrame:
        """Compute variable-importance scores from TFT interpretation output.

        Args:
            dataloader: Evaluation dataloader.

        Returns:
            DataFrame with columns ``variable``, ``importance``, and ``type``.
        """
        with torch.no_grad():
            prediction = self.model.predict(dataloader, mode="raw", return_x=True)
            if isinstance(prediction, tuple):
                raw_output = prediction[0]
            else:
                raw_output = prediction.output if hasattr(prediction, "output") else prediction
            interpretation = self.model.interpret_output(raw_output, reduction="sum")

        encoder_importance = interpretation.get("encoder_variables")
        static_importance = interpretation.get("static_variables")

        enc_np = (
            encoder_importance.detach().cpu().numpy()
            if hasattr(encoder_importance, "detach")
            else np.asarray(encoder_importance)
        )
        static_np = (
            static_importance.detach().cpu().numpy()
            if hasattr(static_importance, "detach")
            else np.asarray(static_importance)
        )

        known_reals = list(getattr(self.dataset, "time_varying_known_reals", []))
        unknown_reals = list(getattr(self.dataset, "time_varying_unknown_reals", []))
        encoder_names = known_reals + unknown_reals
        static_names = list(getattr(self.dataset, "static_categoricals", [])) + list(
            getattr(self.dataset, "static_reals", [])
        )

        enc_total = float(np.sum(enc_np)) if np.sum(enc_np) != 0 else 1.0
        static_total = float(np.sum(static_np)) if np.sum(static_np) != 0 else 1.0

        enc_df = pd.DataFrame(
            {
                "variable": encoder_names[: len(enc_np)],
                "importance": enc_np / enc_total,
                "type": "encoder",
            }
        )
        static_df = pd.DataFrame(
            {
                "variable": static_names[: len(static_np)],
                "importance": static_np / static_total,
                "type": "static",
            }
        )
        output = pd.concat([enc_df, static_df], ignore_index=True)
        output = output.sort_values("importance", ascending=False).reset_index(drop=True)
        return output

    def attention_weights(self, dataloader: DataLoader, n_samples: int = 128) -> np.ndarray:
        """Aggregate attention weights across a subset of batches.

        Args:
            dataloader: Evaluation dataloader.
            n_samples: Maximum number of samples to aggregate.

        Returns:
            Mean attention matrix with shape ``(T, T)``.
        """
        attention_sum: np.ndarray | None = None
        seen = 0

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0]
                out = self.model(x)
                attn = out.get("attention")
                if attn is None:
                    continue
                attn_np = (
                    attn.detach().cpu().numpy() if hasattr(attn, "detach") else np.asarray(attn)
                )
                mean_attn = attn_np.mean(axis=1).mean(axis=0)
                attention_sum = mean_attn if attention_sum is None else attention_sum + mean_attn

                batch_size = int(attn_np.shape[0]) if attn_np.ndim >= 1 else 1
                seen += batch_size
                if n_samples and seen >= n_samples:
                    break

        if attention_sum is None:
            return np.zeros((1, 1), dtype=float)
        denom = max(1, int(np.ceil(seen / max(1, dataloader.batch_size or 1))))
        return attention_sum / float(denom)

    def seasonal_attention_profile(
        self, attention_matrix: np.ndarray, window: int = 7
    ) -> np.ndarray:
        """Build lag-profile from last-step attention row and smooth it.

        Args:
            attention_matrix: Matrix from ``attention_weights``.
            window: Rolling mean window size.

        Returns:
            Smoothed lag-profile array.
        """
        t = int(attention_matrix.shape[0])
        profile = np.array([attention_matrix[t - 1, t - 1 - k] for k in range(t)])
        return pd.Series(profile).rolling(window=window, min_periods=1, center=True).mean().values

    def save_artefacts(
        self, dataloader: DataLoader, output_dir: str = "reports/figures"
    ) -> tuple[Path, Path]:
        """Persist variable-importance and attention interpretability artifacts.

        Args:
            dataloader: Evaluation dataloader used for interpretability extraction.
            output_dir: Output directory path.

        Returns:
            Tuple of (importance_csv_path, attention_png_path).
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        importance_df = self.variable_importance(dataloader)
        importance_csv = out_dir / "tft_variable_importance.csv"
        importance_json = out_dir / "tft_variable_importance.json"
        importance_df.to_csv(importance_csv, index=False)
        importance_df.to_json(importance_json, orient="records", indent=2)

        attention = self.attention_weights(dataloader)
        attention_npy = out_dir / "tft_attention_weights.npy"
        np.save(attention_npy, attention)
        attention_png = out_dir / "tft_attention_map.png"
        plt.figure(figsize=(8, 6))
        plt.imshow(attention, aspect="auto", cmap="viridis")
        plt.colorbar(label="Attention Weight")
        plt.xlabel("Key Time Step")
        plt.ylabel("Query Time Step")
        plt.title("TFT Attention Map")
        plt.tight_layout()
        plt.savefig(attention_png, dpi=180)
        plt.close()

        metadata_path = out_dir / "tft_interpretability_metadata.json"
        metadata = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "importance_rows": int(len(importance_df)),
            "attention_shape": list(attention.shape),
            "files": {
                "importance_csv": str(importance_csv.name),
                "importance_json": str(importance_json.name),
                "attention_npy": str(attention_npy.name),
                "attention_png": str(attention_png.name),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.info("Saved TFT interpretability artifacts under %s", out_dir)
        return importance_csv, attention_png

    def generate_store_figures(
        self,
        store_id: int,
        reference_df: pd.DataFrame,
        output_dir: str = "reports/figures",
        horizon: int = 14,
    ) -> tuple[Path, Path]:
        """Generate variable-importance and attention-map PNG figures.

        Args:
            store_id: Store identifier to focus figure generation.
            reference_df: Reference dataframe containing Store/Date columns.
            output_dir: Output directory path.
            horizon: Forecast horizon used to build inference dataloader.

        Returns:
            Tuple of (variable_importance_png, attention_map_png).
        """
        from pytorch_forecasting import TimeSeriesDataSet

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        store_df = reference_df[reference_df["Store"] == store_id].copy().sort_values("time_idx")
        if store_df.empty:
            raise KeyError(f"Store {store_id} not found in reference dataframe")

        inference_ds = TimeSeriesDataSet.from_dataset(
            self.dataset,
            store_df,
            predict=True,
            stop_randomization=True,
        )
        loader = inference_ds.to_dataloader(train=False, batch_size=32, num_workers=0)

        var_df = self.variable_importance(loader)
        top_df = var_df.head(15).iloc[::-1]

        var_path = out_dir / f"store_{store_id}_variable_importance.png"
        plt.figure(figsize=(10, 6))
        colors = ["#0f766e" if t == "encoder" else "#d97706" for t in top_df["type"]]
        plt.barh(top_df["variable"], top_df["importance"], color=colors)
        plt.xlabel("Relative Importance")
        plt.ylabel("Feature")
        plt.title(f"Store {store_id} Variable Importance")
        plt.tight_layout()
        plt.savefig(var_path, dpi=180)
        plt.close()

        attention = self.attention_weights(loader, n_samples=max(horizon * 8, 64))
        attn_path = out_dir / f"store_{store_id}_attention_map.png"
        plt.figure(figsize=(7, 6))
        plt.imshow(attention, aspect="auto", cmap="viridis")
        plt.colorbar(label="Attention Weight")
        plt.xlabel("Key Time Step")
        plt.ylabel("Query Time Step")
        plt.title(f"Store {store_id} TFT Attention Map")
        plt.tight_layout()
        plt.savefig(attn_path, dpi=180)
        plt.close()

        return var_path, attn_path
