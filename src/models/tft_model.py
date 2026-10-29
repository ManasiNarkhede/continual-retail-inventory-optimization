"""Temporal Fusion Transformer wrapper with MLflow integration.

Includes an asymmetric quantile loss that penalises under-estimation
(low quantiles) more heavily than over-estimation, prioritising stock
availability in downstream inventory decisions.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import torch
from pandas.errors import PerformanceWarning
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from configs.validator import load_yaml_config, validate_config_schema
from src.utils.device import get_device

logger = logging.getLogger(__name__)

__all__ = ["AsymmetricQuantileLoss", "resolve_quantile_indices", "TFTForecaster"]

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=(
        r"Attribute 'loss' is an instance of `nn.Module` and is already saved during "
        r"checkpointing.*"
    ),
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=(
        r"Attribute 'logging_metrics' is an instance of `nn.Module` and is already saved "
        r"during checkpointing.*"
    ),
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The 'predict_dataloader' does not have many workers which may be a bottleneck.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The 'train_dataloader' does not have many workers which may be a bottleneck.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The 'val_dataloader' does not have many workers which may be a bottleneck.*",
)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"pytorch_forecasting.*")
warnings.filterwarnings("ignore", category=PerformanceWarning, module=r"pandas.*")


# -----------------------------------------------------------------------
# Asymmetric quantile loss
# -----------------------------------------------------------------------


class AsymmetricQuantileLoss(QuantileLoss):
    """Quantile loss with tuneable asymmetric penalty.

    For quantiles at or below the median (q <= 0.5), the pinball loss is
    scaled by ``under_penalty_factor`` so that the model is encouraged
    to over-predict rather than under-predict demand.  This directly
    improves downstream service-level metrics.

    Mathematical formulation
    ------------------------
    Standard pinball loss for quantile *q*:

        L_q(y, ŷ) = max(q·(y − ŷ), (q − 1)·(y − ŷ))

    Asymmetric weighting:

        w(q) = under_penalty_factor   if  q <= 0.5
               1.0                     otherwise

        L_asym = Σ_q  w(q) · L_q(y, ŷ)

    Args:
        quantiles: Quantile levels to predict.
        under_penalty_factor: Multiplier for quantiles <= 0.5.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        under_penalty_factor: float = 1.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(quantiles=quantiles, **kwargs)
        self.under_penalty_factor = float(under_penalty_factor)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute asymmetrically weighted quantile loss.

        Args:
            y_pred: Predicted quantiles — shape ``(batch, horizon, n_quantiles)``.
            target: Observed values — shape ``(batch, horizon)`` or broadcastable.

        Returns:
            Loss tensor.
        """
        base_loss = super().loss(y_pred, target)

        # Build per-quantile weight vector on the same device
        weights = torch.ones(len(self.quantiles), device=y_pred.device, dtype=y_pred.dtype)
        for idx, q_val in enumerate(self.quantiles):
            if q_val <= 0.5:
                weights[idx] = self.under_penalty_factor

        # Broadcast multiply — base_loss has quantile dim last
        if base_loss.ndim >= 1 and base_loss.shape[-1] == len(self.quantiles):
            return base_loss * weights
        return base_loss * self.under_penalty_factor


# -----------------------------------------------------------------------
# Quantile index resolver
# -----------------------------------------------------------------------


def resolve_quantile_indices(
    quantiles: list[float],
    targets: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[str, int]:
    """Find column indices for target quantile levels in a quantile list.

    The function picks the index of the quantile value closest to each
    target, so configuration changes do not silently return wrong columns.

    Args:
        quantiles: Ordered quantile list from config (e.g. ``[0.02, 0.1, …, 0.98]``).
        targets: Desired quantile levels to resolve.

    Returns:
        Mapping from ``"p{int(q*100)}"`` to column index.

    Example:
        >>> resolve_quantile_indices([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
        {'p10': 1, 'p50': 3, 'p90': 5}
    """
    indices: dict[str, int] = {}
    for target_q in targets:
        best_idx = int(min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - target_q)))
        label = f"p{int(target_q * 100)}"
        indices[label] = best_idx
    return indices


# -----------------------------------------------------------------------
# TFT Forecaster
# -----------------------------------------------------------------------


class TFTForecaster:
    """Temporal Fusion Transformer for multi-store demand forecasting.

    Args:
        config_path: Path to tft_config.yaml.
        dataset: Fitted TimeSeriesDataSet.
    """

    def __init__(self, config_path: str, dataset: TimeSeriesDataSet | None = None) -> None:
        """Load configuration and initialize forecaster state.

        Args:
            config_path: Path to TFT configuration YAML.
            dataset: Fitted TimeSeriesDataSet used for model construction.
        """
        validate_config_schema(config_path, "configs/schemas/tft_schema.json")
        self.config = load_yaml_config(config_path).get("tft", {})
        self.dataset = dataset
        self.model: TemporalFusionTransformer | None = None
        self._predict_warned: bool = False
        # Pre-compute quantile → column-index mapping from config
        self._q_indices = resolve_quantile_indices(self.config["quantiles"])

    def build(self) -> TemporalFusionTransformer:
        """Instantiate the TFT model from dataset and config.

        Uses ``AsymmetricQuantileLoss`` with the configured quantile set so
        that under-estimation of low quantiles is penalised 1.5× more
        heavily than over-estimation.

        Returns:
            Configured TemporalFusionTransformer.
        """
        cfg = self.config
        if self.dataset is None:
            raise RuntimeError(
                "Dataset is required for build(); "
                "use load_model/load_any for metadata-based loading."
            )
        under_penalty = cfg.get("under_penalty_factor", 1.5)
        self.model = TemporalFusionTransformer.from_dataset(
            self.dataset,
            learning_rate=cfg["learning_rate"],
            hidden_size=cfg["hidden_size"],
            lstm_layers=cfg["lstm_layers"],
            attention_head_size=cfg["num_attention_heads"],
            dropout=cfg["dropout"],
            hidden_continuous_size=cfg["hidden_continuous_size"],
            output_size=cfg["output_size"],
            loss=AsymmetricQuantileLoss(
                quantiles=cfg["quantiles"],
                under_penalty_factor=under_penalty,
            ),
            reduce_on_plateau_patience=cfg["reduce_on_plateau_patience"],
            weight_decay=cfg["weight_decay"],
            log_interval=cfg.get("log_interval", -1),
            log_val_interval=cfg.get("log_val_interval", -1),
        )
        param_count = sum(parameter.numel() for parameter in self.model.parameters())
        logger.info("TFT model built with %d parameters", param_count)
        if mlflow.active_run() is not None:
            mlflow.log_param("tft_param_count", param_count)
            mlflow.log_param("under_penalty_factor", under_penalty)
        self.model.to(get_device("auto"))
        return self.model

    def predict(
        self, dataloader: torch.utils.data.DataLoader, mode: str = "prediction"
    ) -> tuple[Any, Any]:
        """Generate predictions from a dataloader.

        Args:
            dataloader: DataLoader created from a TimeSeriesDataSet.
            mode: prediction, quantiles, or raw.

        Returns:
            Tuple of predictions and index DataFrame.

        Raises:
            RuntimeError: If model has not been built.
        """
        if self.model is None:
            raise RuntimeError("Call build() or load() before predict().")
        result = self.model.predict(dataloader, mode=mode, return_index=True)
        if isinstance(result, tuple):
            predictions = result[0]
            index = result[1] if len(result) > 1 else None
        elif hasattr(result, "output") and hasattr(result, "index"):
            predictions = result.output
            index = result.index
        else:
            predictions = result
            index = None
        return predictions, index

    def predict_for_store(
        self,
        store_id: int,
        horizon: int,
        training_dataset: TimeSeriesDataSet,
        reference_df: pd.DataFrame,
    ) -> tuple[torch.Tensor, list[pd.Timestamp]]:
        """Predict quantiles for a single store from a fitted dataset.

        The returned tensor columns correspond to the configured quantile
        list.  Use ``quantile_indices`` to resolve p10/p50/p90 positions.

        Args:
            store_id: Rossmann store identifier.
            horizon: Number of days to return.
            training_dataset: Reference TimeSeriesDataSet used to create inference dataset.
            reference_df: Feature dataframe containing Store and Date columns.

        Returns:
            Tuple of prediction tensor and forecast dates.

        Raises:
            RuntimeError: If model is unavailable.
            KeyError: If store_id does not exist in reference_df.
        """
        if self.model is None:
            raise RuntimeError("Call build() or load() before predict_for_store().")

        store_col = (
            "Store"
            if "Store" in reference_df.columns
            else "store" if "store" in reference_df.columns else None
        )
        if store_col is None:
            raise KeyError("Store column not found in reference data")
        store_df = reference_df[reference_df[store_col] == store_id].copy()
        if store_df.empty:
            raise KeyError(f"Store {store_id} not found")

        for categorical_column in self.config.get("static_categoricals", []):
            if categorical_column in store_df.columns:
                store_df[categorical_column] = store_df[categorical_column].astype(str)

        real_columns = (
            self.config.get("static_reals", [])
            + self.config.get("time_varying_known_reals", [])
            + self.config.get("time_varying_unknown_reals", [])
            + [self.config.get("target", "")]
        )
        for real_column in [
            column for column in real_columns if column and column in store_df.columns
        ]:
            store_df[real_column] = pd.to_numeric(store_df[real_column], errors="coerce").astype(
                float
            )

        store_df = store_df.sort_values("time_idx")
        inference_ds = TimeSeriesDataSet.from_dataset(
            training_dataset,
            store_df,
            predict=True,
            stop_randomization=True,
        )
        if not self._predict_warned:
            self._predict_warned = True

        loader = inference_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
        predictions = self.model.predict(loader, mode="quantiles")
        if isinstance(predictions, torch.Tensor):
            prediction_tensor = predictions
        else:
            prediction_tensor = torch.as_tensor(predictions)
        if prediction_tensor.ndim == 3:
            prediction_tensor = prediction_tensor[0]
        if prediction_tensor.ndim != 2:
            raise RuntimeError("Unexpected quantile prediction tensor shape.")

        last_date = (
            pd.to_datetime(store_df["Date"].iloc[-1])
            if "Date" in store_df.columns
            else pd.Timestamp.today()
        )
        dates = [last_date + pd.Timedelta(days=i + 1) for i in range(horizon)]
        return prediction_tensor[:horizon], dates

    @property
    def quantile_indices(self) -> dict[str, int]:
        """Return mapping from ``p10`` / ``p50`` / ``p90`` to tensor column index.

        The mapping is derived dynamically from the config ``quantiles``
        list so that API routers never need to hardcode column offsets.
        """
        return self._q_indices

    def save(self, path: str) -> None:
        """Save model weights to disk and optionally log artifact.

        Args:
            path: Destination checkpoint path.

        Raises:
            RuntimeError: If model is not built.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output)
        if mlflow.active_run() is not None:
            mlflow.log_artifact(str(output))

    def save_model(self, base_path: str) -> None:
        """Persist model weights and dataset metadata atomically.

        Produces:
          - ``<base_path>.pt`` (state dict)
          - ``<base_path>.meta.json`` (config + dataset schema)
        """
        if self.model is None:
            raise RuntimeError("No model to save.")

        root = Path(base_path)
        root.parent.mkdir(parents=True, exist_ok=True)
        weights_path = root.with_suffix(".pt")
        meta_path = root.with_suffix(".meta.json")

        torch.save(self.model.state_dict(), weights_path)

        metadata = {
            "quantiles": self.config.get("quantiles", []),
            "target": self.config.get("target", "sales"),
            "max_encoder_length": self.config.get("max_encoder_length"),
            "max_prediction_length": self.config.get("max_prediction_length"),
            "group_ids": ["Store"],
            "static_categoricals": self.config.get("static_categoricals", []),
            "static_reals": self.config.get("static_reals", []),
            "time_varying_known_reals": self.config.get("time_varying_known_reals", []),
            "time_varying_unknown_reals": self.config.get("time_varying_unknown_reals", []),
            "under_penalty_factor": self.config.get("under_penalty_factor", 1.5),
            "model_hparams": self._safe_hparams(dict(getattr(self.model, "hparams", {}))),
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if mlflow.active_run() is not None:
            mlflow.log_artifact(str(weights_path))
            mlflow.log_artifact(str(meta_path))

    @staticmethod
    def _safe_hparams(payload: dict[str, Any]) -> dict[str, Any]:
        """Make model hparams JSON-serializable for bundle metadata."""

        def _convert(value: Any) -> Any:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, dict):
                return {str(k): _convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert(item) for item in value]
            return str(value)

        return {str(key): _convert(value) for key, value in payload.items()}

    @staticmethod
    def _bundle_paths(base_or_checkpoint_path: str) -> tuple[Path, Path, Path]:
        """Resolve bundle base, weights path, and metadata path.

        Accepts either a base path (``.../tft_latest``) or an explicit
        checkpoint path (``.../tft_latest.pt``).
        """
        raw_path = Path(base_or_checkpoint_path)
        base = raw_path.with_suffix("") if raw_path.suffix == ".pt" else raw_path
        weights_path = base.with_suffix(".pt")
        meta_path = base.with_suffix(".meta.json")
        return base, weights_path, meta_path

    @classmethod
    def load(
        cls,
        path: str,
        config_path: str,
        dataset: TimeSeriesDataSet | None = None,
        metadata_path: str | None = None,
    ) -> "TFTForecaster":
        """Load a saved TFT model from disk.

        Args:
            path: Path to state dict file.
            config_path: Path to config file.
            dataset: Dataset used to initialize architecture.
            metadata_path: Optional metadata path for dataset-free reconstruction.

        Returns:
            Loaded TFTForecaster instance.
        """
        forecaster = cls(config_path, dataset)
        meta_candidate = (
            Path(metadata_path) if metadata_path else Path(path).with_suffix(".meta.json")
        )
        if dataset is not None:
            forecaster.build()
        elif meta_candidate.exists():
            forecaster.model = cls._build_model_from_metadata(meta_candidate)
        else:
            raise RuntimeError("Dataset-free load requires metadata bundle (.meta.json).")

        state = torch.load(path, map_location="cpu")
        forecaster.model.load_state_dict(state)
        forecaster.model.to(get_device("auto"))
        return forecaster

    @staticmethod
    def _build_model_from_metadata(meta_path: Path) -> TemporalFusionTransformer:
        """Reconstruct TFT architecture directly from bundle metadata."""
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        hparams = dict(payload.get("model_hparams", {}))
        hparams.pop("loss", None)
        hparams.pop("logging_metrics", None)
        quantiles = payload.get("quantiles", [0.1, 0.5, 0.9])
        under_penalty = float(payload.get("under_penalty_factor", 1.5))
        loss = AsymmetricQuantileLoss(quantiles=quantiles, under_penalty_factor=under_penalty)
        model = TemporalFusionTransformer(loss=loss, **hparams)
        return model

    @classmethod
    def load_model(
        cls,
        base_path: str,
        config_path: str,
        dataset: TimeSeriesDataSet | None = None,
    ) -> "TFTForecaster":
        """Load model bundle produced by ``save_model``.

        Validates metadata file presence and then restores model weights.
        """
        _base, weights_path, meta_path = cls._bundle_paths(base_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights bundle: {weights_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata bundle: {meta_path}")
        _ = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls.load(
            path=str(weights_path),
            config_path=config_path,
            dataset=dataset,
            metadata_path=str(meta_path),
        )

    @classmethod
    def load_any(
        cls,
        path_or_base: str,
        config_path: str,
        dataset: TimeSeriesDataSet | None = None,
    ) -> "TFTForecaster":
        """Load TFT from bundle when available, otherwise from checkpoint.

        Priority:
        1) ``<base>.pt`` + ``<base>.meta.json`` bundle.
        2) checkpoint-only ``.pt`` file.
        """
        base, weights_path, meta_path = cls._bundle_paths(path_or_base)

        if weights_path.exists() and meta_path.exists():
            return cls.load_model(base_path=str(base), config_path=config_path, dataset=dataset)
        if weights_path.exists():
            return cls.load(
                path=str(weights_path),
                config_path=config_path,
                dataset=dataset,
                metadata_path=str(meta_path) if meta_path.exists() else None,
            )

        raise FileNotFoundError(
            "Unable to locate TFT model artifacts. "
            f"Expected bundle/checkpoint near: {weights_path}"
        )
