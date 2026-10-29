"""Avalanche continual learning strategy wrapper for demand forecasting.

Adds intransigence tracking to quantify how much learning a new experience
interferes with the model's ability to retain previous knowledge.  The
intransigence metric ``I_k`` for experience *k* is defined as:

    I_k = L_joint(k) − L_isolated(k)

where ``L_joint`` is the loss on experience *k* after sequential training
on all experiences up to *k*, and ``L_isolated`` is the loss achieved by
a fresh model trained only on experience *k*.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import torch
import torch.nn.functional as F
from torch.optim import Adam

from configs.validator import load_yaml_config, validate_config_schema
from src.models.tft_model import AsymmetricQuantileLoss
from src.utils.device import get_device

if TYPE_CHECKING:
    from src.models.tft_model import TFTForecaster

logger = logging.getLogger(__name__)


def _move_to_device(item: Any, device: torch.device) -> Any:
    """Move tensors nested in lists/tuples/dicts to target device."""
    if torch.is_tensor(item):
        return item.to(device)
    if isinstance(item, dict):
        return {key: _move_to_device(value, device) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        moved = [_move_to_device(value, device) for value in item]
        return type(item)(moved)
    return item


def _prediction_tensor(output: Any) -> torch.Tensor:
    """Extract prediction tensor from heterogeneous model outputs."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        if "prediction" in output and torch.is_tensor(output["prediction"]):
            return output["prediction"]
        first_tensor = next((value for value in output.values() if torch.is_tensor(value)), None)
        if first_tensor is not None:
            return first_tensor
    if hasattr(output, "prediction") and torch.is_tensor(getattr(output, "prediction")):
        return getattr(output, "prediction")
    if isinstance(output, (list, tuple)):
        first_tensor = next((value for value in output if torch.is_tensor(value)), None)
        if first_tensor is not None:
            return first_tensor
    raise TypeError("Model output does not contain a tensor prediction.")


def _batch_loss(
    model: torch.nn.Module, criterion: torch.nn.Module, batch: Any, device: torch.device
) -> torch.Tensor:
    """Compute loss for a batch supporting tensor and dict inputs."""
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise TypeError("Batch must contain at least inputs and targets.")

    x = _move_to_device(batch[0], device)
    y = _move_to_device(batch[1], device)
    if isinstance(y, (list, tuple)) and len(y) > 0:
        y = y[0]

    output = model(x)

    if isinstance(criterion, AsymmetricQuantileLoss):
        pred = _prediction_tensor(output)
        if pred.ndim == 1:
            pred = pred.unsqueeze(0).unsqueeze(-1)
        elif pred.ndim == 2:
            pred = pred.unsqueeze(-1)

        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        y = y.float()
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if y.ndim > 2 and y.shape[-1] == 1:
            y = y.squeeze(-1)

        if pred.ndim >= 2 and y.ndim >= 2 and pred.shape[1] != y.shape[1]:
            aligned = min(pred.shape[1], y.shape[1])
            pred = pred[:, :aligned, ...]
            y = y[:, :aligned]
        return criterion(pred, y)

    return criterion(output, y)


class TFTAwareSmoothL1Loss(torch.nn.Module):
    """SmoothL1 loss that supports TFT outputs and tuple targets."""

    def forward(self, prediction: Any, target: Any) -> torch.Tensor:  # noqa: D401
        if isinstance(target, (list, tuple)) and len(target) > 0:
            target = target[0]
        if not torch.is_tensor(target):
            target = torch.as_tensor(target)

        pred = _prediction_tensor(prediction) if not torch.is_tensor(prediction) else prediction
        pred = pred.float()
        target = target.float()

        if pred.ndim >= 3 and pred.shape[-1] > 1:
            pred = pred[..., pred.shape[-1] // 2]
        if pred.ndim > 1 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        if target.ndim > 1 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        if pred.shape != target.shape:
            pred = pred.reshape(-1)
            target = target.reshape(-1)
            aligned = min(pred.shape[0], target.shape[0])
            pred = pred[:aligned]
            target = target[:aligned]

        return F.smooth_l1_loss(pred, target, beta=1.0)


class AvalancheCLStrategy:
    """Continual learning wrapper for sequential monthly training.

    Args:
        model: Forecasting backbone model.
        config_path: Path to cl_config.yaml.
    """

    STRATEGY_MAP = {"EWC": "EWC", "Replay": "Replay", "Naive": "Naive"}

    def __init__(
        self, model: "TFTForecaster | torch.nn.Module", config_path: str = "configs/cl_config.yaml"
    ) -> None:
        """Initialize continual learning strategy wrapper.

        Args:
            model: Forecasting backbone model instance.
            config_path: Path to continual learning YAML config.
        """
        validate_config_schema(config_path, "configs/schemas/cl_schema.json")
        self.config = load_yaml_config(config_path).get("continual_learning", {})

        if isinstance(model, torch.nn.Module):
            self.forecaster_config = dict(getattr(model, "config", {}) or {})
            self.backbone = model
        elif hasattr(model, "model") and isinstance(getattr(model, "model"), torch.nn.Module):
            self.forecaster_config = dict(getattr(model, "config", {}) or {})
            self.backbone = getattr(model, "model")
        else:
            raise TypeError(
                "AvalancheCLStrategy expects a TFTForecaster instance or torch.nn.Module backbone."
            )

        self.model = self.backbone
        self._initial_state = copy.deepcopy(self.model.state_dict())
        self._log_file_handle = None
        self.eval_plugin = self._build_eval_plugin()
        self.strategy = self._build_strategy()
        self.intransigence_values: list[float] = []
        self.dynamic_prune_threshold = float(self.config.get("dynamic_prune_threshold", 0.08))
        self.packnet_prune_base = float(self.config.get("packnet_base_prune_ratio", 0.5))
        self.packnet_enabled = bool(self.config.get("enable_packnet_scheduler", False))

    def _build_criterion(self) -> AsymmetricQuantileLoss:
        """Build TFT-native criterion used across joint and isolated training."""
        quantiles = list(self.forecaster_config.get("quantiles", [0.1, 0.5, 0.9]))
        under_penalty = float(self.forecaster_config.get("under_penalty_factor", 1.5))
        return AsymmetricQuantileLoss(quantiles=quantiles, under_penalty_factor=under_penalty)

    def _build_eval_plugin(self) -> Any:
        """Create Avalanche evaluation plugin with CL metrics.

        Returns:
            Configured EvaluationPlugin instance.
        """
        from avalanche.evaluation.metrics import (
            accuracy_metrics,
            bwt_metrics,
            forgetting_metrics,
            forward_transfer_metrics,
            loss_metrics,
        )
        from avalanche.logging import InteractiveLogger, TextLogger
        from avalanche.training.plugins import EvaluationPlugin

        log_path = Path("logs/cl_log.txt")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file_handle = log_path.open("a", encoding="utf-8")
        text_logger = TextLogger(self._log_file_handle)

        return EvaluationPlugin(
            accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            bwt_metrics(experience=True, stream=True),
            forward_transfer_metrics(experience=True, stream=True),
            loggers=[InteractiveLogger(), text_logger],
        )

    def _build_strategy(self) -> Any:
        """Instantiate configured Avalanche strategy.

        Returns:
            Avalanche strategy object.

        Raises:
            ValueError: If strategy name is unknown.
        """
        cfg = self.config
        strategy_name = cfg["strategy"]
        if strategy_name not in self.STRATEGY_MAP:
            raise ValueError(f"Unknown strategy '{strategy_name}'.")

        from avalanche.training.supervised import EWC, Naive, Replay

        strategy_map = {"EWC": EWC, "Replay": Replay, "Naive": Naive}

        optimizer = Adam(self.model.parameters(), lr=1e-3)
        criterion = self._build_criterion()

        # Support cpu, cuda, and mps (Apple Silicon)
        device = get_device(cfg.get("device", "auto"))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "optimizer": optimizer,
            "criterion": criterion,
            "train_mb_size": cfg["train_mb_size"],
            "train_epochs": cfg["train_epochs"],
            "eval_mb_size": cfg["eval_mb_size"],
            "evaluator": self.eval_plugin,
            "device": device,
        }
        if strategy_name == "EWC":
            kwargs["ewc_lambda"] = cfg["ewc_lambda"]
        if strategy_name == "Replay":
            kwargs["mem_size"] = cfg["replay_mem_size"]

        return strategy_map[strategy_name](**kwargs)

    def _compute_isolated_loss(self, experience: Any) -> float:
        """Train a fresh model copy on a single experience and return its loss.

        This provides the *isolated* baseline for intransigence computation.

        Args:
            experience: Single Avalanche experience.

        Returns:
            Mean loss of a fresh model trained only on this experience.
        """
        cfg = self.config
        fresh_model = copy.deepcopy(self.model)
        fresh_model.load_state_dict(copy.deepcopy(self._initial_state))

        optimizer = Adam(fresh_model.parameters(), lr=1e-3)
        criterion = self._build_criterion()

        device = get_device(cfg.get("device", "auto"))
        fresh_model.to(device)
        fresh_model.train()

        total_loss = 0.0
        n_batches = 0
        dataset = experience.dataset
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg["train_mb_size"],
            shuffle=True,
        )
        for _ in range(cfg["train_epochs"]):
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                optimizer.zero_grad()
                try:
                    loss = _batch_loss(fresh_model, criterion, batch, device)
                except Exception:
                    continue
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def train_all_experiences(self, benchmark: Any) -> dict[str, dict[str, float]]:
        """Train and evaluate sequentially on all benchmark experiences.

        Computes the **intransigence metric** for each experience as the
        difference between the joint-training loss and the isolated loss.

        Args:
            benchmark: Avalanche benchmark object.

        Returns:
            Per-experience metric dictionary including intransigence values.
        """
        all_results: dict[str, dict[str, float]] = {}
        total_experiences = len(benchmark.train_stream)
        self.intransigence_values = []

        packnet = None
        if self.packnet_enabled:
            try:
                from src.cl.packnet import AdaptivePruningScheduler, PackNetStrategy

                scheduler = AdaptivePruningScheduler(base_prune_ratio=self.packnet_prune_base)
                packnet = PackNetStrategy(
                    model=self.model,
                    prune_ratio=self.packnet_prune_base,
                    adaptive_scheduler=scheduler,
                )
            except Exception as exc:
                logger.warning("PackNet scheduler unavailable: %s", exc)
                packnet = None

        for index, experience in enumerate(benchmark.train_stream):
            logger.info("Training on experience %d/%d", index + 1, total_experiences)

            if packnet is not None:
                try:
                    packnet.apply_masks()
                except Exception:
                    pass

            # Compute isolated loss before joint training modifies the model
            isolated_loss = self._compute_isolated_loss(experience)

            self.strategy.train(experience)
            eval_results = self.strategy.eval(benchmark.test_stream[: index + 1])

            # Extract joint loss for this experience
            joint_loss = isolated_loss  # fallback
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    key_str = str(key).lower()
                    if f"experience_{index}" in key_str and "loss" in key_str:
                        joint_loss = float(value)
                        break

            intransigence = max(0.0, joint_loss - isolated_loss)
            self.intransigence_values.append(intransigence)
            logger.info(
                "Experience %d intransigence: %.4f (joint=%.4f, isolated=%.4f)",
                index,
                intransigence,
                joint_loss,
                isolated_loss,
            )

            flattened: dict[str, float] = {"intransigence": intransigence}
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    metric_key = str(key).replace("/", "_")
                    flattened[metric_key] = float(value)
                    if mlflow.active_run() is not None:
                        mlflow.log_metric(metric_key, float(value), step=index)

            if mlflow.active_run() is not None:
                mlflow.log_metric("intransigence", intransigence, step=index)

            dynamic_pruned = False
            adaptive_ratio = 0.0
            if packnet is not None and intransigence >= self.dynamic_prune_threshold:
                try:
                    packnet.prune_and_freeze(task_id=index, intransigence=intransigence)
                    packnet.apply_masks()
                    dynamic_pruned = True
                    adaptive_ratio = float(packnet.scheduler.compute_ratio(intransigence))
                    logger.info(
                        "Dynamic PackNet pruning triggered at experience %d (I=%.4f, ratio=%.3f)",
                        index,
                        intransigence,
                        adaptive_ratio,
                    )
                except Exception as exc:
                    logger.warning(
                        "Dynamic PackNet pruning failed at experience %d: %s", index, exc
                    )

            flattened["dynamic_pruned"] = float(dynamic_pruned)
            flattened["adaptive_prune_ratio"] = float(adaptive_ratio)
            if packnet is not None:
                flattened["intransigence_critical_alert"] = float(
                    intransigence >= float(packnet.scheduler.critical_intransigence_threshold)
                )
            all_results[f"experience_{index}"] = flattened
        return all_results

    def save_metrics(self, output_path: str = "data/processed/cl_metrics_latest.json") -> None:
        """Persist CL metrics to a JSON artifact for API consumption.

        Args:
            output_path: Destination path for metrics JSON.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "intransigence_per_experience": self.intransigence_values,
            "strategy": self.config["strategy"],
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved CL metrics to %s", output)

    def close(self) -> None:
        """Release resources (file handles)."""
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None
        if self.eval_plugin is not None:
            for logger_obj in getattr(self.eval_plugin, "loggers", []):
                close_fn = getattr(logger_obj, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass

    def __del__(self) -> None:
        self.close()
