"""PackNet-style architectural continual-learning strategy.

Includes an **Adaptive Pruning Scheduler** that adjusts the per-task
pruning ratio based on the intransigence metric of the current
experience:

    prune_ratio_k = base_ratio × (1 − β · clamp(I_k / I_max, 0, 1))

- High intransigence (hard task) → lower pruning → keeps more capacity
  for the current task's unique features.
- Low intransigence (easy task) → higher pruning → frees capacity for
  future tasks.

A safety threshold prevents pruning when remaining capacity is too low,
avoiding catastrophic loss of representational power on later tasks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import mlflow
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Adaptive Pruning Scheduler
# -----------------------------------------------------------------------


@dataclass
class AdaptivePruningScheduler:
    """Adjusts pruning ratio per task based on intransigence.

    Attributes:
        base_prune_ratio: Default pruning ratio when intransigence is zero.
        sensitivity: How strongly intransigence reduces pruning (β in formula).
        min_prune_ratio: Floor on pruning ratio to ensure some capacity is freed.
        max_prune_ratio: Ceiling on pruning ratio.
        capacity_safety_threshold: Halt pruning when remaining capacity falls
            below this fraction (prevents catastrophic capacity exhaustion).
        intransigence_history: Recorded intransigence values per task.
    """

    base_prune_ratio: float = 0.5
    sensitivity: float = 0.8
    min_prune_ratio: float = 0.1
    max_prune_ratio: float = 0.75
    capacity_safety_threshold: float = 0.15
    critical_intransigence_threshold: float = 0.25
    intransigence_history: list[float] = field(default_factory=list)

    def compute_ratio(self, intransigence: float) -> float:
        """Compute adaptive pruning ratio for the current task.

        Formula:
            ratio = base × (1 − β · clamp(I / I_max, 0, 1))

        Where I_max is the running maximum of recorded intransigence,
        preventing the ratio from drifting due to scale changes.

        Args:
            intransigence: Intransigence metric for the current experience.

        Returns:
            Adapted pruning ratio clamped to ``[min_prune_ratio, max_prune_ratio]``.
        """
        self.intransigence_history.append(float(intransigence))
        i_max = max(self.intransigence_history) if self.intransigence_history else 1.0
        i_max = max(i_max, 1e-8)  # prevent division by zero
        normalised = max(0.0, min(float(intransigence) / i_max, 1.0))
        ratio = self.base_prune_ratio * (1.0 - self.sensitivity * normalised)
        return float(max(self.min_prune_ratio, min(self.max_prune_ratio, ratio)))

    def should_prune(self, current_capacity_used: float) -> bool:
        """Check whether pruning should proceed given remaining capacity.

        Args:
            current_capacity_used: Fraction of parameters already assigned
                (output of ``PackNetStrategy.capacity_used()``).

        Returns:
            ``True`` if there is enough remaining capacity to prune safely.
        """
        remaining = 1.0 - current_capacity_used
        if remaining < self.capacity_safety_threshold:
            logger.warning(
                "Remaining capacity %.1f%% below safety threshold %.1f%% — skipping pruning",
                remaining * 100,
                self.capacity_safety_threshold * 100,
            )
            return False
        return True


# -----------------------------------------------------------------------
# PackNet Strategy
# -----------------------------------------------------------------------


class PackNetStrategy:
    """Iterative prune-and-freeze strategy for continual learning.

    Args:
        model: Torch model to partition across tasks.
        prune_ratio: Base fraction of free weights pruned each task.
        retrain_epochs: Retraining epochs after pruning.
        adaptive_scheduler: Optional ``AdaptivePruningScheduler`` for
            intransigence-based ratio adjustment.
    """

    def __init__(
        self,
        model: nn.Module,
        prune_ratio: float = 0.5,
        retrain_epochs: int = 3,
        adaptive_scheduler: AdaptivePruningScheduler | None = None,
    ) -> None:
        self.model = model
        self.prune_ratio = float(prune_ratio)
        self.retrain_epochs = int(retrain_epochs)
        # Per-task 0/1 masks of the weights each task owns (disjoint across tasks).
        self.masks: dict[int, dict[str, torch.Tensor]] = {}
        # Per-parameter boolean union of all owned weights, and their frozen values.
        self.assigned: dict[str, torch.Tensor] = {}
        self.frozen_values: dict[str, torch.Tensor] = {}
        self.scheduler = adaptive_scheduler or AdaptivePruningScheduler(
            base_prune_ratio=self.prune_ratio,
        )
        self.layer_prune_history: dict[int, dict[str, float]] = {}
        self.last_prune_ratio = 0.0

    def train_task(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        task_id: int,
        epochs: int = 5,
        device: str = "cpu",
    ) -> float:
        """Train one task while protecting frozen parameters.

        Args:
            dataloader: Task dataloader.
            optimizer: Optimizer over model parameters.
            criterion: Loss function.
            task_id: Integer task ID.
            epochs: Number of epochs.
            device: Training device name.

        Returns:
            Mean training loss.
        """
        self.model.train()
        dev = torch.device(device)
        self.model.to(dev)
        total_loss = 0.0

        for _ in range(int(epochs)):
            epoch_loss = 0.0
            batches = 0
            for x, y in dataloader:
                x = x.to(dev)
                y = y.to(dev)
                optimizer.zero_grad()
                out = self.model(x)
                loss = criterion(out.squeeze(-1), y)
                loss.backward()

                for name, param in self.model.named_parameters():
                    owned = self.assigned.get(name)
                    if owned is not None and param.grad is not None:
                        param.grad.masked_fill_(owned.to(param.grad.device), 0.0)

                optimizer.step()
                epoch_loss += float(loss.item())
                batches += 1

            total_loss += epoch_loss / max(batches, 1)

        mean_loss = total_loss / max(int(epochs), 1)
        logger.info("PackNet task %d training complete with loss %.4f", task_id, mean_loss)
        return float(mean_loss)

    def prune_and_freeze(
        self,
        task_id: int,
        intransigence: float = 0.0,
    ) -> dict[str, float]:
        """Prune low-magnitude free weights and freeze the rest for this task.

        Only weights not owned by earlier tasks are considered. The largest of
        them are assigned to ``task_id`` and frozen; the pruned ones are zeroed
        and stay free for later tasks. The pruning ratio is adjusted by the
        ``AdaptivePruningScheduler`` based on the intransigence of the experience.

        Args:
            task_id: Integer task ID.
            intransigence: Intransigence metric for this task (from CL strategy).

        Returns:
            Mapping from layer name to the fraction of its free weights pruned.
        """
        # Adaptive ratio based on intransigence
        effective_ratio = self.scheduler.compute_ratio(intransigence)
        self.last_prune_ratio = effective_ratio
        logger.info(
            "PackNet task %d adaptive prune ratio: %.3f (intransigence=%.4f)",
            task_id,
            effective_ratio,
            intransigence,
        )

        # Capacity guard — skip if remaining capacity is too low
        if not self.scheduler.should_prune(self.capacity_used()):
            logger.warning("PackNet task %d: pruning skipped (capacity guard)", task_id)
            return {}

        stats: dict[str, float] = {}
        task_masks = self.masks.setdefault(task_id, {})
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.data.dim() < 1:
                continue
            magnitudes = param.data.detach().abs()
            owned = self.assigned.get(name, torch.zeros_like(magnitudes, dtype=torch.bool))
            free = ~owned
            if not bool(free.any()):
                continue

            # Rank-based: prune exactly floor(ratio * n_free) of the smallest free weights.
            free_idx = free.reshape(-1).nonzero().squeeze(1)
            n_keep = free_idx.numel() - int(effective_ratio * free_idx.numel())
            top = magnitudes.reshape(-1)[free_idx].topk(n_keep).indices
            keep = torch.zeros_like(free).reshape(-1)
            keep[free_idx[top]] = True
            keep = keep.reshape(free.shape)
            with torch.no_grad():
                param.data.masked_fill_(free & ~keep, 0.0)

            # Masks are disjoint, so adding accumulates repeated calls for the same task.
            task_masks[name] = task_masks.get(name, torch.zeros_like(param.data)) + keep.float()
            self.assigned[name] = owned | keep
            frozen = self.frozen_values.get(name, torch.zeros_like(param.data))
            self.frozen_values[name] = torch.where(keep, param.data.detach(), frozen)
            stats[name] = 1.0 - float(keep.sum().item()) / float(free.sum().item())

            if mlflow.active_run() is not None:
                metric_name = f"packnet_layer_prune_ratio_{name.replace('.', '_')}"
                mlflow.log_metric(metric_name, float(stats[name]), step=int(task_id))

        avg_pruned = float(np.mean(list(stats.values()))) if stats else 0.0
        self.layer_prune_history[int(task_id)] = dict(stats)
        logger.info(
            "PackNet task %d pruned %.1f%% weights (effective ratio=%.3f, capacity=%.1f%%)",
            task_id,
            100.0 * avg_pruned,
            effective_ratio,
            100.0 * self.capacity_used(),
        )
        return stats

    def apply_masks(self) -> None:
        """Restore every task-owned weight to its frozen value.

        Training outside ``train_task`` (e.g. via Avalanche) updates all weights;
        this re-imposes the frozen assignments while leaving free weights untouched.
        """
        for name, param in self.model.named_parameters():
            owned = self.assigned.get(name)
            if owned is None:
                continue
            owned = owned.to(param.device)
            with torch.no_grad():
                param.data.copy_(
                    torch.where(owned, self.frozen_values[name].to(param.device), param.data)
                )

    def capacity_used(self) -> float:
        """Compute fraction of prunable parameter capacity already assigned to tasks.

        Only trainable parameters with at least one dimension can be assigned,
        so only those count towards the total.

        Returns:
            Capacity ratio in ``[0, 1]``.
        """
        total = 0
        assigned = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.data.dim() < 1:
                continue
            total += param.numel()
            if name in self.assigned:
                assigned += int(self.assigned[name].sum().item())
        return float(assigned / max(total, 1))
