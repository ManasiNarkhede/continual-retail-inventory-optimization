"""Triple-tier reward functions for inventory control agents.

Cost model tiers
----------------
1. **Holding cost** — linear in ending inventory (excess units).
2. **Order cost** — fixed penalty for any non-zero order to capture
   logistics overhead (truck dispatch, receiving labour).
3. **Stockout penalty** — exponential in missed-demand units:
   ``stockout_cost · (e^(α · shortfall) − 1)`` where *α* controls the
   severity curve.  This penalises large shortfalls far more than small
   ones, discouraging the agent from "barely" stocking out.

The module preserves the ``BaseReward`` / ``ShapedReward`` hierarchy for
backward compatibility and adds ``TripleTierReward`` as the new default.
"""

from __future__ import annotations

import math


class BaseReward:
    """Linear cost-minimization reward (legacy).

    Args:
        holding_cost: Cost per excess unit.
        stockout_cost: Cost per unmet unit.
        order_cost: Fixed cost per order.
    """

    def __init__(
        self,
        holding_cost: float = 0.5,
        stockout_cost: float = 5.0,
        order_cost: float = 10.0,
    ) -> None:
        self.holding_cost = float(holding_cost)
        self.stockout_cost = float(stockout_cost)
        self.order_cost = float(order_cost)

    def __call__(self, excess_units: float, shortfall_units: float, order_placed: bool) -> float:
        """Compute reward from one-step cost terms."""
        total_cost = self.holding_cost * float(excess_units) + self.stockout_cost * float(
            shortfall_units
        )
        if order_placed:
            total_cost += self.order_cost
        return -float(total_cost)


class TripleTierReward:
    """Triple-tier cost model with exponential stockout penalty.

    Tier 1 — Holding:   ``h · excess``
    Tier 2 — Ordering:  ``K`` if ``order_placed`` else 0
    Tier 3 — Stockout:  ``s · (exp(α · shortfall) − 1)``

    The exponential stockout penalty makes large shortfalls
    dramatically more expensive than small ones, which encourages
    the agent to build safety stock against demand spikes.

    Args:
        holding_cost: Per-unit holding cost ``h``.
        order_cost: Fixed ordering cost ``K``.
        stockout_cost: Stockout scaling factor ``s``.
        stockout_severity: Exponential severity factor ``α``.
    """

    def __init__(
        self,
        holding_cost: float = 0.5,
        order_cost: float = 10.0,
        stockout_cost: float = 5.0,
        stockout_severity: float = 0.05,
    ) -> None:
        self.holding_cost = float(holding_cost)
        self.order_cost = float(order_cost)
        self.stockout_cost = float(stockout_cost)
        self.stockout_severity = float(stockout_severity)

    def __call__(self, excess_units: float, shortfall_units: float, order_placed: bool) -> float:
        """Compute triple-tier cost reward.

        Args:
            excess_units: Ending inventory above zero after demand fulfilment.
            shortfall_units: Unmet demand units.
            order_placed: Whether a replenishment order was placed this step.

        Returns:
            Negative cost reward.
        """
        holding = self.holding_cost * float(excess_units)
        ordering = self.order_cost if order_placed else 0.0
        shortfall_val = float(shortfall_units)
        exp_arg = min(self.stockout_severity * shortfall_val, 50.0)
        stockout = self.stockout_cost * (math.exp(exp_arg) - 1.0)
        return -float(holding + ordering + stockout)

    def shaped(
        self,
        stock_before: float,
        stock_after: float,
        excess_units: float,
        shortfall_units: float,
        order_placed: bool,
        gamma: float = 0.99,
        target_stock: float = 100.0,
        shaping_alpha: float = 0.1,
    ) -> float:
        """Compute triple-tier reward with potential-based shaping.

        Adds ``γ · Φ(s') − Φ(s)`` where ``Φ(stock) = −α · |stock − target|``.

        Args:
            stock_before: Stock level before demand was served.
            stock_after: Stock level after demand was served.
            excess_units: Ending inventory above zero.
            shortfall_units: Unmet demand.
            order_placed: Whether an order was placed.
            gamma: Discount factor for shaping.
            target_stock: Desired inventory anchor.
            shaping_alpha: Shaping potential weight.

        Returns:
            Shaped negative cost reward.
        """
        base = self(
            excess_units=excess_units, shortfall_units=shortfall_units, order_placed=order_placed
        )
        phi_before = -shaping_alpha * abs(float(stock_before) - target_stock)
        phi_after = -shaping_alpha * abs(float(stock_after) - target_stock)
        return float(base + gamma * phi_after - phi_before)


class MultiObjectiveReward(TripleTierReward):
    """Multi-objective reward balancing service reliability and cost.

    Objective:
        reward = w_cost * (-cost) + w_service * (service_signal - service_target)

    Where ``service_signal`` is 1.0 when there is no shortfall and declines
    smoothly as shortfall grows.

    Args:
        cost_weight: Weight for negative cost objective.
        service_weight: Weight for service-level objective.
        service_level_target: Desired service level baseline.
    """

    def __init__(
        self,
        holding_cost: float = 0.5,
        order_cost: float = 10.0,
        stockout_cost: float = 5.0,
        stockout_severity: float = 0.05,
        cost_weight: float = 0.6,
        service_weight: float = 0.4,
        service_level_target: float = 0.95,
    ) -> None:
        super().__init__(
            holding_cost=holding_cost,
            order_cost=order_cost,
            stockout_cost=stockout_cost,
            stockout_severity=stockout_severity,
        )
        self.cost_weight = float(cost_weight)
        self.service_weight = float(service_weight)
        self.service_level_target = float(service_level_target)

    def __call__(
        self,
        excess_units: float,
        shortfall_units: float,
        order_placed: bool,
        demand_units: float | None = None,
    ) -> float:
        """Compute weighted multi-objective reward."""
        base_cost_reward = super().__call__(
            excess_units=excess_units,
            shortfall_units=shortfall_units,
            order_placed=order_placed,
        )

        demand = max(
            float(demand_units) if demand_units is not None else float(shortfall_units + 1.0), 1e-6
        )
        unmet_ratio = min(1.0, float(shortfall_units) / demand)
        service_signal = 1.0 - unmet_ratio

        service_term = service_signal - self.service_level_target
        return float(self.cost_weight * base_cost_reward + self.service_weight * service_term)


class ShapedReward(BaseReward):
    """Potential-based shaped reward (legacy, preserved for compatibility).

    Adds ``gamma * Phi(s') - Phi(s)`` to the base reward where
    ``Phi(stock) = -alpha * abs(stock - target_stock)``.

    Args:
        holding_cost: Cost per excess unit.
        stockout_cost: Cost per unmet unit.
        order_cost: Fixed cost per order.
        gamma: Discount factor.
        target_stock: Desired inventory anchor level.
        shaping_alpha: Weight for shaping potential.
    """

    def __init__(
        self,
        holding_cost: float = 0.5,
        stockout_cost: float = 5.0,
        order_cost: float = 10.0,
        gamma: float = 0.99,
        target_stock: float = 100.0,
        shaping_alpha: float = 0.1,
    ) -> None:
        super().__init__(
            holding_cost=holding_cost, stockout_cost=stockout_cost, order_cost=order_cost
        )
        self.gamma = float(gamma)
        self.target_stock = float(target_stock)
        self.shaping_alpha = float(shaping_alpha)

    def potential(self, stock_level: float) -> float:
        """Compute shaping potential Phi(stock)."""
        return -self.shaping_alpha * abs(float(stock_level) - self.target_stock)

    def shaped(
        self,
        stock_before: float,
        stock_after: float,
        excess_units: float,
        shortfall_units: float,
        order_placed: bool,
    ) -> float:
        """Compute base reward plus potential-based shaping term."""
        base_reward = super().__call__(
            excess_units=excess_units,
            shortfall_units=shortfall_units,
            order_placed=order_placed,
        )
        shaping = self.gamma * self.potential(stock_after) - self.potential(stock_before)
        return float(base_reward + shaping)


# -----------------------------------------------------------------------
# Default singleton — cached to avoid per-step allocations
# -----------------------------------------------------------------------

_DEFAULT_REWARD: TripleTierReward | None = None


def compute_inventory_reward(
    excess_units: float,
    shortfall_units: float,
    order_placed: bool,
    holding_cost: float,
    stockout_cost: float,
    order_cost: float,
    stockout_severity: float = 0.05,
) -> float:
    """Backward-compatible functional reward helper using TripleTierReward.

    Caches a singleton ``TripleTierReward`` instance to avoid per-step
    object allocation (millions of calls during training).

    Args:
        excess_units: Unsold inventory after demand.
        shortfall_units: Unmet demand.
        order_placed: Whether an order is placed this step.
        holding_cost: Holding cost per unit.
        stockout_cost: Stockout cost per unit.
        order_cost: Fixed order cost.
        stockout_severity: Exponential severity factor.

    Returns:
        Negative total cost reward.
    """
    global _DEFAULT_REWARD
    if (
        _DEFAULT_REWARD is None
        or _DEFAULT_REWARD.holding_cost != holding_cost
        or _DEFAULT_REWARD.stockout_cost != stockout_cost
        or _DEFAULT_REWARD.order_cost != order_cost
        or _DEFAULT_REWARD.stockout_severity != stockout_severity
    ):
        _DEFAULT_REWARD = TripleTierReward(
            holding_cost=holding_cost,
            order_cost=order_cost,
            stockout_cost=stockout_cost,
            stockout_severity=stockout_severity,
        )
    return _DEFAULT_REWARD(
        excess_units=excess_units, shortfall_units=shortfall_units, order_placed=order_placed
    )
