"""Unified evaluation harness for forecasting and inventory backtests."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.eval.forecast_metrics import (
    mase,
    quantile_coverage,
    rmse,
    smape,
    underestimation_rate,
    weighted_smape,
)
from src.rl.reward import TripleTierReward

logger = logging.getLogger(__name__)

_DEFAULT_TFT_CHECKPOINT = Path("checkpoints/tft/tft_latest.pt")
_DEFAULT_TFT_CONFIG = Path("data/processed/tft_runtime_config.yaml")
_DEFAULT_REFERENCE_PARQUET = Path("data/processed/tft_reference.parquet")
_DEFAULT_RL_CONFIG = Path("configs/rl_config.yaml")
_DEFAULT_RL_CHECKPOINT = Path("checkpoints/rl/inventory_agent.zip")


@dataclass
class ModelResult:
    """Container for one model evaluation row."""

    model_name: str
    smape: float = 0.0
    weighted_smape: float = 0.0
    rmse: float = 0.0
    mase: float = 0.0
    coverage: float = 0.0
    underestimation_rate: float = 0.0
    service_level: float = 0.0
    fill_rate: float = 0.0
    total_cost: float = 0.0
    latency_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class EvaluationHarness:
    """Run end-to-end model and baseline evaluation on held-out demand data."""

    def __init__(
        self,
        output_dir: str = "outputs/evaluation",
        log_to_mlflow: bool = True,
        tft_checkpoint: str | Path | None = None,
        tft_config: str | Path | None = None,
        reference_parquet: str | Path | None = None,
        rl_config: str | Path | None = None,
        rl_checkpoint: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_to_mlflow = bool(log_to_mlflow)

        self.tft_checkpoint = Path(tft_checkpoint) if tft_checkpoint else _DEFAULT_TFT_CHECKPOINT
        self.tft_config = Path(tft_config) if tft_config else _DEFAULT_TFT_CONFIG
        self.reference_parquet = (
            Path(reference_parquet) if reference_parquet else _DEFAULT_REFERENCE_PARQUET
        )
        self.rl_config = Path(rl_config) if rl_config else _DEFAULT_RL_CONFIG
        self.rl_checkpoint = Path(rl_checkpoint) if rl_checkpoint else _DEFAULT_RL_CHECKPOINT

    def run_all(
        self,
        val_df: pd.DataFrame,
        forecaster: Any = None,
        rl_agent: Any = None,
    ) -> pd.DataFrame:
        """Evaluate all available models and baselines.

        Runs Naive, EOQ, and Base-stock baselines on every invocation.
        Optionally evaluates TFT and TFT+RL when models are supplied.
        """
        from src.eval.baselines import BaseStockBaseline, EOQBaseline, NaiveSeasonalBaseline

        results: list[ModelResult] = [
            NaiveSeasonalBaseline(seasonality=7).evaluate(val_df),
            EOQBaseline().evaluate(val_df),
            BaseStockBaseline().evaluate(val_df),
        ]

        loaded_forecaster = (
            self._load_tft_forecaster(forecaster) if forecaster is not None else None
        )
        if loaded_forecaster is not None:
            results.append(self._eval_tft(forecaster=loaded_forecaster, val_df=val_df))

        loaded_agent = (
            self._load_rl_agent(rl_agent, val_df=val_df) if rl_agent is not None else None
        )
        if loaded_forecaster is not None and loaded_agent is not None:
            results.append(
                self._eval_rl(rl_agent=loaded_agent, val_df=val_df, forecaster=loaded_forecaster)
            )

        output_df = pd.DataFrame([vars(item) for item in results])
        out_path = self.output_dir / "evaluation_results.csv"
        output_df.to_csv(out_path, index=False)

        if self.log_to_mlflow and mlflow.active_run() is None:
            self._log_to_mlflow(output_df, out_path)
        return output_df

    def _load_tft_forecaster(self, forecaster: Any) -> Any | None:
        """Return a ready-to-use TFT forecaster object."""
        if forecaster is not True and hasattr(forecaster, "predict"):
            return forecaster

        if (
            not self.tft_checkpoint.exists()
            or not self.tft_config.exists()
            or not self.reference_parquet.exists()
        ):
            return None

        try:
            from src.models.tft_dataset import build_tft_dataset
            from src.models.tft_model import TFTForecaster

            ref_df = pd.read_parquet(self.reference_parquet)
            training_dataset = build_tft_dataset(ref_df, config_path=str(self.tft_config))
            loaded = TFTForecaster.load_any(
                path_or_base=str(self.tft_checkpoint),
                config_path=str(self.tft_config),
                dataset=training_dataset,
            )
            loaded.model.eval()
            return loaded
        except Exception as exc:
            logger.warning("Failed to load TFT forecaster: %s", exc)
            return None

    def _load_rl_agent(self, rl_agent: Any, val_df: pd.DataFrame) -> Any | None:
        """Return a ready-to-use RL agent object."""
        if rl_agent is not True and hasattr(rl_agent, "predict"):
            return rl_agent

        if not self.rl_checkpoint.exists() or not self.rl_config.exists():
            return None

        try:
            from stable_baselines3 import PPO

            from src.rl.agent import RLInventoryAgent
            from src.rl.inventory_env import RetailInventoryEnv

            demand = self._sales_series(val_df)
            env = RetailInventoryEnv(config_path=str(self.rl_config), demand_series=demand)
            agent = RLInventoryAgent(config_path=str(self.rl_config), env=env)
            agent.model = PPO.load(str(self.rl_checkpoint))
            return agent
        except Exception as exc:
            logger.warning("Failed to load RL agent: %s", exc)
            return None

    @staticmethod
    def _sales_series(val_df: pd.DataFrame) -> np.ndarray:
        if "Sales" in val_df.columns:
            return val_df["Sales"].to_numpy(dtype=float)
        if "sales" in val_df.columns:
            return val_df["sales"].to_numpy(dtype=float)
        raise KeyError("Validation dataframe must contain 'Sales' or 'sales'.")

    def _eval_tft(self, forecaster: Any, val_df: pd.DataFrame) -> ModelResult:
        """Backtest TFT with real quantile predictions and inventory simulation."""
        from src.models.tft_dataset import build_tft_dataset

        reference_df = pd.read_parquet(self.reference_parquet)
        train_ds = build_tft_dataset(reference_df, config_path=str(self.tft_config))
        val_ds = build_tft_dataset(
            reference_df,
            config_path=str(self.tft_config),
            training=False,
            training_dataset=train_ds,
        )
        val_loader = val_ds.to_dataloader(train=False, batch_size=128, num_workers=0)

        y_true_batches: list[np.ndarray] = []
        baseline_batches: list[np.ndarray] = []
        for x_batch, target_tuple in val_loader:
            target = target_tuple[0]
            target_np = (
                target.detach().cpu().numpy() if hasattr(target, "detach") else np.asarray(target)
            )
            y_true_batches.append(target_np.reshape(-1).astype(float))

            encoder_target = x_batch.get("encoder_target")
            if encoder_target is not None:
                enc_np = (
                    encoder_target.detach().cpu().numpy()
                    if hasattr(encoder_target, "detach")
                    else np.asarray(encoder_target)
                )
                last_value = enc_np[:, -1].reshape(-1, 1)
                repeated = np.repeat(last_value, target_np.shape[1], axis=1)
                baseline_batches.append(repeated.reshape(-1).astype(float))

        t_start = time.perf_counter()
        predictions, _ = forecaster.predict(val_loader, mode="quantiles")
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        pred_np = (
            predictions.detach().cpu().numpy()
            if hasattr(predictions, "detach")
            else np.asarray(predictions)
        )
        if pred_np.ndim == 3:
            pred_np = pred_np.reshape(-1, pred_np.shape[-1])

        y_true = np.concatenate(y_true_batches) if y_true_batches else np.array([], dtype=float)
        aligned = min(len(y_true), len(pred_np))
        y_true = y_true[:aligned]
        pred_np = pred_np[:aligned]
        if aligned == 0:
            raise RuntimeError("TFT evaluation produced zero aligned rows.")

        q_idx = forecaster.quantile_indices
        idx_p10 = q_idx.get("p10", 1)
        idx_p50 = q_idx.get("p50", 3)
        idx_p90 = q_idx.get("p90", 5)

        p10 = pred_np[:, idx_p10]
        p50 = pred_np[:, idx_p50]
        p90 = pred_np[:, idx_p90]

        finite = np.isfinite(y_true) & np.isfinite(p10) & np.isfinite(p50) & np.isfinite(p90)
        y_true = y_true[finite]
        p10 = p10[finite]
        p50 = p50[finite]
        p90 = p90[finite]
        if y_true.size == 0:
            raise RuntimeError("No finite rows available for TFT metric computation.")

        baseline_pred = np.array([], dtype=float)
        if baseline_batches:
            baseline_pred = np.concatenate(baseline_batches)[:aligned][finite]

        sl, fr, cost = self._inventory_backtest_from_forecast(
            demand=y_true,
            forecast_p10=p10,
            forecast_p50=p50,
            forecast_p90=p90,
        )

        metrics_extra: dict[str, Any] = {
            "baseline_smape": (
                float(smape(y_true, baseline_pred)) if baseline_pred.size == y_true.size else 0.0
            ),
            "smape_uplift_vs_baseline": (
                float(smape(y_true, baseline_pred) - smape(y_true, p50))
                if baseline_pred.size == y_true.size
                else 0.0
            ),
        }

        return ModelResult(
            model_name="TFT (ours)",
            smape=float(smape(y_true, p50)),
            weighted_smape=float(weighted_smape(y_true, p50, under_weight=1.5, over_weight=1.0)),
            rmse=float(rmse(y_true, p50)),
            mase=float(mase(y_true, p50)),
            coverage=float(quantile_coverage(y_true, p10, p90)),
            underestimation_rate=float(underestimation_rate(y_true, p50)),
            service_level=float(sl),
            fill_rate=float(fr),
            total_cost=float(cost),
            latency_ms=float(elapsed_ms / max(len(y_true), 1)),
            extra=metrics_extra,
        )

    def _inventory_backtest_from_forecast(
        self,
        demand: np.ndarray,
        forecast_p10: np.ndarray,
        forecast_p50: np.ndarray,
        forecast_p90: np.ndarray,
        initial_stock: float = 120.0,
        lead_time: int = 2,
    ) -> tuple[float, float, float]:
        """Simulate inventory policy driven by forecast median and spread."""
        reward_fn = TripleTierReward(
            holding_cost=0.5, order_cost=10.0, stockout_cost=5.0, stockout_severity=0.05
        )

        n = len(demand)
        stock = float(initial_stock)
        pipeline = [0.0] * max(int(lead_time), 1)

        stockout_days = 0
        total_demand = 0.0
        total_fulfilled = 0.0
        total_cost = 0.0

        for t in range(n):
            arriving = pipeline.pop(0)
            stock += arriving

            d_t = float(max(demand[t], 0.0))
            fulfilled = min(stock, d_t)
            shortfall = max(0.0, d_t - stock)
            stock = max(0.0, stock - d_t)

            total_demand += d_t
            total_fulfilled += fulfilled
            if shortfall > 0.0:
                stockout_days += 1

            spread = max(0.0, float(forecast_p90[t] - forecast_p10[t]))
            target = max(0.0, float(forecast_p50[t]) * (lead_time + 1) + 0.5 * spread)
            inv_position = stock + sum(pipeline)
            order_qty = max(0.0, target - inv_position)
            order_placed = order_qty > 0.5
            pipeline.append(order_qty)

            reward = reward_fn(
                excess_units=stock, shortfall_units=shortfall, order_placed=order_placed
            )
            total_cost += -float(reward)

        service = float((n - stockout_days) / max(n, 1))
        fill = float(min(1.0, total_fulfilled / max(total_demand, 1e-8)))
        return service, fill, float(round(total_cost, 2))

    def _eval_rl(
        self, rl_agent: Any, val_df: pd.DataFrame, forecaster: Any | None = None
    ) -> ModelResult:
        """Roll out RL policy using demand series and optional TFT uncertainty."""
        from src.rl.inventory_env import RetailInventoryEnv

        demand = self._sales_series(val_df)

        forecast_horizon = 7
        p10_full: np.ndarray | None = None
        p50_full: np.ndarray | None = None
        p90_full: np.ndarray | None = None

        if forecaster is not None:
            try:
                ref_df = pd.read_parquet(self.reference_parquet)
                from src.models.tft_dataset import build_tft_dataset

                train_ds = build_tft_dataset(ref_df, config_path=str(self.tft_config))
                val_ds = build_tft_dataset(
                    ref_df,
                    config_path=str(self.tft_config),
                    training=False,
                    training_dataset=train_ds,
                )
                loader = val_ds.to_dataloader(train=False, batch_size=128, num_workers=0)
                preds, _ = forecaster.predict(loader, mode="quantiles")
                pred_np = (
                    preds.detach().cpu().numpy() if hasattr(preds, "detach") else np.asarray(preds)
                )
                if pred_np.ndim == 3:
                    pred_np = pred_np.reshape(-1, pred_np.shape[-1])
                q_idx = forecaster.quantile_indices
                p10_full = pred_np[:, q_idx.get("p10", 1)]
                p50_full = pred_np[:, q_idx.get("p50", 3)]
                p90_full = pred_np[:, q_idx.get("p90", 5)]
                env_container = getattr(rl_agent, "env", None)
                env_obj = getattr(env_container, "env", env_container)
                forecast_horizon = max(7, int(getattr(env_obj, "horizon", 7)))
            except Exception as exc:
                logger.warning("Failed to wire TFT quantiles into RL eval: %s", exc)

        def _forecast_quantiles_fn(t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if (
                p50_full is not None
                and p10_full is not None
                and p90_full is not None
                and len(p50_full) > 0
            ):
                idxs = [min(t + i, len(p50_full) - 1) for i in range(forecast_horizon)]
                return p10_full[idxs], p50_full[idxs], p90_full[idxs]
            idxs = [min(t + i, len(demand) - 1) for i in range(forecast_horizon)]
            naive = demand[idxs]
            return naive * 0.85, naive, naive * 1.15

        env = RetailInventoryEnv(
            config_path=str(self.rl_config),
            demand_series=demand,
            forecast_quantiles_fn=_forecast_quantiles_fn,
            store_open_series=(
                val_df["Open"].to_numpy(dtype=float) if "Open" in val_df.columns else None
            ),
        )

        obs, _ = env.reset()
        t_start = time.perf_counter()
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = rl_agent.model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return ModelResult(
            model_name="TFT + PPO (ours)",
            smape=0.0,
            weighted_smape=0.0,
            rmse=0.0,
            mase=0.0,
            coverage=0.0,
            underestimation_rate=0.0,
            service_level=float(env.service_level()),
            fill_rate=float(env.fill_rate()),
            total_cost=float(env.total_cost()),
            latency_ms=float(elapsed_ms / max(steps, 1)),
        )

    def stress_test_stockout_severity(
        self,
        val_df: pd.DataFrame,
        forecaster: Any,
        rl_agent: Any,
        severities: list[float] | None = None,
        output_filename: str = "rl_stress_test.csv",
    ) -> pd.DataFrame:
        """Run RL stress-test across stockout severity settings.

        The test keeps the trained policy fixed and evaluates it under
        progressively harsher shortfall penalties in ``TripleTierReward``.
        Results include action statistics (order volume/dispersion) and
        operational KPIs for paper-ready reporting.
        """
        from src.rl.inventory_env import RetailInventoryEnv

        loaded_forecaster = self._load_tft_forecaster(forecaster)
        loaded_agent = self._load_rl_agent(rl_agent, val_df=val_df)
        if loaded_forecaster is None or loaded_agent is None:
            raise RuntimeError(
                "Both TFT forecaster and RL agent must be available for stress tests."
            )

        if severities is None:
            severities = [round(value, 3) for value in np.linspace(0.01, 0.1, 10)]

        demand = self._sales_series(val_df)
        reference_df = pd.read_parquet(self.reference_parquet)
        from src.models.tft_dataset import build_tft_dataset

        train_ds = build_tft_dataset(reference_df, config_path=str(self.tft_config))
        val_ds = build_tft_dataset(
            reference_df,
            config_path=str(self.tft_config),
            training=False,
            training_dataset=train_ds,
        )
        loader = val_ds.to_dataloader(train=False, batch_size=128, num_workers=0)

        preds, _ = loaded_forecaster.predict(loader, mode="quantiles")
        pred_np = preds.detach().cpu().numpy() if hasattr(preds, "detach") else np.asarray(preds)
        if pred_np.ndim == 3:
            pred_np = pred_np.reshape(-1, pred_np.shape[-1])
        q_idx = loaded_forecaster.quantile_indices
        p10_full = pred_np[:, q_idx.get("p10", 1)]
        p50_full = pred_np[:, q_idx.get("p50", 3)]
        p90_full = pred_np[:, q_idx.get("p90", 5)]

        y_true = []
        for _x_batch, target_tuple in loader:
            target = target_tuple[0]
            target_np = (
                target.detach().cpu().numpy() if hasattr(target, "detach") else np.asarray(target)
            )
            y_true.append(target_np.reshape(-1).astype(float))
        y_true_np = np.concatenate(y_true) if y_true else np.array([], dtype=float)
        aligned = min(len(y_true_np), len(p50_full))
        y_true_np = y_true_np[:aligned]
        p50_eval = p50_full[:aligned]
        asym_smape_ref = (
            float(weighted_smape(y_true_np, p50_eval, under_weight=1.5, over_weight=1.0))
            if aligned > 0
            else 0.0
        )

        base_loss = getattr(getattr(loaded_forecaster, "model", None), "loss", None)
        under_penalty_factor = float(getattr(base_loss, "under_penalty_factor", 1.5))

        horizon = int(getattr(getattr(loaded_agent, "env", None), "env", loaded_agent.env).horizon)

        def _forecast_quantiles_fn(t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if len(p50_full) == 0:
                idxs = [min(t + i, len(demand) - 1) for i in range(horizon)]
                naive = demand[idxs]
                return naive * 0.85, naive, naive * 1.15
            idxs = [min(t + i, len(p50_full) - 1) for i in range(horizon)]
            return p10_full[idxs], p50_full[idxs], p90_full[idxs]

        rows: list[dict[str, float]] = []
        for idx, severity in enumerate(severities):
            reward_fn = TripleTierReward(
                holding_cost=0.5,
                order_cost=10.0,
                stockout_cost=5.0,
                stockout_severity=float(severity),
            )
            env = RetailInventoryEnv(
                config_path=str(self.rl_config),
                demand_series=demand,
                forecast_quantiles_fn=_forecast_quantiles_fn,
                store_open_series=(
                    val_df["Open"].to_numpy(dtype=float) if "Open" in val_df.columns else None
                ),
                reward_fn=reward_fn,
            )

            obs, _ = env.reset()
            orders = []
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = loaded_agent.model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
                orders.append(float(info.get("order_qty", 0.0)))

            orders_np = np.asarray(orders, dtype=float) if orders else np.zeros(1, dtype=float)
            row = {
                "stockout_severity": float(severity),
                "mean_order_qty": float(np.mean(orders_np)),
                "order_qty_std": float(np.std(orders_np)),
                "order_qty_p95": float(np.percentile(orders_np, 95)),
                "order_zero_ratio": float(np.mean(orders_np <= 1e-8)),
                "service_level": float(env.service_level()),
                "fill_rate": float(env.fill_rate()),
                "total_cost": float(env.total_cost()),
                "forecast_weighted_smape": asym_smape_ref,
                "under_penalty_factor": under_penalty_factor,
            }
            rows.append(row)

            if mlflow.active_run() is not None:
                for key, value in row.items():
                    mlflow.log_metric(f"stress_{key}", float(value), step=idx)

        result_df = pd.DataFrame(rows)
        out_path = self.output_dir / output_filename
        result_df.to_csv(out_path, index=False)

        if self.log_to_mlflow and mlflow.active_run() is None:
            with mlflow.start_run(run_name="rl_stress_test"):
                mlflow.log_param("stress_points", len(rows))
                mlflow.log_param("stress_min_severity", float(min(severities)))
                mlflow.log_param("stress_max_severity", float(max(severities)))
                mlflow.log_param("forecast_under_penalty_factor", under_penalty_factor)
                for idx, row in enumerate(rows):
                    for key, value in row.items():
                        mlflow.log_metric(f"stress_{key}", float(value), step=idx)
                mlflow.log_artifact(str(out_path))

        return result_df

    def _log_to_mlflow(self, df: pd.DataFrame, csv_path: Path) -> None:
        """Log compact evaluation metrics and CSV artifact."""
        with mlflow.start_run(run_name="evaluation_harness"):
            for _, row in df.iterrows():
                prefix = (
                    str(row["model_name"])
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("+", "plus")
                )
                mlflow.log_metric(f"{prefix}_service_level", float(row.get("service_level", 0.0)))
                mlflow.log_metric(f"{prefix}_fill_rate", float(row.get("fill_rate", 0.0)))
                mlflow.log_metric(f"{prefix}_total_cost", float(row.get("total_cost", 0.0)))
                if float(row.get("smape", 0.0)) > 0.0:
                    mlflow.log_metric(f"{prefix}_smape", float(row.get("smape", 0.0)))
                    mlflow.log_metric(
                        f"{prefix}_weighted_smape", float(row.get("weighted_smape", 0.0))
                    )
                    mlflow.log_metric(f"{prefix}_rmse", float(row.get("rmse", 0.0)))
                    mlflow.log_metric(f"{prefix}_coverage", float(row.get("coverage", 0.0)))
                    mlflow.log_metric(
                        f"{prefix}_underestimation_rate",
                        float(row.get("underestimation_rate", 0.0)),
                    )
            mlflow.log_artifact(str(csv_path))

    def print_table(self, df: pd.DataFrame) -> None:
        """Print a compact comparison table to stdout."""
        columns = [
            "model_name",
            "smape",
            "weighted_smape",
            "rmse",
            "coverage",
            "underestimation_rate",
            "service_level",
            "fill_rate",
            "total_cost",
            "latency_ms",
        ]
        existing = [column for column in columns if column in df.columns]
        print("\n=== Evaluation Table ===")
        print(df[existing].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
