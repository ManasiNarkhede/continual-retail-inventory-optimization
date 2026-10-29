"""Alert endpoints for stock and model-health monitoring."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.model_registry import get_cl_metrics, get_tft_model, get_tft_quantile_indices

router = APIRouter()

AlertType = Literal["STOCKOUT_RISK", "HIGH_COST", "LOW_SERVICE", "CL_DEGRADATION"]
Severity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class Alert(BaseModel):
    """Alert payload model."""

    alert_id: str
    alert_type: AlertType
    store_id: int | None
    severity: Severity
    message: str
    created_at: str
    resolved: bool = False


def _check_stockout_risk(
    store_id: int, current_stock: float, forecast: list[float]
) -> Alert | None:
    """Detect near-term stockout risk from lead-time demand."""
    lead_time = 2
    projected = float(sum(forecast[:lead_time]))
    if projected <= current_stock * 0.9:
        return None
    deficit = max(0.0, projected - current_stock)
    severity: Severity = "HIGH" if deficit > current_stock * 0.5 else "MEDIUM"
    if deficit > 0:
        message = f"Store {store_id}: projected deficit {deficit:.0f} units in {lead_time} days."
    else:
        message = (
            f"Store {store_id}: projected demand {projected:.0f} units in {lead_time} days "
            f"uses over 90% of current stock ({current_stock:.0f} units)."
        )
    return Alert(
        alert_id=f"STOCKOUT-{store_id}-{datetime.utcnow().strftime('%Y%m%d%H%M')}",
        alert_type="STOCKOUT_RISK",
        store_id=int(store_id),
        severity=severity,
        message=message,
        created_at=datetime.utcnow().isoformat(),
    )


@router.get("/active")
def get_active_alerts() -> list[Alert]:
    """Return currently active system alerts."""
    alerts: list[Alert] = []
    cl = get_cl_metrics()
    bwt_value = float(cl.get("bwt", 0.0)) if cl else 0.0
    intransigence_value = float(cl.get("intransigence_mean", 0.0)) if cl else 0.0
    if bwt_value < -0.10:
        alerts.append(
            Alert(
                alert_id="CL-DEGRADATION-001",
                alert_type="CL_DEGRADATION",
                store_id=None,
                severity="HIGH",
                message=(
                    f"Model forgetting detected: BWT={bwt_value:.3f} (threshold -0.10). "
                    "Re-run CL training."
                ),
                created_at=datetime.utcnow().isoformat(),
            )
        )
    if intransigence_value > 0.25:
        alerts.append(
            Alert(
                alert_id="CL-INTRANSIGENCE-CRITICAL",
                alert_type="CL_DEGRADATION",
                store_id=None,
                severity="CRITICAL",
                message=(
                    f"Critical intransigence detected: I_k={intransigence_value:.3f} "
                    "(threshold 0.25). Model is failing to adapt to new distribution."
                ),
                created_at=datetime.utcnow().isoformat(),
            )
        )
    return alerts


@router.post("/check/{store_id}")
def check_store_alerts(store_id: int, current_stock: float = 100.0) -> list[Alert]:
    """Evaluate alerts for a single store."""
    model, dataset, reference_df = get_tft_model()
    if model is None or dataset is None or reference_df is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "TFT model not loaded. Ensure artifacts exist: "
                "checkpoints/tft/tft_latest.pt and data/processed/tft_reference.parquet"
            ),
        )

    try:
        predictions, _dates = model.predict_for_store(
            store_id=store_id,
            horizon=7,
            training_dataset=dataset,
            reference_df=reference_df,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Forecast generation failed: {exc}") from exc

    q_idx = get_tft_quantile_indices()
    idx_p50 = q_idx.get("p50", 3)
    forecast_tensor = predictions[:, idx_p50]
    if hasattr(forecast_tensor, "detach"):
        forecast = forecast_tensor.detach().cpu().numpy().tolist()
    else:
        forecast = [float(value) for value in forecast_tensor]

    alerts: list[Alert] = []
    stockout = _check_stockout_risk(
        store_id=store_id, current_stock=current_stock, forecast=forecast
    )
    if stockout is not None:
        alerts.append(stockout)
    return alerts


@router.get("/rules")
def get_alert_rules() -> list[dict[str, str]]:
    """Return alert-rule definitions used by the service."""
    return [
        {
            "type": "STOCKOUT_RISK",
            "condition": "sum(forecast[:lead_time]) > current_stock * 0.9",
        },
        {
            "type": "CL_DEGRADATION",
            "condition": "bwt < -0.10",
        },
    ]
