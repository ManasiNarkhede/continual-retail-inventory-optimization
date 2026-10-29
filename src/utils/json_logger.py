"""Simple JSON logging adapter for structured service logs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import structlog

_STRUCTLOGGER = structlog.get_logger("retail_inventory")


def log_json(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a JSON-formatted log line with common metadata."""
    store_id = fields.pop("store_id", None)
    model_version = fields.pop("model_version", None)
    request_id = fields.pop("request_id", None)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "store_id": store_id,
        "model_version": model_version,
        "request_id": request_id,
        **fields,
    }
    logger.log(level, json.dumps(payload, default=str))
    _STRUCTLOGGER.msg(event, **{key: value for key, value in payload.items() if key != "event"})
