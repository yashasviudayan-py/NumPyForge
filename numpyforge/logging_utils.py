"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
from typing import Any


def configure_json_logging(level: int = logging.INFO) -> None:
    """Configure root logging for compact JSON records."""
    logging.basicConfig(level=level, format="%(message)s")


def log_json(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a single structured JSON log event."""
    payload = {"event": event, **fields}
    logger.log(level, json.dumps(payload, sort_keys=True))
