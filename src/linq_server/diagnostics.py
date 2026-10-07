"""Structured, secret-safe diagnostics for the mobile enrichment pipeline."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .models import AnalyzeRequest


def configure_logging() -> int:
    """Attach LinQ diagnostics to Uvicorn's configured Render log stream."""
    level_name = (os.getenv("LINQ_LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    linq_logger = logging.getLogger("linq")
    uvicorn_logger = logging.getLogger("uvicorn.error")

    # Render starts the app with the uvicorn executable, so basicConfig() is not
    # used by linq_server.__main__. Reuse Uvicorn's handlers to guarantee that
    # INFO lifecycle and payload lines reach stdout and the Render Logs page.
    if uvicorn_logger.handlers:
        linq_logger.handlers = list(uvicorn_logger.handlers)
        linq_logger.propagate = False
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        linq_logger.propagate = True
    linq_logger.setLevel(level)
    return level


def payload_logging_enabled() -> bool:
    return (os.getenv("LINQ_LOG_PAYLOADS", "0").strip().lower()
            in {"1", "true", "yes", "on"})


def analyze_request_payload(request: AnalyzeRequest) -> dict[str, Any]:
    """Return useful request data without dumping a potentially huge rendered DOM."""
    payload = request.model_dump(mode="json", exclude={"html"})
    payload["html_chars"] = len(request.html or "")
    return payload


def log_json(
    logger: logging.Logger,
    trace_id: str,
    label: str,
    payload: Any,
) -> None:
    if not payload_logging_enabled():
        return
    try:
        formatted = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError) as exc:
        logger.warning("[search-pipeline:%s] could not format %s: %s", trace_id, label, exc)
        return
    logger.info("[search-pipeline:%s] %s:\n%s", trace_id, label, formatted)
