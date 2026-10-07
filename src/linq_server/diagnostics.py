"""Structured, secret-safe diagnostics for the mobile enrichment pipeline."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .models import AnalyzeRequest


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
