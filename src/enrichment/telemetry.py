"""Non-blocking Supabase ai_api_telemetry reporter for search enrichment Gemini calls."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import httpx

log = logging.getLogger("linq.telemetry")


def mask_api_key(api_key: str | None) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 14:
        return api_key[:4] + "..."
    return f"{api_key[:10]}...{api_key[-4:]}"


def _supabase_telemetry_config() -> tuple[str, str] | None:
    url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not key:
        return None
    return url, key


def _post_telemetry_row(url: str, service_key: str, row: dict[str, Any]) -> None:
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{url}/rest/v1/ai_api_telemetry",
                headers={
                    "apikey": service_key,
                    "Authorization": f"Bearer {service_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json=row,
            )
            if resp.status_code >= 400:
                log.debug("ai_api_telemetry insert returned %s: %s", resp.status_code, resp.text[:120])
    except Exception as exc:
        log.debug("ai_api_telemetry insert skipped: %s", exc)


def record_ai_telemetry(
    *,
    model: str,
    api_key: str | None,
    status_code: int,
    duration_ms: int,
    rpm_in_window: int,
    prompt_tokens: int = 0,
    candidates_tokens: int = 0,
    total_tokens: int = 0,
    error_message: str | None = None,
    is_fallback: bool = False,
    fallback_reason: str | None = None,
    model_switched: bool = False,
    stage: str | None = None,
) -> None:
    """Fire-and-forget insert into public.ai_api_telemetry without blocking enrichment."""
    cfg = _supabase_telemetry_config()
    if not cfg:
        return
    url, service_key = cfg
    row: dict[str, Any] = {
        "feature": "search_enrichment",
        "model": model,
        "prompt_tokens": max(0, int(prompt_tokens)),
        "candidates_tokens": max(0, int(candidates_tokens)),
        "total_tokens": max(0, int(total_tokens)),
        "duration_ms": max(0, int(duration_ms)),
        "status_code": int(status_code),
        "masked_key": mask_api_key(api_key),
        "error_message": error_message[:500] if error_message else None,
        "rpm_in_window": max(1, int(rpm_in_window)),
        "is_fallback": bool(is_fallback),
        "fallback_reason": fallback_reason[:200] if fallback_reason else None,
        "model_switched": bool(model_switched),
        "stage": stage,
    }
    threading.Thread(
        target=_post_telemetry_row,
        args=(url, service_key, row),
        daemon=True,
    ).start()
