"""Adaptive Pacing & Quota Telemetry for upstream API dependencies.

Guards external providers (Gemini, Embedder, Scraper endpoints) against:
- Per-minute rate limits (RPM / TPM 429 Too Many Requests)
- Daily allocation ceilings (RPD / TPD)
- Burst exhaustion and concurrency saturation

Exposes thread-safe sliding window telemetry and adaptive backoff pacing.
"""

from __future__ import annotations

from .governor import (
    AdaptivePacer,
    QuotaGovernor,
    QuotaState,
    QuotaTelemetry,
    SlidingWindowMeter,
    embedder_governor,
    gemini_governor,
    get_all_telemetry,
    scraper_governor,
)

__all__ = [
    "AdaptivePacer",
    "QuotaGovernor",
    "QuotaState",
    "QuotaTelemetry",
    "SlidingWindowMeter",
    "embedder_governor",
    "gemini_governor",
    "get_all_telemetry",
    "scraper_governor",
]
