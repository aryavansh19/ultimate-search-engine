"""Adaptive Pacing & Quota Telemetry for upstream API dependencies.

Features:
1. Thread-safe sliding window tracking (60s RPM/TPM, 24h RPD/TPD).
2. Proactive rate smoothing when utilization reaches warning thresholds (>= 75%).
3. Hard pacing / capacity drainage when saturation approaches limits (>= 90%).
4. Cooldown recovery on HTTP 429 (respects Retry-After header + jittered exponential backoff).
5. Comprehensive telemetry snapshots for diagnostics, /health, and benchmark HUDs.
"""

from __future__ import annotations

import collections
import logging
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

log = logging.getLogger("quota.governor")


class QuotaState(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    THROTTLED = "throttled"
    COOLING_DOWN = "cooling_down"


@dataclass(slots=True)
class QuotaTelemetry:
    service: str
    rpm: int
    max_rpm: float
    rpm_utilization_pct: float
    tpm: int
    max_tpm: int
    tpm_utilization_pct: float
    rpd: int
    max_rpd: int
    rpd_utilization_pct: float
    status: str
    cooldown_remaining_s: float
    lifetime_requests: int
    lifetime_tokens: int
    error_counts: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SlidingWindowMeter:
    """Thread-safe sliding window meter for request and token counts."""

    def __init__(self, short_window_s: float = 60.0, long_window_s: float = 86400.0) -> None:
        self.short_window_s = short_window_s
        self.long_window_s = long_window_s
        self._lock = threading.Lock()

        # Entries are tuples: (timestamp_monotonic, timestamp_epoch, tokens)
        self._requests: collections.deque[tuple[float, float, int]] = collections.deque()
        self._lifetime_requests = 0
        self._lifetime_tokens = 0
        self._errors: collections.Counter[int] = collections.Counter()

    def _purge(self, now_mono: float) -> None:
        cutoff = now_mono - self.long_window_s
        while self._requests and self._requests[0][0] < cutoff:
            self._requests.popleft()

    def record_request(self, tokens: int = 0, now_mono: float | None = None) -> None:
        now_m = time.monotonic() if now_mono is None else now_mono
        now_e = time.time()
        with self._lock:
            self._purge(now_m)
            self._requests.append((now_m, now_e, max(0, tokens)))
            self._lifetime_requests += 1
            self._lifetime_tokens += max(0, tokens)

    def record_tokens(self, tokens: int) -> None:
        """Update tokens for the most recent request once response usage is known."""
        if tokens <= 0:
            return
        with self._lock:
            self._lifetime_tokens += tokens
            if self._requests:
                last_mono, last_epoch, prev_tokens = self._requests[-1]
                self._requests[-1] = (last_mono, last_epoch, prev_tokens + tokens)

    def record_error(self, status_code: int) -> None:
        with self._lock:
            self._errors[status_code] += 1

    def metrics(self, now_mono: float | None = None) -> tuple[int, int, int, int]:
        """Returns (rpm, tpm, rpd, tpd) in current sliding windows."""
        now_m = time.monotonic() if now_mono is None else now_mono
        short_cutoff = now_m - self.short_window_s
        long_cutoff = now_m - self.long_window_s

        rpm = 0
        tpm = 0
        rpd = 0
        tpd = 0

        with self._lock:
            self._purge(now_m)
            for m_time, _, tokens in self._requests:
                if m_time >= long_cutoff:
                    rpd += 1
                    tpd += tokens
                if m_time >= short_cutoff:
                    rpm += 1
                    tpm += tokens

        return rpm, tpm, rpd, tpd

    def time_to_free_capacity(self, needed_requests: int = 1, now_mono: float | None = None) -> float:
        """Time in seconds until at least `needed_requests` expire from the 60s window."""
        now_m = time.monotonic() if now_mono is None else now_mono
        short_cutoff = now_m - self.short_window_s

        with self._lock:
            self._purge(now_m)
            # Filter entries within short window
            short_entries = [e for e in self._requests if e[0] >= short_cutoff]
            if len(short_entries) < needed_requests:
                return 0.0

            # Target the entry that needs to expire
            target_idx = needed_requests - 1
            oldest_relevant = short_entries[target_idx]
            expire_at = oldest_relevant[0] + self.short_window_s
            return max(0.0, expire_at - now_m)

    @property
    def lifetime_requests(self) -> int:
        with self._lock:
            return self._lifetime_requests

    @property
    def lifetime_tokens(self) -> int:
        with self._lock:
            return self._lifetime_tokens

    @property
    def error_counts(self) -> dict[int, int]:
        with self._lock:
            return dict(self._errors)


class AdaptivePacer:
    """Calculates pacing intervals, throttle buffers, and backoff delays."""

    def __init__(
        self,
        max_rpm: float = 15.0,
        max_tpm: int = 1_000_000,
        max_rpd: int = 1500,
        warning_threshold: float = 0.75,
        critical_threshold: float = 0.90,
    ) -> None:
        self.max_rpm = max(1.0, max_rpm)
        self.max_tpm = max(1000, max_tpm)
        self.max_rpd = max(1, max_rpd)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self._cooldown_until_mono = 0.0
        self._consecutive_429 = 0
        self._last_request_mono = 0.0
        self._lock = threading.Lock()

    def trigger_backoff(self, retry_after: float | None = None) -> float:
        """Trigger backoff due to HTTP 429. Returns cooldown duration."""
        now_m = time.monotonic()
        with self._lock:
            self._consecutive_429 += 1
            if retry_after and retry_after > 0:
                duration = min(retry_after, 120.0)
            else:
                # Exponential backoff with full jitter
                base = min(15.0 * (2 ** min(self._consecutive_429 - 1, 4)), 90.0)
                jitter = random.uniform(0.5, 2.5)
                duration = base + jitter

            self._cooldown_until_mono = now_m + duration
            return duration

    def record_success(self) -> None:
        with self._lock:
            if self._consecutive_429 > 0:
                self._consecutive_429 = 0

    def calculate_wait(
        self,
        meter: SlidingWindowMeter,
        estimated_tokens: int = 0,
        now_mono: float | None = None,
    ) -> tuple[float, QuotaState, str]:
        """Calculates needed delay, current state, and diagnostic reason."""
        now_m = time.monotonic() if now_mono is None else now_mono

        with self._lock:
            # 1. Check if under cooldown from previous 429
            if now_m < self._cooldown_until_mono:
                wait_time = self._cooldown_until_mono - now_m
                return wait_time, QuotaState.COOLING_DOWN, f"Active 429 cooldown ({wait_time:.1f}s remaining)"

        rpm, tpm, rpd, _ = meter.metrics(now_m)
        rpm_ratio = (rpm + 1) / self.max_rpm
        tpm_ratio = (tpm + estimated_tokens) / self.max_tpm
        rpd_ratio = (rpd + 1) / self.max_rpd

        # 2. Hard capacity saturation (100%+): must wait for window to clear
        if rpm_ratio >= 1.0:
            drain_wait = meter.time_to_free_capacity(needed_requests=1, now_mono=now_m) + 0.1
            return (
                drain_wait,
                QuotaState.THROTTLED,
                f"RPM limit reached ({rpm}/{int(self.max_rpm)}). Draining window for {drain_wait:.2f}s",
            )

        if tpm_ratio >= 1.0:
            drain_wait = meter.time_to_free_capacity(needed_requests=1, now_mono=now_m) + 0.1
            return (
                drain_wait,
                QuotaState.THROTTLED,
                f"TPM limit reached ({tpm}/{self.max_tpm}). Draining window for {drain_wait:.2f}s",
            )

        if rpd_ratio >= 1.0:
            return (
                60.0,
                QuotaState.THROTTLED,
                f"Daily RPD limit reached ({rpd}/{self.max_rpd}). Exhausted.",
            )

        # 3. Critical threshold (>= 90%): Proactive smooth pacing
        if rpm_ratio >= self.critical_threshold or tpm_ratio >= self.critical_threshold:
            # Enforce strict pacing interval
            target_interval = 60.0 / (self.max_rpm * 0.85)
            with self._lock:
                elapsed = now_m - self._last_request_mono
                wait_time = max(0.0, target_interval - elapsed)
            return (
                wait_time,
                QuotaState.THROTTLED,
                f"Critical saturation ({max(rpm_ratio, tpm_ratio)*100:.1f}%). Pacing interval {target_interval:.2f}s",
            )

        # 4. Warning threshold (>= 75%): Mild proactive pacing
        if rpm_ratio >= self.warning_threshold or tpm_ratio >= self.warning_threshold:
            target_interval = 60.0 / self.max_rpm
            with self._lock:
                elapsed = now_m - self._last_request_mono
                wait_time = max(0.0, target_interval - elapsed)
            return (
                wait_time,
                QuotaState.WARNING,
                f"Approaching capacity ({max(rpm_ratio, tpm_ratio)*100:.1f}%). Pacing interval {target_interval:.2f}s",
            )

        # 5. Healthy: normal execution
        return 0.0, QuotaState.HEALTHY, "Within capacity limits"

    def mark_request_dispatched(self, now_mono: float | None = None) -> None:
        now_m = time.monotonic() if now_mono is None else now_mono
        with self._lock:
            self._last_request_mono = now_m


class QuotaGovernor:
    """Coordinates sliding window metering, adaptive pacing, and telemetry."""

    def __init__(
        self,
        service_name: str,
        max_rpm: float,
        max_tpm: int,
        max_rpd: int,
        enabled: bool = True,
    ) -> None:
        self.service_name = service_name
        self.meter = SlidingWindowMeter()
        self.pacer = AdaptivePacer(max_rpm=max_rpm, max_tpm=max_tpm, max_rpd=max_rpd)
        self.enabled = enabled
        self._acquire_lock = threading.Lock()

    def acquire(self, estimated_tokens: int = 0) -> float:
        """Proactively checks limits and sleeps if needed before dispatching request.

        Returns total time waited in seconds.
        """
        if not self.enabled:
            self.meter.record_request(tokens=estimated_tokens)
            return 0.0

        with self._acquire_lock:
            now_m = time.monotonic()
            wait_time, state, reason = self.pacer.calculate_wait(
                self.meter, estimated_tokens=estimated_tokens, now_mono=now_m
            )

            if state in (QuotaState.WARNING, QuotaState.THROTTLED, QuotaState.COOLING_DOWN):
                log.warning(
                    "[quota:%s] %s state=%s wait=%.2fs",
                    self.service_name,
                    reason,
                    state.value,
                    wait_time,
                )

            total_waited = 0.0
            while wait_time > 0:
                time.sleep(wait_time)
                total_waited += wait_time
                now_m = time.monotonic()
                wait_time, state, reason = self.pacer.calculate_wait(
                    self.meter, estimated_tokens=estimated_tokens, now_mono=now_m
                )

            self.meter.record_request(tokens=estimated_tokens, now_mono=time.monotonic())
            self.pacer.mark_request_dispatched(now_mono=time.monotonic())
            return total_waited

    def record_success(self, actual_tokens: int = 0) -> None:
        """Call after receiving an HTTP 200 response to update usage and clear backoffs."""
        self.meter.record_tokens(actual_tokens)
        self.pacer.record_success()

    def record_error(
        self,
        status_code: int,
        retry_after: float | None = None,
        error_body: str = "",
    ) -> float:
        """Call on HTTP error. Triggers adaptive backoff if status_code == 429."""
        self.meter.record_error(status_code)
        if status_code == 429:
            cooldown = self.pacer.trigger_backoff(retry_after=retry_after)
            log.error(
                "[quota:%s] HTTP 429 Too Many Requests! Initiating adaptive cooldown for %.1fs. body=%s",
                self.service_name,
                cooldown,
                error_body[:200],
            )
            return cooldown
        return 0.0

    def telemetry(self) -> QuotaTelemetry:
        """Returns a snapshot of real-time quota telemetry."""
        now_m = time.monotonic()
        rpm, tpm, rpd, _ = self.meter.metrics(now_m)
        _, state, _ = self.pacer.calculate_wait(self.meter, estimated_tokens=0, now_mono=now_m)

        cooldown_remaining = max(0.0, self.pacer._cooldown_until_mono - now_m)

        return QuotaTelemetry(
            service=self.service_name,
            rpm=rpm,
            max_rpm=self.pacer.max_rpm,
            rpm_utilization_pct=round((rpm / self.pacer.max_rpm) * 100, 1),
            tpm=tpm,
            max_tpm=self.pacer.max_tpm,
            tpm_utilization_pct=round((tpm / self.pacer.max_tpm) * 100, 2),
            rpd=rpd,
            max_rpd=self.pacer.max_rpd,
            rpd_utilization_pct=round((rpd / self.pacer.max_rpd) * 100, 2),
            status=state.value,
            cooldown_remaining_s=round(cooldown_remaining, 2),
            lifetime_requests=self.meter.lifetime_requests,
            lifetime_tokens=self.meter.lifetime_tokens,
            error_counts=self.meter.error_counts,
        )


# ---------------------------------------------------------------------- Singletons
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return default


_pacing_enabled = os.getenv("QUOTA_PACING_ENABLED", "1").strip().lower() not in ("0", "false", "no")

# Gemini free tier default: 15 RPM, 1,000,000 TPM, 1,500 RPD
# (We default max_rpm to 14.0 for a tiny safety margin against network clock drift)
gemini_governor = QuotaGovernor(
    service_name="gemini",
    max_rpm=_env_float("GEMINI_MAX_RPM", 14.0),
    max_tpm=_env_int("GEMINI_MAX_TPM", 1_000_000),
    max_rpd=_env_int("GEMINI_MAX_RPD", 1500),
    enabled=_pacing_enabled,
)

# Embedder provider: 30 RPM default, 500,000 TPM
embedder_governor = QuotaGovernor(
    service_name="embedder",
    max_rpm=_env_float("SEARCH_EMBED_RPM", _env_float("EMBED_MAX_RPM", 30.0)),
    max_tpm=_env_int("EMBED_MAX_TPM", 500_000),
    max_rpd=_env_int("EMBED_MAX_RPD", 10_000),
    enabled=_pacing_enabled,
)

# Scraper endpoints: 60 RPM default
scraper_governor = QuotaGovernor(
    service_name="scraper",
    max_rpm=_env_float("SCRAPER_MAX_RPM", 60.0),
    max_tpm=_env_int("SCRAPER_MAX_TPM", 2_000_000),
    max_rpd=_env_int("SCRAPER_MAX_RPD", 20_000),
    enabled=_pacing_enabled,
)


def get_all_telemetry() -> dict[str, dict[str, Any]]:
    """Return dictionary of all active governor telemetries."""
    return {
        "gemini": gemini_governor.telemetry().to_dict(),
        "embedder": embedder_governor.telemetry().to_dict(),
        "scraper": scraper_governor.telemetry().to_dict(),
    }
