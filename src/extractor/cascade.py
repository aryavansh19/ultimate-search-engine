"""The fallback cascade: run tiers in order until one produces usable content.

    client payload -> yt-dlp -> managed API -> OpenGraph/HTML -> degraded

Three properties this is built to guarantee:

* No single tier's failure is fatal. Every attempt is wrapped, timed and recorded.
  A tier that raises, hangs or returns junk costs you the next tier's latency and
  nothing else.
* Nothing is ever dropped. If every tier fails, you still get an envelope carrying
  the canonical URL and platform, flagged `degraded` and `reprocessable`, so the item
  stays in the system and can be retried later when a better tier exists. Losing a
  user's saved link because Instagram was hostile that afternoon is not acceptable.
* Partial results accumulate. A tier that returns a thumbnail but no text does not
  win outright -- it is held and merged into whatever the next tier finds. The
  winning tier is the one that contributed actual text.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Sequence

from .base import Extractor, ExtractorUnavailable
from .cache import EnvelopeCache
from .config import ExtractorConfig
from .envelope import (
    ContentEnvelope,
    ExtractionAttempt,
    ExtractionResult,
    ExtractionTarget,
    ExtractionTier,
    SignalStrength,
)
from .extractors import DEFAULT_EXTRACTORS
from .urls import canonicalize, detect_platform, url_hash

log = logging.getLogger("extractor.cascade")


class TierTimeout(Exception):
    """A tier exceeded its wall-clock ceiling."""


class ExtractionCascade:
    def __init__(
        self,
        config: ExtractorConfig | None = None,
        cache: EnvelopeCache | None = None,
        extractors: Sequence[Extractor] | None = None,
    ) -> None:
        self.config = config or ExtractorConfig.from_env()
        self.cache = cache if cache is not None else EnvelopeCache(self.config.cache_path)
        built = (
            list(extractors)
            if extractors is not None
            else [cls(self.config) for cls in DEFAULT_EXTRACTORS]
        )
        self.extractors = sorted(
            (e for e in built if e.tier not in self.config.disabled_tiers),
            key=lambda e: e.tier.rank,
        )

    # --------------------------------------------------------------- public API
    def extract(
        self,
        url: str,
        *,
        client_payload: dict[str, Any] | None = None,
        use_cache: bool = True,
        force: bool = False,
        retry_degraded: bool = False,
        follow_shorteners: bool = True,
    ) -> ExtractionResult:
        canonical = canonicalize(url, follow_shorteners=follow_shorteners)
        target = ExtractionTarget(
            canonical_url=canonical,
            url_hash=url_hash(canonical),
            platform=detect_platform(canonical),
            original_url=url,
            client_payload=client_payload,
        )

        if use_cache and not force:
            cached = self.cache.get(target.url_hash)
            if cached is not None and not (cached.degraded and retry_degraded):
                log.debug("cache hit %s", target.canonical_url)
                return ExtractionResult(envelope=cached, cache_hit=True)

        result = self._run_cascade(target)

        if use_cache:
            self.cache.put(result.envelope)
            self.cache.record_attempts(
                target.url_hash, target.platform.value, result.attempts
            )
        return result

    # ------------------------------------------------------------------ internals
    def _run_cascade(self, target: ExtractionTarget) -> ExtractionResult:
        attempts: list[ExtractionAttempt] = []
        best: ContentEnvelope | None = None

        for extractor in self.extractors:
            available, reason = self._availability(extractor)
            if not available:
                attempts.append(
                    ExtractionAttempt(
                        tier=extractor.tier, ok=False, skipped=True, reason=reason
                    )
                )
                continue

            if not extractor.can_handle(target):
                attempts.append(
                    ExtractionAttempt(
                        tier=extractor.tier,
                        ok=False,
                        skipped=True,
                        reason="not applicable to this target",
                    )
                )
                continue

            started = time.perf_counter()
            envelope: ContentEnvelope | None = None
            error: str | None = None
            try:
                envelope = self._run_with_timeout(extractor, target)
            except TierTimeout:
                error = f"timed out after {self.config.tier_timeout:.0f}s"
            except ExtractorUnavailable as exc:
                error = f"unavailable: {exc}"
            except Exception as exc:  # noqa: BLE001 - never fatal, by design
                error = f"{type(exc).__name__}: {exc}"
                log.debug("tier %s failed on %s", extractor.tier.value, target.canonical_url,
                          exc_info=True)
            duration_ms = int((time.perf_counter() - started) * 1000)

            if envelope is None:
                attempts.append(
                    ExtractionAttempt(
                        tier=extractor.tier,
                        ok=False,
                        reason=error or "no usable content",
                        duration_ms=duration_ms,
                    )
                )
                continue

            partial = envelope.signal is SignalStrength.NONE
            attempts.append(
                ExtractionAttempt(
                    tier=extractor.tier,
                    ok=True,
                    reason="partial: media only, no text" if partial else None,
                    duration_ms=duration_ms,
                )
            )

            best = envelope if best is None else envelope.merged_with(best)
            if not partial:
                break

        if best is None:
            return ExtractionResult(
                envelope=self._degraded(target), attempts=attempts
            )

        if best.signal is SignalStrength.NONE:
            # Everything ran and nobody found text. Keep the media we did find, but
            # flag it so a re-processing pass can pick it up later.
            best.degraded = True
            best.extractor_note = (
                (best.extractor_note or "") + " (no text signal)"
            ).strip()
        else:
            best.degraded = False
        return ExtractionResult(envelope=best, attempts=attempts)

    def _availability(self, extractor: Extractor) -> tuple[bool, str | None]:
        try:
            return extractor.availability()
        except Exception as exc:  # noqa: BLE001
            return False, f"availability check failed: {exc}"

    def _run_with_timeout(
        self, extractor: Extractor, target: ExtractionTarget
    ) -> ContentEnvelope | None:
        """Run a tier under a wall-clock ceiling.

        Honest limitation: a timed-out tier is abandoned, not killed. yt-dlp offers no
        cooperative cancellation, so the worker is a daemon thread -- the cascade stops
        waiting and moves on, the orphan finishes in the background and its result is
        discarded, and it cannot hold up interpreter shutdown.

        A ThreadPoolExecutor would be wrong here: its context manager joins on exit,
        which would silently reinstate the full blocking wait the timeout exists to
        avoid.
        """
        outcome: list[ContentEnvelope | None] = []
        failure: list[BaseException] = []

        def work() -> None:
            try:
                outcome.append(extractor.extract(target))
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
                failure.append(exc)

        thread = threading.Thread(
            target=work, daemon=True, name=f"tier-{extractor.tier.value}"
        )
        thread.start()
        thread.join(self.config.tier_timeout)

        if thread.is_alive():
            raise TierTimeout(extractor.tier.value)
        if failure:
            raise failure[0]
        return outcome[0] if outcome else None

    def _degraded(self, target: ExtractionTarget) -> ContentEnvelope:
        envelope = target.envelope(ExtractionTier.DEGRADED)
        envelope.degraded = True
        envelope.reprocessable = True
        envelope.extractor_note = "all tiers failed; URL retained for reprocessing"
        return envelope

    def close(self) -> None:
        self.cache.close()
