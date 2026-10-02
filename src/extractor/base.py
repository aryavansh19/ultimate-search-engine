"""The Extractor interface every tier implements.

Adding a new source of content -- a browser extension, a platform's official API,
a stealth browser -- means writing one class here and appending it to the cascade.
Nothing else in the system changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .config import ExtractorConfig
from .envelope import ContentEnvelope, ExtractionTarget, ExtractionTier


class ExtractorUnavailable(Exception):
    """Raised when a tier cannot run at all (missing config, missing dependency).

    Distinct from an extraction failure on purpose: unavailable tiers are recorded
    as *skipped*, so an unconfigured optional provider never pollutes your success
    rate metrics.
    """


class Extractor(ABC):
    """One strategy for turning a URL into a `ContentEnvelope`."""

    tier: ExtractionTier
    name: str = "extractor"

    def __init__(self, config: ExtractorConfig) -> None:
        self.config = config

    def availability(self) -> tuple[bool, str | None]:
        """Whether this tier can run in the current environment.

        Returns (available, reason_if_not). Checked once per attempt, before
        `can_handle`, and reported as a skip rather than a failure.
        """
        return True, None

    def can_handle(self, target: ExtractionTarget) -> bool:
        """Whether this tier is applicable to this particular target."""
        return True

    @abstractmethod
    def extract(self, target: ExtractionTarget) -> ContentEnvelope | None:
        """Attempt extraction.

        Return an envelope on success, or None if this tier found nothing usable.
        Raise on hard failure -- the cascade catches it, records it, and moves on.
        Never let an exception here be fatal to the pipeline.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} tier={self.tier.value}>"
