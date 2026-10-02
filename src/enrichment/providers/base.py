"""Provider interface for the tagging call.

Same shape as the extractor's tier interface, and for the same reason: swapping
Gemini for OpenAI, or for Apple's on-device Foundation Models framework once this
moves to iOS, should be one class rather than a refactor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from extractor import ContentEnvelope

from ..config import EnrichmentConfig
from ..schema import Enrichment, EnrichmentMode


class ProviderError(Exception):
    """The provider could not produce an enrichment. Never fatal to the pipeline."""


class ProviderUnavailable(ProviderError):
    """Provider cannot run at all -- missing key, missing dependency."""


class EnrichmentProvider(ABC):
    name: str = "provider"

    def __init__(self, config: EnrichmentConfig) -> None:
        self.config = config

    def availability(self) -> tuple[bool, str | None]:
        return True, None

    def supports(self, mode: EnrichmentMode) -> bool:
        return True

    @abstractmethod
    def enrich(
        self, envelope: ContentEnvelope, mode: EnrichmentMode, document: str
    ) -> Enrichment:
        """Produce an enrichment, or raise ProviderError."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name}>"
