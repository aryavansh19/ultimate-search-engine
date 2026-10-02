"""The enrichment orchestrator: decide the mode, spend accordingly, never fail hard.

Mirrors the extractor's cascade deliberately. The escalation rule is the whole reason
this stage exists as its own module:

    strong text signal  ->  cheap text model         (~0.02 cents)
    weak / no signal    ->  multimodal media model   (~0.6 cents)
    provider failure    ->  offline heuristic        (free, flagged degraded)

The ratio between the first two is roughly 25x, which is why the gate matters more
than the model choice. Most saved links already describe themselves -- a caption and
five hashtags are usually enough -- and paying to watch a video that came with a
paragraph explaining itself is the easiest money to waste in this whole system.

Nothing here raises on a provider failure. A link that cannot be tagged well still
gets tagged badly, stays searchable, and is marked for a later re-run.
"""

from __future__ import annotations

import logging
from typing import Sequence

from extractor import ContentEnvelope, EnvelopeCache, ExtractionCascade, SignalStrength
from extractor.envelope import MediaKind, content_hash

from .config import EnrichmentConfig
from .prompts import PROMPT_VERSION, trim_document
from .providers import (
    EnrichmentProvider,
    GeminiProvider,
    HeuristicProvider,
    OpenAICompatProvider,
    ProviderError,
)
from .schema import Enrichment, EnrichmentMode
from .store import EnrichmentStore

log = logging.getLogger("enrichment.enricher")

# Name -> constructor. `ENRICH_PROVIDER` selects and orders these.
PROVIDER_REGISTRY: dict[str, type[EnrichmentProvider]] = {
    "gemini": GeminiProvider,
    "ashna": OpenAICompatProvider,
    "compat": OpenAICompatProvider,
    "openai": OpenAICompatProvider,
}


def build_providers(config: EnrichmentConfig) -> list[EnrichmentProvider]:
    """Instantiate providers in the configured order, always ending in the fallback.

    The heuristic provider is appended unconditionally rather than being selectable. It
    costs nothing, needs no network, and its only job is guaranteeing that a saved link is
    never left completely untagged -- which is not something worth letting configuration
    switch off by accident.
    """
    providers: list[EnrichmentProvider] = []
    seen: set[type[EnrichmentProvider]] = set()

    for name in config.provider_order:
        cls = PROVIDER_REGISTRY.get(name)
        if cls is None:
            log.warning(
                "unknown enrichment provider %r; known: %s",
                name,
                ", ".join(sorted(PROVIDER_REGISTRY)),
            )
            continue
        if cls in seen:
            continue
        seen.add(cls)
        providers.append(cls(config))

    if not providers:
        providers.append(GeminiProvider(config))
    providers.append(HeuristicProvider(config))
    return providers


class Enricher:
    def __init__(
        self,
        config: EnrichmentConfig | None = None,
        store: EnrichmentStore | None = None,
        providers: Sequence[EnrichmentProvider] | None = None,
    ) -> None:
        self.config = config or EnrichmentConfig.from_env()
        self.store = store if store is not None else EnrichmentStore(self.config.store_path)
        if providers is not None:
            self.providers = list(providers)
        else:
            self.providers = build_providers(self.config)

    # ------------------------------------------------------------------ public API
    def decide_mode(self, envelope: ContentEnvelope) -> EnrichmentMode:
        """Cheapest mode that can plausibly describe this item."""
        if not self.config.allow_media:
            return EnrichmentMode.METADATA
        if envelope.signal is SignalStrength.STRONG:
            return EnrichmentMode.METADATA
        if not envelope.needs_media_understanding:
            return EnrichmentMode.METADATA

        has_media = bool(envelope.media_url) or envelope.media_kind is MediaKind.VIDEO
        if not has_media:
            return EnrichmentMode.METADATA

        duration = envelope.duration_s or 0.0
        if duration > self.config.max_media_duration_s:
            # A feature-length video is not worth a full pass for tagging purposes.
            return EnrichmentMode.METADATA
        return EnrichmentMode.MEDIA

    def enrich(
        self,
        envelope: ContentEnvelope,
        *,
        force: bool = False,
        mode: EnrichmentMode | None = None,
        use_store: bool = True,
    ) -> Enrichment:
        source = source_hash(envelope)

        if use_store and not force:
            if self.store.is_current(envelope.url_hash, source, PROMPT_VERSION):
                cached = self.store.get(envelope.url_hash)
                if cached is not None:
                    log.debug("enrichment cache hit %s", envelope.canonical_url)
                    return cached

        chosen = mode or self.decide_mode(envelope)
        chosen, budget_note = self._apply_budget(envelope, chosen)
        document = trim_document(
            self._document(envelope), self.config.max_text_chars
        )

        enrichment, notes = self._run_providers(envelope, chosen, document)
        enrichment.source_hash = source
        if budget_note:
            notes.insert(0, budget_note)
        if notes:
            enrichment.note = "; ".join(
                part for part in [enrichment.note, *notes] if part
            )

        if use_store:
            self.store.put(enrichment)
        return enrichment

    def enrich_url(
        self,
        url: str,
        *,
        cascade: ExtractionCascade | None = None,
        force: bool = False,
        mode: EnrichmentMode | None = None,
    ) -> tuple[ContentEnvelope, Enrichment]:
        """Extract then enrich, for the common single-link path."""
        owned = cascade is None
        runner = cascade or ExtractionCascade()
        try:
            result = runner.extract(url)
        finally:
            if owned:
                runner.close()
        return result.envelope, self.enrich(result.envelope, force=force, mode=mode)

    def enrich_pending(
        self,
        limit: int = 50,
        cache: EnvelopeCache | None = None,
        *,
        upgrade_degraded: bool = True,
    ) -> list[Enrichment]:
        """Backfill: enrich cached envelopes that have no current, non-degraded result.

        `upgrade_degraded` is on by default so that items tagged by the offline fallback
        -- everything processed before an API key was available -- get re-run properly
        rather than sitting at fallback quality forever.
        """
        owned = cache is None
        store = cache or EnvelopeCache(self.config.store_path)
        try:
            envelopes = _all_envelopes(store, limit * 4)
        finally:
            if owned:
                store.close()

        degraded = self.store.degraded_hashes() if upgrade_degraded else set()

        out: list[Enrichment] = []
        for envelope in envelopes:
            already_current = self.store.is_current(
                envelope.url_hash, source_hash(envelope), PROMPT_VERSION
            )
            if already_current and envelope.url_hash not in degraded:
                continue
            out.append(self.enrich(envelope, force=envelope.url_hash in degraded))
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------- internals
    def _apply_budget(
        self, envelope: ContentEnvelope, mode: EnrichmentMode
    ) -> tuple[EnrichmentMode, str | None]:
        """Downgrade a media call that is predicted to blow the per-item ceiling.

        Predicting rather than discovering matters: token counts for video scale with
        duration, so the expensive case is knowable in advance and there is no reason
        to find out by being billed for it.
        """
        if mode is not EnrichmentMode.MEDIA:
            return mode, None
        gemini = next(
            (p for p in self.providers if isinstance(p, GeminiProvider)), None
        )
        if gemini is None:
            return mode, None
        predicted = gemini.preflight_cost(envelope)
        if predicted > self.config.max_cost_per_item_usd:
            return (
                EnrichmentMode.METADATA,
                f"media skipped: predicted ${predicted:.4f} exceeds ceiling "
                f"${self.config.max_cost_per_item_usd:.4f}",
            )
        return mode, None

    def _run_providers(
        self, envelope: ContentEnvelope, mode: EnrichmentMode, document: str
    ) -> tuple[Enrichment, list[str]]:
        """Try the requested mode, then text, then the offline fallback."""
        notes: list[str] = []
        attempts: list[tuple[EnrichmentMode, str]] = []
        if mode is EnrichmentMode.MEDIA:
            attempts.append((EnrichmentMode.MEDIA, "media"))
        attempts.append((EnrichmentMode.METADATA, "metadata"))

        for attempt_mode, label in attempts:
            for provider in self.providers:
                available, reason = provider.availability()
                if not available:
                    note = f"{provider.name} unavailable ({reason})"
                    if note not in notes:
                        notes.append(note)
                    continue
                if not provider.supports(attempt_mode):
                    continue
                try:
                    return provider.enrich(envelope, attempt_mode, document), notes
                except ProviderError as exc:
                    notes.append(f"{provider.name} {label} failed: {exc}")
                    log.debug(
                        "provider %s failed on %s", provider.name, envelope.canonical_url,
                        exc_info=True,
                    )
                except Exception as exc:  # noqa: BLE001 - never fatal
                    notes.append(f"{provider.name} {label} error: {type(exc).__name__}")
                    log.debug("provider %s crashed", provider.name, exc_info=True)

        # Last resort: the heuristic provider, constructed directly in case it was not
        # in the configured list at all.
        fallback = HeuristicProvider(self.config)
        enrichment = fallback.enrich(envelope, EnrichmentMode.HEURISTIC, document)
        notes.append("fell back to offline keyword extraction")
        return enrichment, notes

    def _document(self, envelope: ContentEnvelope) -> str:
        return envelope.search_document()

    def close(self) -> None:
        self.store.close()


# ----------------------------------------------------------------------- helpers
def source_hash(envelope: ContentEnvelope) -> str:
    """Fingerprint of everything the enrichment was derived from.

    Includes the media URL so that an item whose extraction improved -- a transcript
    appearing where there was none -- is recognized as stale and re-enriched.
    """
    parts = [
        envelope.platform.value,
        envelope.search_document(),
        envelope.media_url or "",
        str(envelope.duration_s or ""),
    ]
    return content_hash("\u241f".join(parts))


def _all_envelopes(cache: EnvelopeCache, limit: int) -> list[ContentEnvelope]:
    return cache.iter_envelopes(limit=limit)
