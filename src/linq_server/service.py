"""The glue: URL in, enrichment plus vectors out.

Nothing novel here — every stage is the existing engine, wired without its storage layer:

    ExtractionCascade  ->  Enricher (GeminiProvider)  ->  chunk_document  ->  NvidiaEmbedder

Two decisions are worth knowing about.

**Enrichment caching is kept.** `Enricher` keys its cache on a hash of the extracted content,
so re-analysing an unchanged link is free. That matters more than it sounds: the same reel
gets shared into the app from several devices, and a retry after a dropped connection is
common on mobile. The cache is content-addressed, not user-addressed, so it holds no personal
data — but it is also the one piece of state in an otherwise stateless service, and it lives
in the same SQLite file the CLI uses.

**One vector per detail, not one per document.** This is the whole reason a query like
"salting shoes" can find a reel whose caption says nothing of the sort: the detail *"sprinkles
foot powder into brown leather penny loafers"* is embedded on its own, so its meaning is not
averaged away into a paragraph about something else. `chunk_document` already implements this;
the service just has to not flatten it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from enrichment import Enricher, EnrichmentConfig
from enrichment.schema import Enrichment, EnrichmentMode
from extractor import ExtractionCascade, ExtractorConfig
from extractor.envelope import ContentEnvelope
from search import SearchConfig
from search.chunking import chunk_document
from search.embedder import BaseEmbedder, EmbedderError, EmbedderUnavailable, build_embedder

from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    EmbeddingBlock,
    Entities,
    ExtractionInfo,
)

log = logging.getLogger("linq.service")


class ServiceError(RuntimeError):
    """Analysis failed in a way the client should hear about."""


def _media_is_fetchable(envelope: ContentEnvelope) -> bool:
    """Whether something could actually send this item's pixels to a model.

    Being classified as a video is not the same as having a video in hand — which is the
    distinction the escalation rule does not draw on its own.
    """
    url = (envelope.canonical_url or "").lower()
    if "youtube.com" in url or "youtu.be" in url:
        # Gemini resolves YouTube from the URL; nothing needs downloading.
        return True
    return bool(envelope.media_url)


@dataclass(slots=True)
class EmbedResult:
    model: str
    dim: int
    vectors: list[list[float]]
    kinds: list[str]


class AnalysisService:
    def __init__(self) -> None:
        self.extractor_config = ExtractorConfig.from_env()
        self.enrichment_config = EnrichmentConfig.from_env()
        self.search_config = SearchConfig.from_env()

        self.cascade = ExtractionCascade(self.extractor_config)
        self.enricher = Enricher(self.enrichment_config)
        self.embedder: BaseEmbedder = build_embedder(self.search_config)

    # ------------------------------------------------------------------ capabilities
    def capabilities(self) -> dict[str, object]:
        vector_ok, vector_reason = self.embedder.availability()
        model, dim = self.embedder.signature
        cookie_file = self.extractor_config.cookie_file
        return {
            "extraction": {
                # Booleans only: lets you confirm the Instagram cookies secret file is
                # mounted without exposing anything about it.
                "cookie_file_configured": bool(cookie_file),
                "cookie_file_found": bool(cookie_file) and os.path.isfile(cookie_file),
                "managed_api": self.extractor_config.managed_api.enabled,
            },
            "enrichment": {
                "text_model": self.enrichment_config.text_model,
                "media_model": self.enrichment_config.media_model,
                "media_fallback": self.enrichment_config.media_fallback_model,
                "allow_media": self.enrichment_config.allow_media,
                "has_api_key": self.enrichment_config.has_api_key,
            },
            "embedding": {
                "model": model,
                "dim": dim,
                "available": vector_ok,
                "reason": vector_reason,
                "query_cache_hits": self.embedder.query_cache_hits,
            },
        }

    # ----------------------------------------------------------------------- analyze
    def analyze(self, request: AnalyzeRequest, *, trace_id: str = "local") -> AnalyzeResponse:
        """Extract, enrich, chunk and embed one link. Blocking; call off the event loop."""
        log.info(
            "[search-pipeline:%s] 2. extraction started url=%s client_payload=%s",
            trace_id,
            request.url,
            bool(request.client_payload()),
        )
        # retry_degraded: a link that failed extraction earlier (Instagram blocking an
        # anonymous request, cookies not yet configured) must be tried again rather than
        # served from cache as a permanent failure. Successful extractions still hit cache.
        result = self.cascade.extract(
            request.url,
            client_payload=request.client_payload(),
            retry_degraded=True,
        )
        envelope = result.envelope
        log.info(
            "[search-pipeline:%s] 2. extraction complete tier=%s signal=%s "
            "media_kind=%s degraded=%s words=%s transcript=%s media_url=%s title=%r",
            trace_id,
            envelope.tier.value,
            envelope.signal.value,
            envelope.media_kind.value,
            envelope.degraded,
            envelope.prose_word_count,
            bool(envelope.transcript),
            bool(envelope.media_url),
            envelope.title,
        )

        mode = EnrichmentMode.MEDIA if request.force_media else None

        # Downgrade to text when media mode is unsatisfiable.
        #
        # `decide_mode` escalates on `media_kind is VIDEO`, and the OpenGraph tier labels
        # every Instagram, TikTok and YouTube link VIDEO by platform — before knowing whether
        # a media URL was actually obtained. For a reel the client could not resolve (no
        # Instagram session) that produces MEDIA mode with nothing to look at, and
        # `GeminiProvider._media_parts` raises "no media URL available". The item then fails,
        # retries on a backoff, and fails again, when perfectly good text enrichment was
        # available the whole time.
        #
        # YouTube is exempt: Google fetches those from the URL itself, so no media_url is
        # needed.
        if not _media_is_fetchable(envelope):
            if mode is EnrichmentMode.MEDIA:
                log.info("force_media requested but no media available for %s; using text",
                         request.url)
            mode = EnrichmentMode.METADATA
        log.info(
            "[search-pipeline:%s] 3. enrichment started selected_mode=%s",
            trace_id,
            mode.value if mode else "automatic",
        )

        try:
            enrichment = self.enricher.enrich(
                envelope, force=request.force_media, mode=mode
            )
        except Exception as exc:  # noqa: BLE001
            # A failed enrichment must not lose the extraction. The client still gets the
            # title and whatever text was captured, and can retry the expensive part later.
            log.exception(
                "[search-pipeline:%s] ERROR enrichment failed url=%s: %s",
                trace_id,
                request.url,
                exc,
            )
            raise ServiceError(f"enrichment failed: {type(exc).__name__}: {exc}") from exc
        log.info(
            "[search-pipeline:%s] 4. enrichment complete provider=%s model=%s mode=%s "
            "details=%s tags=%s input_tokens=%s output_tokens=%s cost_usd=%.6f "
            "duration_ms=%s degraded=%s",
            trace_id,
            enrichment.provider,
            enrichment.model,
            enrichment.mode.value,
            len(enrichment.details),
            len(enrichment.tags),
            enrichment.input_tokens,
            enrichment.output_tokens,
            enrichment.cost_usd,
            enrichment.duration_ms,
            enrichment.degraded,
        )

        embedding = None
        if request.embed:
            embedding = self._embed_document(envelope, enrichment, trace_id=trace_id)
        else:
            log.info(
                "[search-pipeline:%s] 5. server embedding skipped; Apple embedding is local",
                trace_id,
            )

        response = AnalyzeResponse(
            url=envelope.canonical_url,
            url_hash=envelope.url_hash,
            title=envelope.title,
            summary=enrichment.summary,
            description=enrichment.description,
            details=list(enrichment.details),
            tags=list(enrichment.tags),
            category=enrichment.category,
            content_type=enrichment.content_type,
            entities=Entities(
                people=list(enrichment.entities.people),
                organizations=list(enrichment.entities.organizations),
                places=list(enrichment.entities.places),
                products=list(enrichment.entities.products),
            ),
            language=enrichment.language,
            mode=enrichment.mode.value,
            provider=enrichment.provider,
            model=enrichment.model,
            input_tokens=enrichment.input_tokens,
            output_tokens=enrichment.output_tokens,
            cost_usd=enrichment.cost_usd,
            duration_ms=enrichment.duration_ms,
            degraded=enrichment.degraded or envelope.degraded,
            note=enrichment.note,
            extraction=ExtractionInfo(
                tier=envelope.tier.value,
                signal=envelope.signal.value,
                media_kind=envelope.media_kind.value,
                degraded=envelope.degraded,
                word_count=envelope.prose_word_count,
                duration_s=envelope.duration_s,
                has_transcript=bool(envelope.transcript),
                note=envelope.extractor_note,
            ),
            embedding=(
                EmbeddingBlock(
                    model=embedding.model,
                    dim=embedding.dim,
                    vectors=embedding.vectors,
                    kinds=embedding.kinds,
                )
                if embedding
                else None
            ),
        )
        log.info(
            "[search-pipeline:%s] 5. response assembled summary_chars=%s description_chars=%s",
            trace_id,
            len(response.summary),
            len(response.description),
        )
        return response

    def _embed_document(
        self, envelope: ContentEnvelope, enrichment: Enrichment, *, trace_id: str = "local"
    ) -> EmbedResult | None:
        chunks = chunk_document(envelope, enrichment)
        texts = [chunk.text for chunk in chunks if chunk.text.strip()]
        kinds = [chunk.kind for chunk in chunks if chunk.text.strip()]
        if not texts:
            log.info("[search-pipeline:%s] no embedding chunks produced", trace_id)
            return None
        try:
            matrix = self.embedder.embed_documents(texts)
        except EmbedderUnavailable as exc:
            # No key: enrichment is still valuable on its own, since it feeds keyword
            # search. Return without vectors rather than failing the whole request.
            log.warning("embedding unavailable: %s", exc)
            return None
        except EmbedderError as exc:
            log.warning("embedding failed: %s", exc)
            return None
        model, dim = self.embedder.signature
        log.info(
            "[search-pipeline:%s] server embeddings complete model=%s dimension=%s vectors=%s",
            trace_id,
            model,
            dim,
            len(matrix),
        )
        return EmbedResult(
            model=model,
            dim=dim,
            vectors=[[float(value) for value in row] for row in matrix],
            kinds=kinds,
        )

    # ------------------------------------------------------------------------- embed
    def embed(self, texts: list[str], *, kind: str) -> tuple[EmbedResult, int]:
        """Embed arbitrary text. `kind` picks the asymmetric input type.

        Query vs passage is not cosmetic: these models are trained to place a short question
        near the long passage that answers it, and that only holds if you declare which side
        you are embedding.
        """
        cleaned = [text for text in (t.strip() for t in texts) if text]
        if not cleaned:
            raise ServiceError("no non-empty texts supplied")

        before = self.embedder.query_cache_hits
        try:
            if kind == "query":
                # embed_query caches by exact text, which search-as-you-type leans on
                # heavily — typing, pausing and backspacing reissues the same prefix.
                rows = [self.embedder.embed_query(text) for text in cleaned]
                vectors = [[float(v) for v in row] for row in rows]
            else:
                matrix = self.embedder.embed_documents(cleaned)
                vectors = [[float(v) for v in row] for row in matrix]
        except EmbedderUnavailable as exc:
            raise ServiceError(str(exc)) from exc
        except EmbedderError as exc:
            raise ServiceError(f"embedding failed: {exc}") from exc

        model, dim = self.embedder.signature
        cached = self.embedder.query_cache_hits - before
        return EmbedResult(model=model, dim=dim, vectors=vectors, kinds=[kind] * len(vectors)), cached

    # ------------------------------------------------------------------------- close
    def close(self) -> None:
        self.enricher.close()
        self.cascade.close()
