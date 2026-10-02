"""Build the search index from extracted envelopes and their enrichments.

Reads both earlier stages and writes one searchable document per saved link: FTS columns
for keyword matching, chunks plus vectors for semantic matching.

Indexing is idempotent and change-aware. Each document records a source hash covering the
envelope text, the enrichment and the embedding signature, so re-running only touches what
actually changed. That matters because embedding is the one step here that costs money.

Two hard-won details:

* **Chunks and vectors live and die together.** Replacing an item's chunks drops its
  vectors, because vectors are keyed by chunk id and chunk text may have changed. So a
  reindex that cannot embed must not rewrite chunks either -- doing so wiped every vector
  in the library in one command while reporting success.
* **Embedding is batched across items, not per item.** The free tier's quota counts
  requests, and one request per item hit a 429 partway through a 30-link library, leaving
  14 items with no vectors. `batchEmbedContents` takes up to 100 texts per call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from enrichment import Enrichment, EnrichmentStore
from enrichment.schema import EnrichmentMode
from extractor import ContentEnvelope, EnvelopeCache
from extractor.envelope import MediaKind, content_hash

from .chunking import build_keyword_fields, chunk_document
from .config import SearchConfig
from .embedder import BaseEmbedder, EmbedderError, EmbedderUnavailable, build_embedder
from .store import SearchStore

log = logging.getLogger("search.indexer")


@dataclass(slots=True)
class IndexReport:
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    chunks: int = 0
    vectors: int = 0
    estimated_cost_usd: float = 0.0
    notes: list[str] | None = None

    def note(self, message: str) -> None:
        if self.notes is None:
            self.notes = []
        if message not in self.notes:
            self.notes.append(message)


class Indexer:
    def __init__(
        self,
        config: SearchConfig | None = None,
        store: SearchStore | None = None,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        self.config = config or SearchConfig.from_env()
        self.store = store if store is not None else SearchStore(self.config.store_path)
        self.embedder = embedder if embedder is not None else build_embedder(self.config)

    # ------------------------------------------------------------------ public API
    def index_all(
        self, limit: int = 1000, *, force: bool = False, embed: bool = True
    ) -> IndexReport:
        cache = EnvelopeCache(self.config.store_path)
        enrichments = EnrichmentStore(self.config.store_path)
        report = IndexReport()

        can_embed = embed
        if embed:
            available, reason = self.embedder.availability()
            if not available:
                can_embed = False
                report.note(f"{reason}; indexed keyword-only")

        pending: list[tuple[str, list[int], list[str]]] = []

        try:
            for envelope in cache.iter_envelopes(limit=limit):
                enrichment = enrichments.get(envelope.url_hash)
                try:
                    outcome = self._write(
                        envelope,
                        enrichment,
                        force=force,
                        report=report,
                        rewrite_chunks=self._may_rewrite_chunks(envelope, can_embed, report),
                    )
                    if outcome is None:
                        report.skipped += 1
                        continue
                    report.indexed += 1
                    chunk_ids, texts = outcome
                    if can_embed and chunk_ids:
                        pending.append((envelope.url_hash, chunk_ids, texts))
                except Exception as exc:  # noqa: BLE001 - one bad item must not stop a backfill
                    report.failed += 1
                    report.note(f"{envelope.canonical_url}: {type(exc).__name__}: {exc}")
                    log.debug(
                        "indexing failed for %s", envelope.canonical_url, exc_info=True
                    )

            if pending:
                self._embed_batched(pending, report)
        finally:
            cache.close()
            enrichments.close()

        report.estimated_cost_usd = self.embedder.estimated_cost_usd
        return report

    def index_one(
        self,
        envelope: ContentEnvelope,
        enrichment: Enrichment | None,
        *,
        force: bool = False,
        embed: bool = True,
        report: IndexReport | None = None,
    ) -> bool:
        """Index a single item, embedding inline. Returns True if anything was written."""
        outcome = self._write(
            envelope,
            enrichment,
            force=force,
            report=report,
            rewrite_chunks=self._may_rewrite_chunks(envelope, embed, report),
        )
        if outcome is None:
            return False

        chunk_ids, texts = outcome
        if embed and chunk_ids:
            self._embed_batched([(envelope.url_hash, chunk_ids, texts)], report)
        return True

    # ------------------------------------------------------------------- internals
    def _may_rewrite_chunks(
        self, envelope: ContentEnvelope, embed: bool, report: IndexReport | None
    ) -> bool:
        """Whether it is safe to replace this item's chunks.

        Rewriting chunks deletes their vectors. If we cannot regenerate them, keeping the
        old chunks is strictly better than destroying working semantic search to refresh a
        keyword row.
        """
        if embed:
            return True
        if not self.store.has_vectors(envelope.url_hash):
            return True
        if report:
            report.note(
                "kept existing chunks and vectors; re-chunking without embedding would "
                "have deleted them"
            )
        return False

    def _write(
        self,
        envelope: ContentEnvelope,
        enrichment: Enrichment | None,
        *,
        force: bool,
        report: IndexReport | None,
        rewrite_chunks: bool,
    ) -> tuple[list[int], list[str]] | None:
        """Write the document row, FTS entry and (optionally) chunks.

        Returns (chunk_ids, chunk_texts) awaiting embedding, or None if the item was
        already current.
        """
        signature = index_source_hash(
            envelope, enrichment, self.config.embed_model, self.config.dimensions
        )
        if not force and self.store.indexed_source_hash(envelope.url_hash) == signature:
            return None

        fields = build_keyword_fields(envelope, enrichment)
        self.store.upsert_document(
            {
                "url_hash": envelope.url_hash,
                "canonical_url": envelope.canonical_url,
                "platform": envelope.platform.value,
                "title": envelope.title,
                "author": envelope.author,
                "summary": enrichment.summary if enrichment else None,
                "description": enrichment.description if enrichment else None,
                "category": enrichment.category if enrichment else None,
                "content_type": enrichment.content_type if enrichment else None,
                "tags_text": ", ".join(enrichment.tags) if enrichment else None,
                "thumbnail_url": envelope.thumbnail_url,
                "published_at": (
                    envelope.published_at.isoformat() if envelope.published_at else None
                ),
                "source_hash": signature,
                "enriched": bool(enrichment and not enrichment.degraded),
                # Analysis coverage, so the UI can state how deeply this item was examined
                # rather than leaving it a mystery. A media-mode enrichment counts as
                # visual coverage because that pass is what produces the detail list.
                "analysis_mode": enrichment.mode.value if enrichment else None,
                "audio_covered": envelope.has_audio_coverage,
                "visual_covered": (
                    envelope.has_visual_coverage
                    or bool(enrichment and enrichment.mode is EnrichmentMode.MEDIA)
                ),
                "detail_count": len(enrichment.details) if enrichment else 0,
                "is_video": envelope.media_kind is MediaKind.VIDEO,
            },
            fields,
        )

        if not rewrite_chunks:
            return [], []

        chunks = chunk_document(
            envelope,
            enrichment,
            chunk_chars=self.config.chunk_chars,
            overlap_chars=self.config.chunk_overlap_chars,
        )
        chunk_ids = self.store.replace_chunks(
            envelope.url_hash, [(c.ordinal, c.kind, c.text) for c in chunks]
        )
        if report:
            report.chunks += len(chunks)
        return chunk_ids, [c.text for c in chunks]

    def _embed_batched(
        self,
        pending: list[tuple[str, list[int], list[str]]],
        report: IndexReport | None,
    ) -> None:
        """Embed every collected chunk in as few requests as possible, then store."""
        flat: list[str] = []
        spans: list[tuple[str, list[int], int, int]] = []
        for url_hash, chunk_ids, texts in pending:
            if len(chunk_ids) != len(texts):
                # Mismatched ids would pair vectors with the wrong chunk: invisible
                # corruption rather than a visible error. Skip rather than guess.
                if report:
                    report.note(f"chunk id mismatch for {url_hash}; embedding skipped")
                continue
            start = len(flat)
            flat.extend(texts)
            spans.append((url_hash, chunk_ids, start, len(flat)))

        if not flat:
            return

        try:
            vectors = self.embedder.embed_documents(flat)
        except EmbedderUnavailable as exc:
            if report:
                report.note(f"vector skipped: {exc}")
            return
        except EmbedderError as exc:
            # The keyword index is already written and useful; record and move on.
            if report:
                report.note(f"embedding failed: {exc}")
            log.debug("embedding failed", exc_info=True)
            return

        for url_hash, chunk_ids, start, end in spans:
            self.store.put_vectors(
                url_hash,
                chunk_ids,
                vectors[start:end],
                self.config.embed_model,
                self.config.dimensions,
            )
            if report:
                report.vectors += end - start

    def close(self) -> None:
        self.store.close()


def index_source_hash(
    envelope: ContentEnvelope,
    enrichment: Enrichment | None,
    model: str,
    dimensions: int,
) -> str:
    """Fingerprint of every input the index entry depends on.

    The embedding model and dimension are part of it deliberately. Switching either makes
    stored vectors incomparable with new ones, and folding them into the hash turns that
    from a silent correctness problem into an ordinary re-index.
    """
    parts = [
        envelope.search_document(),
        enrichment.search_text() if enrichment else "",
        str(enrichment.prompt_version) if enrichment else "0",
        model,
        str(dimensions),
    ]
    return content_hash("\u241f".join(parts))
