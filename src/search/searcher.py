"""Hybrid search: BM25 and vector similarity, fused with RRF.

Both legs run because they fail in opposite directions, which is the single most
important property of this design:

* Vector search finds "that video about making pasta from scratch" when the caption never
  said pasta. It is useless at "@zuck" or "M4 Pro", because embeddings smear rare exact
  tokens into their neighbourhoods.
* Keyword search nails those exact tokens and fails completely on paraphrase. Ask for
  "how to remember things better" and a document about spaced repetition scores zero.

Each retriever contributes more candidates than the final result count, because fusion can
only reorder what it was handed -- a document ranked 30th by keyword and 3rd by vector is
recoverable only if the keyword leg looked deeper than the limit.

Vector scoring is max-over-chunks: an item's score is its single best-matching chunk. Mean
pooling would penalize long documents for containing anything besides the answer, which is
the normal state of a long document.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .config import SearchConfig
from .embedder import BaseEmbedder, EmbedderError, EmbedderUnavailable, build_embedder
from .fusion import reciprocal_rank_fusion
from .store import SearchStore

log = logging.getLogger("search.searcher")


class SearchMode(str, Enum):
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    VECTOR = "vector"


@dataclass(slots=True)
class SearchHit:
    url_hash: str
    score: float
    title: str | None
    summary: str | None
    canonical_url: str
    platform: str
    category: str | None
    content_type: str | None
    tags: list[str] = field(default_factory=list)
    author: str | None = None
    thumbnail_url: str | None = None
    keyword_rank: int | None = None
    vector_rank: int | None = None
    vector_similarity: float | None = None
    matched_text: str | None = None
    # Analysis coverage, carried through so a search result shows the same badge as the
    # library list. Without these the UI cannot distinguish an item whose video was fully
    # analysed from one tagged off a caption, and defaults to showing the weakest label.
    analysis_mode: str | None = None
    audio_covered: bool = False
    visual_covered: bool = False
    detail_count: int = 0
    is_video: bool = False

    @property
    def found_by(self) -> str:
        if self.keyword_rank and self.vector_rank:
            return "both"
        if self.keyword_rank:
            return "keyword"
        if self.vector_rank:
            return "vector"
        return "none"


@dataclass(slots=True)
class SearchResponse:
    query: str
    mode: SearchMode
    hits: list[SearchHit]
    keyword_candidates: int = 0
    vector_candidates: int = 0
    notes: list[str] = field(default_factory=list)
    vector_available: bool = True


class Searcher:
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
    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        category: str | None = None,
        content_type: str | None = None,
        platform: str | None = None,
        tag: str | None = None,
    ) -> SearchResponse:
        query = (query or "").strip()
        notes: list[str] = []
        if not query:
            return SearchResponse(query=query, mode=mode, hits=[], notes=["empty query"])

        filters: dict[str, str] = {}
        if category:
            filters["category"] = category
        if content_type:
            filters["content_type"] = content_type
        if platform:
            filters["platform"] = platform

        allowed: set[str] | None = self.store.matching_hashes(filters)
        if tag:
            tagged = self.store.hashes_with_tag(tag)
            allowed = tagged if allowed is None else (allowed & tagged)
            if not tagged:
                notes.append(f"no items carry the tag '{tag}'")

        depth = max(limit, self.config.candidate_depth)
        ranked: dict[str, list[str]] = {}
        keyword_scores: dict[str, float] = {}
        vector_scores: dict[str, float] = {}
        vector_chunk: dict[str, int] = {}
        vector_available = True

        if mode in (SearchMode.HYBRID, SearchMode.KEYWORD):
            pairs = self.store.keyword_search(
                query,
                depth,
                self.config.bm25_weights,
                filters or None,
                prefix_last=self.config.prefix_last,
            )
            if tag and allowed is not None:
                pairs = [p for p in pairs if p[0] in allowed]
            keyword_scores = dict(pairs)
            ranked["keyword"] = [url_hash for url_hash, _ in pairs]

        if mode in (SearchMode.HYBRID, SearchMode.VECTOR):
            try:
                vector_scores, vector_chunk = self._vector_search(query, depth, allowed)
                ranked["vector"] = sorted(
                    vector_scores, key=lambda h: -vector_scores[h]
                )[:depth]
            except EmbedderUnavailable as exc:
                vector_available = False
                notes.append(str(exc))
            except EmbedderError as exc:
                vector_available = False
                notes.append(f"vector search failed: {exc}")

        if not ranked:
            return SearchResponse(
                query=query, mode=mode, hits=[], notes=notes,
                vector_available=vector_available,
            )

        fused = reciprocal_rank_fusion(
            ranked,
            k=self.config.rrf_k,
            weights={
                "keyword": self.config.keyword_weight,
                "vector": self.config.vector_weight,
            },
        )[:limit]

        hits = self._hydrate(fused, keyword_scores, vector_scores, vector_chunk)
        return SearchResponse(
            query=query,
            mode=mode,
            hits=hits,
            keyword_candidates=len(ranked.get("keyword", [])),
            vector_candidates=len(ranked.get("vector", [])),
            notes=notes,
            vector_available=vector_available,
        )

    # ------------------------------------------------------------------- internals
    def _vector_search(
        self, query: str, depth: int, allowed: set[str] | None
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Cosine similarity against every chunk, reduced to best-chunk-per-document."""
        model, dim = self.embedder.signature
        chunk_ids, hashes, matrix = self.store.load_vectors(model, dim)
        if matrix.shape[0] == 0:
            signatures = self.store.vector_signatures()
            if signatures:
                # Vectors exist but under a different model/dimension. Silently returning
                # nothing here would look like "search is broken" rather than "the index
                # was built with a different embedding model".
                have = ", ".join(f"{m}@{d} ({c})" for m, d, c in signatures)
                raise EmbedderError(
                    f"no vectors for {model}@{dim}; index contains {have}. Re-index."
                )
            raise EmbedderError("no vectors indexed yet; run `search index`")

        query_vector = self.embedder.embed_query(query)
        # Both sides are L2-normalized, so the dot product *is* cosine similarity.
        similarities = matrix @ query_vector

        best: dict[str, float] = {}
        best_chunk: dict[str, int] = {}
        for position in np.argsort(-similarities):
            url_hash = hashes[position]
            if allowed is not None and url_hash not in allowed:
                continue
            score = float(similarities[position])
            if url_hash not in best or score > best[url_hash]:
                best[url_hash] = score
                best_chunk[url_hash] = chunk_ids[position]
            if len(best) >= depth * 3:
                # argsort is descending, so the remainder cannot beat what we have.
                break
        return best, best_chunk

    def _hydrate(
        self,
        fused: list[Any],
        keyword_scores: dict[str, float],
        vector_scores: dict[str, float],
        vector_chunk: dict[str, int],
    ) -> list[SearchHit]:
        hashes = [item.key for item in fused]
        documents = self.store.get_documents(hashes)
        chunk_texts = self.store.get_chunk_texts(
            [vector_chunk[h] for h in hashes if h in vector_chunk]
        )

        hits: list[SearchHit] = []
        for item in fused:
            document = documents.get(item.key)
            if document is None:
                continue
            tags_text = document.get("tags_text") or ""
            matched = chunk_texts.get(vector_chunk.get(item.key, -1))
            hits.append(
                SearchHit(
                    url_hash=item.key,
                    score=round(item.score, 6),
                    title=document.get("title"),
                    summary=document.get("summary"),
                    canonical_url=document.get("canonical_url", ""),
                    platform=document.get("platform", "web"),
                    category=document.get("category"),
                    content_type=document.get("content_type"),
                    tags=[t.strip() for t in tags_text.split(",") if t.strip()],
                    author=document.get("author"),
                    thumbnail_url=document.get("thumbnail_url"),
                    keyword_rank=item.ranks.get("keyword"),
                    vector_rank=item.ranks.get("vector"),
                    vector_similarity=(
                        round(vector_scores[item.key], 4)
                        if item.key in vector_scores
                        else None
                    ),
                    matched_text=_excerpt(matched) if matched else None,
                    analysis_mode=document.get("analysis_mode"),
                    audio_covered=bool(document.get("audio_covered")),
                    visual_covered=bool(document.get("visual_covered")),
                    detail_count=int(document.get("detail_count") or 0),
                    is_video=bool(document.get("is_video")),
                )
            )
        return hits

    def close(self) -> None:
        self.store.close()


def _excerpt(text: str, limit: int = 220) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."
