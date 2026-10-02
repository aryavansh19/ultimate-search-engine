"""Search stage configuration.

The embedding provider, model and dimension are the three settings you cannot change
casually: vectors from different models are not comparable, so altering any of them
invalidates the whole index. All three are folded into each document's source hash, which
turns that from a silent correctness bug into an ordinary re-index.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# Embedding providers, all verified against the live APIs rather than assumed.
#
#   nemotron  free, 2048 dims, 33K context, pre-normalized (L2 exactly 1.0).
#             Measured noise floor 0.111 -- a query with no true match in the library
#             tops out there, versus 0.501 on gemini-embedding, which is what let a
#             nonsense query ("feromones") rank first by accident. Limited to
#             ~20 requests/minute, with no daily cap: the whole library embeds in two
#             requests.
#   gemini    768 dims, $0.15/1M, and capped at 1000 embed requests per day per model
#             where every *text* counts as one request. That cap is what left 14 of 30
#             items with no vectors mid-reindex.
#   3-small   paid ($0.02/1M) fallback with no free-tier rate limit, noise floor 0.111.
#
# Free models can be withdrawn at short notice, which is exactly why this is a selectable
# provider rather than a hardcoded endpoint.
EMBED_PROVIDERS: dict[str, dict[str, object]] = {
    # NVIDIA's own NIM endpoint. Same model as `nemotron` below, but reached directly, which
    # buys two things measured against the live APIs:
    #
    #   * Quota headroom. 45 rapid requests passed with no 429, where OpenRouter's free tier
    #     caps at 50 per *day* -- the limit that actually took semantic search down.
    #   * `input_type`, so passages and queries are embedded in their proper asymmetric
    #     modes. Worth having, though measurement showed the gain is small: OpenRouter
    #     silently uses query mode for everything, which benchmarked level with doing it
    #     correctly (noise floor 0.108 vs 0.110, avg gap 0.209 vs 0.212). The configuration
    #     that would have hurt is passage/passage at 0.264, and that is not what was running.
    "nvidia": {
        "kind": "nvidia",
        "model": "nvidia/nemotron-3-embed-1b",
        "dimensions": 2048,
        "rpm": 40,
        "supports_dimensions": False,
    },
    "nemotron": {
        "kind": "openrouter",
        "model": "nvidia/nemotron-3-embed-1b:free",
        "dimensions": 2048,
        "rpm": 15,
        "supports_dimensions": False,
    },
    "3-small": {
        "kind": "openrouter",
        "model": "openai/text-embedding-3-small",
        "dimensions": 768,
        "rpm": 60,
        "supports_dimensions": True,
    },
    "bge-m3": {
        "kind": "openrouter",
        "model": "baai/bge-m3",
        "dimensions": 1024,
        "rpm": 60,
        "supports_dimensions": False,
    },
    "gemini": {
        "kind": "gemini",
        "model": "gemini-embedding-001",
        "dimensions": 768,
        "rpm": 6,
        "supports_dimensions": True,
    },
    "gemini-2": {
        "kind": "gemini",
        "model": "gemini-embedding-2",
        "dimensions": 768,
        "rpm": 6,
        "supports_dimensions": True,
    },
}

DEFAULT_EMBED_PROVIDER = "nvidia"

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class SearchConfig:
    # --- embedding provider --------------------------------------------------
    embed_provider: str = DEFAULT_EMBED_PROVIDER
    embed_kind: str = "nvidia"
    embed_model: str = "nvidia/nemotron-3-embed-1b"
    dimensions: int = 2048
    supports_dimensions: bool = False

    gemini_api_key: str | None = None
    gemini_api_base: str = GEMINI_API_BASE
    openrouter_api_key: str | None = None
    openrouter_api_base: str = OPENROUTER_API_BASE
    nvidia_api_key: str | None = None
    nvidia_api_base: str = NVIDIA_API_BASE

    store_path: Path = Path("data/extractor_cache.sqlite3")
    request_timeout: float = 60.0
    embed_batch_size: int = 100
    # Requests per minute for *bulk document* embedding. Interactive query embedding is
    # exempt: pacing a keystroke-driven search behind a multi-second gate would make the
    # app unusable, and queries are single-text, cached, and rare next to a reindex.
    embed_requests_per_minute: int = 15
    # Retry budget for *interactive query* embedding. Small on purpose: a search that cannot
    # embed within a second or two should fall back to keyword results, not make the user
    # wait out a rate-limit window. Bulk indexing keeps the full budget.
    query_retry_attempts: int = 2

    # Chunking. Most saved links are short enough to be a single chunk; this only matters
    # for long articles and transcripts.
    chunk_chars: int = 2000
    chunk_overlap_chars: int = 300

    # Reciprocal Rank Fusion. k=60 is the value from the original paper and is deliberately
    # not tuned per query -- its appeal is needing no calibration between retrievers whose
    # scores are on incomparable scales.
    rrf_k: int = 60
    # Keyword contributes at half weight. Measured on the real library across two batteries
    # -- 10 descriptive queries and 8 exact-identifier queries:
    #
    #                        descriptive   identifiers
    #   vector only              10/10         8/8
    #   keyword only              6/10         7/8
    #   hybrid, keyword 1.0       8/10         8/8   <- fusion loses to vector alone
    #   hybrid, keyword 0.5       9/10         8/8
    #
    # At equal weight, a document matching one incidental query token ("apple", "shoes")
    # earns keyword rank 1 and, fused with a mediocre vector rank, outranks the document the
    # vector leg placed first with a decisive margin. Half weight recovers that at no cost
    # to identifier lookups.
    #
    # Keyword is deliberately kept rather than dropped despite scoring worse here: it is the
    # local, sub-millisecond leg that makes search-as-you-type feel instant, and the fallback
    # when the embedding API has no key or no quota. It is also under-measured at 34
    # documents -- rare-token precision matters far more at several thousand.
    keyword_weight: float = 0.5
    vector_weight: float = 1.0

    # Treat a half-typed final word as a prefix. Required for search-as-you-type: FTS5
    # matches whole tokens, so without this a live search box shows nothing until the user
    # finishes a word.
    prefix_last: bool = True

    # Candidates each retriever contributes before fusion. Larger than the final result
    # count on purpose: fusion can only reorder what it was given, so a document ranked
    # 30th by keyword and 3rd by vector is only recoverable if the keyword leg looked
    # deeper than the limit.
    candidate_depth: int = 50

    # BM25 column weights, in FTS5 column order. Title and tags describe the item directly;
    # body text is diluted by everything else in the document.
    bm25_weights: tuple[float, ...] = (0.0, 8.0, 6.0, 4.0, 6.0, 5.0, 1.0)

    @property
    def api_key(self) -> str | None:
        """Key for the active embedding provider."""
        return {
            "gemini": self.gemini_api_key,
            "nvidia": self.nvidia_api_key,
        }.get(self.embed_kind, self.openrouter_api_key)

    @property
    def api_base(self) -> str:
        return {
            "gemini": self.gemini_api_base,
            "nvidia": self.nvidia_api_base,
        }.get(self.embed_kind, self.openrouter_api_base)

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> SearchConfig:
        name = (os.getenv("SEARCH_EMBED_PROVIDER") or DEFAULT_EMBED_PROVIDER).strip().lower()
        spec = EMBED_PROVIDERS.get(name)
        if spec is None:
            spec = EMBED_PROVIDERS[DEFAULT_EMBED_PROVIDER]
            name = DEFAULT_EMBED_PROVIDER

        # An explicit model or dimension overrides the provider default, so a new model can
        # be tried without a code change.
        model = os.getenv("SEARCH_EMBED_MODEL") or str(spec["model"])
        dimensions = _int("SEARCH_DIMENSIONS", int(spec["dimensions"]))  # type: ignore[arg-type]

        return cls(
            embed_provider=name,
            embed_kind=str(spec["kind"]),
            embed_model=model,
            dimensions=dimensions,
            supports_dimensions=bool(spec["supports_dimensions"]),
            gemini_api_key=(
                os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None
            ),
            gemini_api_base=os.getenv("GEMINI_API_BASE", GEMINI_API_BASE).rstrip("/"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            openrouter_api_base=os.getenv(
                "OPENROUTER_API_BASE", OPENROUTER_API_BASE
            ).rstrip("/"),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY") or None,
            nvidia_api_base=os.getenv("NVIDIA_API_BASE", NVIDIA_API_BASE).rstrip("/"),
            store_path=Path(
                os.getenv("EXTRACTOR_CACHE_PATH", "data/extractor_cache.sqlite3")
            ),
            request_timeout=_float("SEARCH_TIMEOUT", 60.0),
            embed_batch_size=_int("SEARCH_EMBED_BATCH", 100),
            embed_requests_per_minute=_int("SEARCH_EMBED_RPM", int(spec["rpm"])),  # type: ignore[arg-type]
            query_retry_attempts=_int("SEARCH_QUERY_RETRIES", 2),
            chunk_chars=_int("SEARCH_CHUNK_CHARS", 2000),
            chunk_overlap_chars=_int("SEARCH_CHUNK_OVERLAP", 300),
            rrf_k=_int("SEARCH_RRF_K", 60),
            keyword_weight=_float("SEARCH_KEYWORD_WEIGHT", 0.5),
            vector_weight=_float("SEARCH_VECTOR_WEIGHT", 1.0),
            candidate_depth=_int("SEARCH_CANDIDATE_DEPTH", 50),
        )
