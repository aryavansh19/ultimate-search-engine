"""Search stage: index extracted and enriched links, then retrieve them.

    from search import Indexer, Searcher, SearchMode

    Indexer().index_all()
    response = Searcher().search("that pasta video", mode=SearchMode.HYBRID)
    for hit in response.hits:
        print(hit.score, hit.found_by, hit.title)

Keyword and vector retrieval run in parallel and are fused with Reciprocal Rank Fusion.
Keyword indexing works without an API key; vector search reports itself unavailable
rather than substituting a meaningless fallback.
"""

from .chunking import Chunk, build_digest, build_keyword_fields, chunk_document
from .config import EMBED_PROVIDERS, SearchConfig
from .embedder import (
    BaseEmbedder,
    EmbedderError,
    EmbedderUnavailable,
    GeminiEmbedder,
    NvidiaEmbedder,
    OpenAICompatEmbedder,
    build_embedder,
    normalize,
)
from .fusion import FusedItem, reciprocal_rank_fusion
from .indexer import Indexer, IndexReport, index_source_hash
from .searcher import SearchHit, SearchMode, SearchResponse, Searcher
from .store import SearchStore, to_fts_query

__all__ = [
    "Indexer",
    "IndexReport",
    "Searcher",
    "SearchMode",
    "SearchHit",
    "SearchResponse",
    "SearchConfig",
    "SearchStore",
    "BaseEmbedder",
    "GeminiEmbedder",
    "NvidiaEmbedder",
    "OpenAICompatEmbedder",
    "build_embedder",
    "EMBED_PROVIDERS",
    "EmbedderError",
    "EmbedderUnavailable",
    "normalize",
    "reciprocal_rank_fusion",
    "FusedItem",
    "Chunk",
    "chunk_document",
    "build_digest",
    "build_keyword_fields",
    "index_source_hash",
    "to_fts_query",
]

__version__ = "0.1.0"
