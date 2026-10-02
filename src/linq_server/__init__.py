"""Stateless enrichment + embedding service for the LinQ iOS app.

This is deliberately *not* `src/api`. That one is a full library server: it owns a SQLite
store, a job queue, a web UI, and it is the authority on what you have saved. LinQ already
has all of that — its own SQLite in an App Group container, its own Supabase sync, its own
UI. What LinQ cannot do on device is run Gemini or Nemotron.

So this service stores nothing and remembers nothing about the user. It answers two
questions:

    "here is a link, what is in it?"        -> POST /v1/analyze
    "here is some text, vectorise it"       -> POST /v1/embed

Everything underneath is reused from the existing engine rather than reimplemented:
`ExtractionCascade` for getting at the content, `Enricher`/`GeminiProvider` for the video
understanding, `chunk_document` for the one-vector-per-detail split that makes
half-remembered queries work, and `NvidiaEmbedder` for the 2048-dimension vectors.

The reason a server has to exist at all: both models are network models, and the API keys
cannot ship inside an iOS binary — anyone can pull them out of the IPA and spend the quota.
The keys live here; the app authenticates to this instead.
"""

from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    EmbedRequest,
    EmbedResponse,
    EmbeddingBlock,
)
from .service import AnalysisService

__all__ = [
    "AnalysisService",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "EmbedRequest",
    "EmbedResponse",
    "EmbeddingBlock",
]
