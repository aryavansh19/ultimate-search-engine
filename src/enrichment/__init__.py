"""Enrichment stage: turn an extracted envelope into searchable tags and summaries.

    from extractor import ExtractionCascade
    from enrichment import Enricher

    cascade = ExtractionCascade()
    enricher = Enricher()

    envelope = cascade.extract(url).envelope
    enrichment = enricher.enrich(envelope)

    enrichment.tags            # normalized, de-duplicated, noise-stripped
    enrichment.search_text()   # feeds the embedding stage
    enrichment.cost_usd        # what this item actually cost

Consumes a `ContentEnvelope` and nothing else, so it works identically whether the
envelope came from a server-side scrape or from an iOS client with the user's own
session.
"""

from .config import EnrichmentConfig
from .enricher import Enricher, source_hash
from .pricing import (
    MODEL_PRICES,
    describe_video_budget,
    estimate_cost,
    estimate_video_tokens,
)
from .providers import (
    EnrichmentProvider,
    GeminiProvider,
    HeuristicProvider,
    ProviderError,
    ProviderUnavailable,
)
from .schema import Enrichment, EnrichmentMode, Entities
from .store import EnrichmentStore
from .taxonomy import CATEGORIES, CONTENT_TYPES, normalize_tags, slugify_tag

__all__ = [
    "Enricher",
    "Enrichment",
    "EnrichmentMode",
    "Entities",
    "EnrichmentConfig",
    "EnrichmentStore",
    "EnrichmentProvider",
    "GeminiProvider",
    "HeuristicProvider",
    "ProviderError",
    "ProviderUnavailable",
    "CATEGORIES",
    "CONTENT_TYPES",
    "normalize_tags",
    "slugify_tag",
    "source_hash",
    "estimate_cost",
    "estimate_video_tokens",
    "describe_video_budget",
    "MODEL_PRICES",
]

__version__ = "0.1.0"
