"""Link content extraction layer.

Public surface, in the order you will usually touch it:

    from extractor import ExtractionCascade

    cascade = ExtractionCascade()
    result = cascade.extract("https://www.reddit.com/r/python/comments/...")
    result.envelope.search_document()      # text to embed / index
    result.envelope.needs_media_understanding  # whether to pay for a video call
    result.winning_tier                    # which tier actually delivered

Everything downstream of this package consumes a `ContentEnvelope` and never a raw
URL, which is what lets extraction migrate to the client later without a rewrite.
"""

from .base import Extractor, ExtractorUnavailable
from .cache import EnvelopeCache
from .cascade import ExtractionCascade, TierTimeout
from .config import ExtractorConfig, ManagedApiConfig
from .envelope import (
    ContentEnvelope,
    ExtractionAttempt,
    ExtractionResult,
    ExtractionTarget,
    ExtractionTier,
    MediaKind,
    Platform,
    SignalStrength,
)
from .net_guard import UnsafeUrl, assert_public_url
from .urls import canonicalize, detect_platform, looks_like_url, url_hash

__all__ = [
    "ExtractionCascade",
    "TierTimeout",
    "ContentEnvelope",
    "ExtractionResult",
    "ExtractionAttempt",
    "ExtractionTarget",
    "ExtractionTier",
    "Platform",
    "SignalStrength",
    "MediaKind",
    "Extractor",
    "ExtractorUnavailable",
    "ExtractorConfig",
    "ManagedApiConfig",
    "EnvelopeCache",
    "canonicalize",
    "detect_platform",
    "looks_like_url",
    "url_hash",
    "assert_public_url",
    "UnsafeUrl",
]

__version__ = "0.1.0"
