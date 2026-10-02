"""Extractor tier implementations.

`DEFAULT_EXTRACTORS` is the fallback chain. Order is derived from each tier's rank,
not from this list, so appending a new extractor is enough -- the cascade sorts it
into position.
"""

from ..base import Extractor
from .client_payload import ClientPayloadExtractor
from .managed_api import ManagedApiExtractor
from .opengraph import OpenGraphExtractor
from .ytdlp import YtDlpExtractor

DEFAULT_EXTRACTORS: tuple[type[Extractor], ...] = (
    ClientPayloadExtractor,
    YtDlpExtractor,
    ManagedApiExtractor,
    OpenGraphExtractor,
)

__all__ = [
    "ClientPayloadExtractor",
    "YtDlpExtractor",
    "ManagedApiExtractor",
    "OpenGraphExtractor",
    "DEFAULT_EXTRACTORS",
]
