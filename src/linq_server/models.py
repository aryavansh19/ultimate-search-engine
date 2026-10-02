"""Request and response shapes for the LinQ service.

These are the contract the Swift client codes against, so they are written to be boring:
flat, explicitly typed, no polymorphism, and every field the app is expected to persist has
a direct counterpart in the LinQ SQLite schema.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    """What the app knows about a link at save time.

    Only `url` is required. Everything else is metadata the app already scraped on device,
    passed through so the server does not have to re-fetch the page — which matters because
    a logged-in iPhone can read an Instagram page that a datacenter IP cannot. These map
    onto the extraction cascade's client-payload tier.
    """

    model_config = ConfigDict(extra="ignore")

    url: str = Field(min_length=4)

    title: str | None = None
    caption: str | None = None
    description: str | None = None
    author: str | None = None
    keywords: list[str] = Field(default_factory=list)
    thumbnail_url: str | None = None
    media_url: str | None = None
    duration_s: float | None = None

    # Full rendered DOM from a session-bearing WKWebView, if the app ever captures one.
    # This is the highest-quality input available and beats every server-side tier.
    html: str | None = None

    force_media: bool = Field(
        default=False,
        description=(
            "Force the video path even when the text already looks informative. This is "
            "the 'Analyse video' action — it costs real money, so it is opt-in."
        ),
    )
    embed: bool = Field(
        default=True, description="Also return vectors, so saving is one round trip."
    )

    def has_substantial_content(self) -> bool:
        """Whether the client actually captured content, as opposed to just a label.

        A title and a hostname are not content. This distinction decides whether the
        client-payload tier is allowed to win the extraction cascade — see
        ``client_payload``.
        """
        return bool(self.html or self.caption or self.media_url)

    def client_payload(self) -> dict[str, Any]:
        """Flat payload for `ClientPayloadExtractor`, which aliases these key names.

        **Returns empty unless the client captured real content**, and that is the whole
        point of this method.

        The cascade stops at the first tier whose envelope has any text signal at all
        (`signal is not NONE`, which is five prose words). A payload of just
        `{"title": "How I trained for a marathon", "author": "youtube.com"}` clears that bar
        — so passing it made the client-payload tier win instantly, yt-dlp never ran, no
        `media_url` was ever discovered, `needs_media_understanding` was therefore False, and
        `decide_mode` returned METADATA. Net effect: the video was never looked at, and the
        model described the link from its title alone. Silently, and for almost every
        bookmark, since most titles are five words or more.

        So thin metadata is withheld deliberately. The server's own tiers recover the title
        anyway, and `merged_with` fills any gaps a later tier leaves. When the app does
        capture something real — rendered DOM from a session-bearing webview, a caption, a
        direct media URL — this tier wins on merit and skips the server-side fetch entirely,
        which is the case it was built for.
        """
        if not self.has_substantial_content():
            return {}

        payload: dict[str, Any] = {
            "title": self.title,
            "caption": self.caption or self.description,
            "author": self.author,
            "thumbnail_url": self.thumbnail_url,
            "media_url": self.media_url,
        }
        if self.html:
            payload["html"] = self.html
        if self.keywords:
            payload["tags"] = list(self.keywords)
        if self.duration_s is not None:
            payload["duration_s"] = self.duration_s
        return {key: value for key, value in payload.items() if value not in (None, "", [])}


class EmbeddingBlock(BaseModel):
    """Vectors for one document, plus the identity needed to compare them safely.

    `model` and `dim` are not decoration. Cosine similarity between vectors from different
    models is meaningless, and LinQ's `VectorMath.cosineSimilarity` returns 0 on a length
    mismatch rather than raising — so a mixed index degrades silently instead of failing.
    The client must store these alongside the vectors and refuse to compare across them.
    """

    model: str
    dim: int
    vectors: list[list[float]]
    kinds: list[str] = Field(
        default_factory=list,
        description="Per-vector provenance: 'digest', 'detail' or 'body'.",
    )
    normalized: bool = True


class Entities(BaseModel):
    model_config = ConfigDict(extra="ignore")

    people: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)


class ExtractionInfo(BaseModel):
    """How the content was obtained, so the client can tell thin results from rich ones."""

    tier: str
    signal: str
    media_kind: str
    degraded: bool
    word_count: int = 0
    duration_s: float | None = None
    has_transcript: bool = False
    note: str | None = None


class AnalyzeResponse(BaseModel):
    """Everything the app should persist for one bookmark.

    `summary` and `details` are the two fields that make hybrid search work: `summary` goes
    into the FTS `summary` column, and each entry in `details` becomes both FTS body text
    and its own embedding vector.
    """

    url: str
    url_hash: str
    title: str | None = None

    summary: str = ""
    description: str = ""
    details: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    category: str = "other"
    content_type: str = "other"
    entities: Entities = Field(default_factory=Entities)
    language: str | None = None

    # Provenance and cost, so the client can show what happened and the caller can audit
    # spend without a separate logging channel.
    mode: str = "heuristic"
    provider: str = "heuristic"
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    degraded: bool = False
    note: str | None = None

    extraction: ExtractionInfo | None = None
    embedding: EmbeddingBlock | None = None


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    texts: list[str] = Field(min_length=1)
    kind: Literal["passage", "query"] = "passage"


class EmbedResponse(BaseModel):
    model: str
    dim: int
    vectors: list[list[float]]
    normalized: bool = True
    cached: int = Field(
        default=0, description="How many query vectors were served from cache."
    )
