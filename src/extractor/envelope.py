"""The content envelope: the single contract every extraction tier produces.

This module is the load-bearing piece of the architecture. `ContentEnvelope` is
deliberately both:

  1. the normalized *output* of every server-side extractor tier, and
  2. the accepted *input* shape of the ingest API.

That duality is the whole point. Today a server-side tier (yt-dlp, a managed
scraper) fills the envelope. Tomorrow an iOS client fills it on-device from a
WKWebView running the user's own logged-in session, POSTs it, and nothing
downstream changes -- enrichment, embedding, indexing and search all consume an
envelope and neither know nor care who produced it.

So: never let downstream code accept a bare URL. It accepts an envelope.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


_HASHTAG = re.compile(r"#\w+")


class Platform(str, Enum):
    """Where a link came from. Drives which extractors are eligible."""

    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    REDDIT = "reddit"
    LINKEDIN = "linkedin"
    WEB = "web"


class ExtractionTier(str, Enum):
    """The fallback chain, in priority order.

    Ordering here is authoritative -- the cascade sorts by it. Lower `rank` runs
    first, and the first tier to return usable content wins.
    """

    CLIENT_PAYLOAD = "client_payload"
    YT_DLP = "yt_dlp"
    MANAGED_API = "managed_api"
    OPEN_GRAPH = "open_graph"
    DEGRADED = "degraded"

    @property
    def rank(self) -> int:
        return _TIER_ORDER.index(self)


_TIER_ORDER: list[ExtractionTier] = [
    ExtractionTier.CLIENT_PAYLOAD,
    ExtractionTier.YT_DLP,
    ExtractionTier.MANAGED_API,
    ExtractionTier.OPEN_GRAPH,
    ExtractionTier.DEGRADED,
]


class SignalStrength(str, Enum):
    """How much usable text an envelope carries.

    This is the escalation gate for the cost cascade discussed in design: cheap
    metadata-only enrichment is enough for a large share of real links, and only
    weak-signal items justify paying for full video understanding. Compute it
    here, once, so the enrichment stage never has to guess.
    """

    STRONG = "strong"
    WEAK = "weak"
    NONE = "none"


class MediaKind(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    ARTICLE = "article"
    TEXT = "text"
    UNKNOWN = "unknown"


class ContentEnvelope(BaseModel):
    """Normalized, platform-agnostic representation of a saved link.

    Every field except the URL trio is optional and frequently absent. Downstream
    code must tolerate nulls everywhere -- a reel with no caption, no speech and
    no on-screen text is a real case, not an error.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # --- identity ------------------------------------------------------------
    canonical_url: str
    url_hash: str
    original_url: str | None = None
    platform: Platform = Platform.WEB

    # --- shallow metadata ----------------------------------------------------
    title: str | None = None
    author: str | None = None
    author_url: str | None = None
    caption: str | None = None
    thumbnail_url: str | None = None
    media_url: str | None = None
    media_kind: MediaKind = MediaKind.UNKNOWN
    duration_s: float | None = None
    published_at: datetime | None = None
    platform_tags: list[str] = Field(default_factory=list)
    like_count: int | None = None
    view_count: int | None = None
    comment_count: int | None = None

    # --- deep content (big; select explicitly, store in its own table) -------
    transcript: str | None = None
    ocr_text: str | None = None
    article_text: str | None = None
    visual_description: str | None = None

    # --- provenance ----------------------------------------------------------
    tier: ExtractionTier = ExtractionTier.DEGRADED
    degraded: bool = False
    reprocessable: bool = True
    extractor_note: str | None = None
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    raw: dict[str, Any] | None = Field(default=None, repr=False)

    # ------------------------------------------------------------------ utils
    @property
    def text_parts(self) -> list[tuple[str, str]]:
        """Labelled non-empty text fields, in descending order of signal value."""
        candidates = [
            ("title", self.title),
            ("author", self.author),
            ("caption", self.caption),
            ("tags", " ".join(self.platform_tags) if self.platform_tags else None),
            ("visual", self.visual_description),
            ("onscreen", self.ocr_text),
            ("transcript", self.transcript),
            ("article", self.article_text),
        ]
        return [(k, v.strip()) for k, v in candidates if v and v.strip()]

    def search_document(self) -> str:
        """Concatenated text used for embedding and full-text indexing.

        Field labels are included on purpose: they survive chunking and give the
        embedding model a little structure to hold onto.
        """
        return "\n".join(f"{label}: {value}" for label, value in self.text_parts)

    @property
    def word_count(self) -> int:
        return sum(len(v.split()) for _, v in self.text_parts)

    @property
    def prose_word_count(self) -> int:
        """Words excluding hashtags and the platform tag list.

        Hashtags are not prose and must not count toward "this item describes itself".
        A real reel caption read `Consistency > Motivation #Claude #AI #ChatGPT
        #Microsoft #Google` -- 24 words by naive counting, and almost no information
        about what happens in the video.
        """
        total = 0
        for label, value in self.text_parts:
            if label == "tags":
                continue
            total += len(_HASHTAG.sub(" ", value).split())
        return total

    @property
    def has_audio_coverage(self) -> bool:
        """Whether what was *said* has been captured, via a real transcript."""
        return bool(self.transcript)

    @property
    def has_visual_coverage(self) -> bool:
        """Whether what was *shown* has been captured -- frames or on-screen text."""
        return bool(self.visual_description or self.ocr_text)

    @property
    def has_examined_media(self) -> bool:
        """Whether anything at all looked past the metadata.

        Note the deliberate asymmetry, because it is easy to misread: a transcript
        satisfies this even though nothing has looked at a single frame. That is a cost
        decision, not a claim of completeness -- a YouTube transcript is a richer and
        cheaper record of a talking-head video than a sampled visual pass would be, so
        having one suppresses the expensive tier.

        The consequence is that a captioned video ends up with audio coverage and no
        visual coverage. Use the two properties above when you need to know which.
        """
        return self.has_audio_coverage or self.has_visual_coverage

    @property
    def signal(self) -> SignalStrength:
        """How searchable this item is on text alone.

        The NONE floor is deliberately above zero words. A blocked page hands back a
        one-word title like "Reddit" or "Instagram", which is technically text and
        completely useless -- counting it as signal would let a bot wall masquerade as
        a successful extraction and quietly suppress the escalation to a better tier.
        """
        body = self.transcript or self.article_text or self.visual_description or self.ocr_text
        words = self.prose_word_count

        # A caption cannot be strong evidence about what a video *contains*. However
        # descriptive the text is, nothing has looked at the audio or the pixels, so the
        # specifics people actually search for -- the objects, the actions, the on-screen
        # text -- are still unrecorded. Capping unexamined video at WEAK is what routes it
        # to media understanding instead of letting a hashtag-padded caption pass as a
        # full description.
        #
        # The previous rule did the opposite: it treated three or more hashtags as
        # *evidence of strength*, so the more reach bait a post carried, the less likely
        # it was to get analyzed.
        unexamined_video = self.media_kind is MediaKind.VIDEO and not self.has_examined_media

        if not unexamined_video:
            if body and words >= 40:
                return SignalStrength.STRONG
            if words >= 60:
                return SignalStrength.STRONG
        if (body and words >= 8) or words >= 5:
            return SignalStrength.WEAK
        return SignalStrength.NONE

    @property
    def needs_media_understanding(self) -> bool:
        """True when it is worth spending a multimodal LLM call on this item.

        Weak or absent text plus an actual video to look at. Strong-signal items
        skip the expensive tier entirely -- that decision is the difference
        between fractions of a cent and a couple of cents per saved link.
        """
        if self.signal is SignalStrength.STRONG:
            return False
        return bool(self.media_url) or self.media_kind is MediaKind.VIDEO

    def merged_with(self, other: ContentEnvelope) -> ContentEnvelope:
        """Fill this envelope's gaps from `other` without overwriting anything set.

        Used when a lower tier partially succeeds: a client payload may carry the
        caption while an OpenGraph fetch supplies the thumbnail. Keeping both
        beats discarding one.
        """
        merged = self.model_dump()
        for key, value in other.model_dump().items():
            if key in {"tier", "degraded", "extracted_at", "raw", "extractor_note"}:
                continue
            current = merged.get(key)
            if value in (None, "", [], {}):
                continue
            if current in (None, "", [], {}):
                merged[key] = value
        return ContentEnvelope(**merged)

    def is_usable(self) -> bool:
        """Whether this envelope carries enough to be worth accepting from a tier.

        Deliberately permissive: a title alone is a real result. Only a bare URL
        with nothing attached counts as unusable.
        """
        return bool(self.text_parts) or bool(self.thumbnail_url) or bool(self.media_url)


class ExtractionAttempt(BaseModel):
    """Telemetry for one tier's turn. Persisted for every attempt, win or lose.

    After a hundred real links these rows tell you exactly which tiers are
    carrying the system and which you are paying for without benefit.
    """

    tier: ExtractionTier
    ok: bool
    skipped: bool = False
    reason: str | None = None
    duration_ms: int = 0
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExtractionResult(BaseModel):
    """What the cascade returns: an envelope plus the story of how it got there."""

    envelope: ContentEnvelope
    attempts: list[ExtractionAttempt] = Field(default_factory=list)
    cache_hit: bool = False

    @property
    def winning_tier(self) -> ExtractionTier:
        return self.envelope.tier

    @property
    def degraded(self) -> bool:
        return self.envelope.degraded


class ExtractionTarget(BaseModel):
    """A canonicalized request to extract, handed to each tier in turn."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    canonical_url: str
    url_hash: str
    platform: Platform
    original_url: str
    # Pre-extracted content supplied by the caller -- an iOS share payload, a
    # browser extension, anything with a real session. Present means tier 1 has
    # something to work with.
    client_payload: dict[str, Any] | None = None

    def envelope(self, tier: ExtractionTier, **fields: Any) -> ContentEnvelope:
        """Build an envelope pre-stamped with this target's identity."""
        return ContentEnvelope(
            canonical_url=self.canonical_url,
            url_hash=self.url_hash,
            original_url=self.original_url,
            platform=self.platform,
            tier=tier,
            **fields,
        )


def content_hash(text: str) -> str:
    """Stable hash helper, used for URL hashing and change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
