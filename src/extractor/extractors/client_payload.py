"""Tier 1 -- content the caller already extracted. The endgame path.

This tier exists to make on-device extraction a drop-in, not a rewrite. On iOS the
share sheet hands you more than a bare URL, and a WKWebView carrying the user's own
logged-in Instagram session can read a page that no server-side scraper will ever
reach: real device, real IP, real session, nothing to detect.

So the ingest API accepts a payload, not just a URL. Today that payload is usually
empty and the server-side tiers below do the work. The day the iOS client starts
filling it in, this tier wins first and every tier below becomes a fallback --
with no changes to enrichment, embedding, indexing or search.

Accepted payload shapes, in order of preference:

  {"envelope": {...}}            a full ContentEnvelope, passed straight through
  {"html": "<!doctype html>..."} rendered DOM from a real session; parsed here
  {"caption": "...", ...}        flat fields, with generous key aliasing
"""

from __future__ import annotations

from typing import Any

from ..base import Extractor
from ..envelope import (
    ContentEnvelope,
    ExtractionTarget,
    ExtractionTier,
    MediaKind,
    Platform,
)
from ..html_parse import extract_article_text, extract_hashtags, parse_meta
from ..normalize import pick, to_datetime, to_float, to_int, to_str, to_tags

# Keys iOS share extensions, browser extensions and hand-rolled clients actually
# send. Being liberal here costs nothing and saves the client from having to know
# our internal field names.
_ALIASES: dict[str, tuple[str, ...]] = {
    "title": ("title", "name", "headline", "og_title", "public.plain-text"),
    "caption": (
        "caption", "description", "text", "body", "snippet", "summary",
        "og_description", "public.text", "sharedtext", "message",
    ),
    "author": ("author", "username", "handle", "owner", "creator", "channel", "uploader"),
    "author_url": ("author_url", "profile_url", "channel_url", "uploader_url"),
    "thumbnail_url": ("thumbnail_url", "thumbnail", "image", "poster", "preview", "cover"),
    "media_url": ("media_url", "video_url", "videourl", "url_media", "stream_url"),
    "transcript": ("transcript", "captions", "subtitles", "speech", "audio_text"),
    "ocr_text": ("ocr_text", "ocr", "onscreen_text", "screen_text", "visiontext"),
    "article_text": ("article_text", "content", "article", "readable_text", "maintext"),
    "visual_description": ("visual_description", "visual", "scene_description"),
}


class ClientPayloadExtractor(Extractor):
    tier = ExtractionTier.CLIENT_PAYLOAD
    name = "client-payload"

    def can_handle(self, target: ExtractionTarget) -> bool:
        return bool(target.client_payload)

    def extract(self, target: ExtractionTarget) -> ContentEnvelope | None:
        payload = target.client_payload or {}
        if not payload:
            return None

        # Shape 1: a complete envelope. Trust the identity fields we computed, not
        # the client's -- canonicalization must stay authoritative server-side.
        if isinstance(payload.get("envelope"), dict):
            data = dict(payload["envelope"])
            data.update(
                canonical_url=target.canonical_url,
                url_hash=target.url_hash,
                original_url=target.original_url,
                platform=target.platform,
                tier=self.tier,
            )
            data.pop("degraded", None)
            envelope = ContentEnvelope(**data)
            envelope.extractor_note = "client-supplied envelope"
            return envelope if envelope.is_usable() else None

        fields: dict[str, Any] = {}

        # Shape 2: rendered HTML from a session-bearing webview.
        html = to_str(pick(payload, "html", "dom", "document", "outerhtml"))
        if html:
            meta = parse_meta(html, base_url=target.canonical_url)
            for key in (
                "title", "caption", "author", "thumbnail_url", "media_url",
            ):
                if meta.get(key):
                    fields[key] = meta[key]
            if meta.get("duration_s"):
                fields["duration_s"] = to_float(meta["duration_s"])
            if meta.get("published_at_raw"):
                fields["published_at"] = to_datetime(meta["published_at_raw"])
            article = meta.get("article_text") or extract_article_text(
                html, self.config.max_article_chars
            )
            if article:
                fields["article_text"] = article

        # Shape 3: flat fields. These win over anything scraped from the HTML,
        # since an explicit client field is a stronger signal than a meta tag.
        for field, keys in _ALIASES.items():
            value = to_str(pick(payload, *keys))
            if value:
                fields[field] = value

        duration = to_float(pick(payload, "duration_s", "duration", "length"))
        if duration:
            fields["duration_s"] = duration

        published = to_datetime(
            pick(payload, "published_at", "timestamp", "taken_at", "created_at", "date")
        )
        if published:
            fields["published_at"] = published

        for field, keys in (
            ("like_count", ("like_count", "likes", "favorite_count")),
            ("view_count", ("view_count", "views", "play_count")),
            ("comment_count", ("comment_count", "comments")),
        ):
            count = to_int(pick(payload, *keys))
            if count is not None:
                fields[field] = count

        tags = to_tags(pick(payload, "platform_tags", "tags", "hashtags", "keywords"))
        tags += [t for t in extract_hashtags(fields.get("caption")) if t not in tags]
        if tags:
            fields["platform_tags"] = tags

        fields["media_kind"] = _guess_media_kind(target.platform, fields)

        envelope = target.envelope(self.tier, **fields)
        envelope.extractor_note = "client payload" + (" + parsed DOM" if html else "")
        return envelope if envelope.is_usable() else None


def _guess_media_kind(platform: Platform, fields: dict[str, Any]) -> MediaKind:
    if fields.get("media_url") or fields.get("transcript"):
        return MediaKind.VIDEO
    if platform in {Platform.INSTAGRAM, Platform.TIKTOK, Platform.YOUTUBE}:
        return MediaKind.VIDEO
    if fields.get("article_text"):
        return MediaKind.ARTICLE
    if fields.get("thumbnail_url"):
        return MediaKind.IMAGE
    return MediaKind.UNKNOWN
