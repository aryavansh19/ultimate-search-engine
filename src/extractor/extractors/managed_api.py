"""Tier 3 -- a third-party scraper API, behind a provider-agnostic adapter.

This is the tier you pay for, and the reason it exists is Instagram. Blocking there
is four independent layers deep -- TLS handshake fingerprint, HTTP/2 frame ordering,
a JavaScript challenge, and IP reputation -- and header spoofing defeats none of
them. Providers like Bright Data, Apify, Scrape Creators and HikerAPI maintain that
bypass full time on residential proxy pools. Rebuilding it yourself is weeks of work
that breaks monthly.

Unconfigured, this tier reports itself unavailable and the cascade *skips* it. That
keeps your success-rate telemetry honest: a provider you have not signed up for is
not a failed extraction.

Response mapping is deliberately loose. Every provider names fields differently, so
the adapter searches a nested JSON body for recognizable keys rather than assuming
one provider's schema.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..base import Extractor
from ..envelope import (
    ContentEnvelope,
    ExtractionTarget,
    ExtractionTier,
    MediaKind,
    Platform,
)
from ..html_parse import clean_text, extract_hashtags
from ..normalize import pick, to_datetime, to_float, to_int, to_str, to_tags

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "title": ("title", "name", "headline"),
    "caption": ("caption", "description", "text", "edge_media_to_caption", "post_text"),
    "author": ("author", "username", "owner_username", "uploader", "channel", "user_name"),
    "author_url": ("author_url", "profile_url", "user_url", "channel_url"),
    "thumbnail_url": (
        "thumbnail_url", "thumbnail", "display_url", "image_url", "cover", "cover_url",
    ),
    "media_url": ("media_url", "video_url", "play_url", "download_url", "src"),
    "transcript": ("transcript", "captions_text", "subtitles_text", "audio_transcript"),
    "ocr_text": ("ocr_text", "ocr", "text_in_image", "onscreen_text"),
    "article_text": ("article_text", "content", "body", "main_text", "markdown"),
}

_COUNT_ALIASES: dict[str, tuple[str, ...]] = {
    "like_count": ("like_count", "likes", "likes_count", "edge_liked_by"),
    "view_count": ("view_count", "views", "play_count", "video_view_count"),
    "comment_count": ("comment_count", "comments", "comments_count"),
}


class ManagedApiExtractor(Extractor):
    tier = ExtractionTier.MANAGED_API
    name = "managed-api"

    def availability(self) -> tuple[bool, str | None]:
        if not self.config.managed_api.enabled:
            return False, "MANAGED_API_URL not configured"
        return True, None

    def can_handle(self, target: ExtractionTarget) -> bool:
        # Providers charge per request. Route only the platforms that actually need
        # them -- paying to scrape a public blog post is money set on fire.
        return self.config.managed_api.handles(target.platform.value)

    def extract(self, target: ExtractionTarget) -> ContentEnvelope | None:
        settings = self.config.managed_api
        headers = {"Accept": "application/json", "User-Agent": self.config.user_agent}
        if settings.api_key:
            prefix = f"{settings.auth_prefix} " if settings.auth_prefix else ""
            headers[settings.auth_header] = f"{prefix}{settings.api_key}"

        with httpx.Client(timeout=self.config.http_timeout, follow_redirects=True) as client:
            if settings.method == "GET":
                response = client.get(
                    str(settings.url),
                    params={settings.url_param: target.canonical_url},
                    headers=headers,
                )
            else:
                response = client.post(
                    str(settings.url),
                    json={settings.url_param: target.canonical_url},
                    headers=headers,
                )
            response.raise_for_status()
            body = response.json()

        flat = _flatten(body)
        if not flat:
            return None

        fields: dict[str, Any] = {}
        for field, keys in _FIELD_ALIASES.items():
            value = to_str(pick(flat, *keys))
            if value:
                fields[field] = clean_text(value)

        for field, keys in _COUNT_ALIASES.items():
            count = to_int(pick(flat, *keys))
            if count is not None:
                fields[field] = count

        duration = to_float(pick(flat, "duration_s", "duration", "video_duration", "length"))
        if duration:
            fields["duration_s"] = duration

        published = to_datetime(
            pick(flat, "published_at", "taken_at_timestamp", "timestamp", "created_at", "date")
        )
        if published:
            fields["published_at"] = published

        tags = to_tags(pick(flat, "platform_tags", "hashtags", "tags", "keywords"))
        tags += [t for t in extract_hashtags(fields.get("caption")) if t not in tags]
        if tags:
            fields["platform_tags"] = tags

        if target.platform in {Platform.INSTAGRAM, Platform.TIKTOK, Platform.YOUTUBE}:
            fields.setdefault("media_kind", MediaKind.VIDEO)

        envelope = target.envelope(self.tier, **fields)
        envelope.extractor_note = f"managed api: {settings.url}"
        return envelope if envelope.is_usable() else None


def _flatten(data: Any, depth: int = 0) -> dict[str, Any]:
    """Collapse a nested provider response into one lowercase key -> scalar map.

    First value wins, so shallower keys take precedence over deeply nested ones.
    Crude, but it means supporting a new provider is usually a config change rather
    than a code change.
    """
    out: dict[str, Any] = {}
    if depth > 6:
        return out

    if isinstance(data, dict):
        for key, value in data.items():
            lowered = str(key).lower()
            if isinstance(value, (str, int, float)) and value not in (None, ""):
                out.setdefault(lowered, value)
            elif isinstance(value, list) and value and all(
                isinstance(v, str) for v in value
            ):
                out.setdefault(lowered, value)
            else:
                for nested_key, nested_value in _flatten(value, depth + 1).items():
                    out.setdefault(nested_key, nested_value)
    elif isinstance(data, list):
        for item in data[:5]:
            for nested_key, nested_value in _flatten(item, depth + 1).items():
                out.setdefault(nested_key, nested_value)
    return out
