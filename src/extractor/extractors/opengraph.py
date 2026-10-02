"""Tier 4 -- plain HTTP: oEmbed, OpenGraph tags, and readable article text.

The last tier before degraded, and the one that carries the whole generic web. Three
sources, tried cheapest first and merged:

1. oEmbed. Several platforms expose an unauthenticated endpoint that hands over
   title, author and thumbnail with no scraping at all -- TikTok and YouTube both do.
   Free, stable, and rude not to use. Instagram's equivalent now requires an app
   token and a review process, so it is deliberately absent.
2. Reddit's `.json`. Append it to any post URL and you get structured JSON including
   the self-text and top comments. Genuinely free and stable, which makes Reddit the
   right platform to build the rest of the pipeline against first.
3. Direct fetch, then OpenGraph / Twitter Card / JSON-LD tags plus naive main-content
   extraction.

This tier will not get past a JavaScript challenge or a login wall -- that is what
tier 3 is for.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urlparse

import httpx

from ..base import Extractor
from ..envelope import (
    ContentEnvelope,
    ExtractionTarget,
    ExtractionTier,
    MediaKind,
    Platform,
)
from ..html_parse import (
    clean_text,
    extract_article_text,
    extract_hashtags,
    parse_meta,
)
from ..net_guard import UnsafeUrl, assert_public_url
from ..normalize import to_datetime, to_float, to_int, to_str, to_tags

_OEMBED_ENDPOINTS: dict[Platform, str] = {
    Platform.YOUTUBE: "https://www.youtube.com/oembed",
    Platform.TIKTOK: "https://www.tiktok.com/oembed",
    Platform.TWITTER: "https://publish.twitter.com/oembed",
}

_MAX_BYTES = 3_000_000  # a 3 MB page is already pathological; stop reading

# A blocked or JS-shell page still returns HTTP 200 with a title. Left unchecked,
# that title becomes a "successful" extraction, poisons your telemetry and hides the
# fact that the URL needs a better tier. Verified live: a Reddit post fetched
# server-side returns 200 with an 8 KB shell whose only metadata is `<title>Reddit`.
_JUNK_TITLES = {
    "reddit", "instagram", "tiktok", "x", "twitter", "linkedin", "facebook",
    "youtube", "login", "log in", "sign in", "just a moment...", "access denied",
    "attention required!", "are you a robot?", "robot check", "security check",
    "page not found", "error", "forbidden", "captcha", "bot verification",
    "verify you are human", "one moment, please...",
}

_BLOCK_MARKERS = (
    "cf-browser-verification",
    "challenge-platform",
    "captcha-delivery",
    "px-captcha",
    "please enable javascript and cookies",
    "checking if the site connection is secure",
)


# Medium's article pages sit behind Cloudflare and answer a server-side fetch with 403 and
# an "Attention Required!" interstitial -- verified live, and the reason a Medium essay was
# saved with a completely empty envelope. Its RSS feed is not guarded and returns 200 with
# the whole post in `content:encoded`, so the feed is the way in.
_MEDIUM_NS = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}
# Trailing hex id on a Medium slug, which also appears in the item's guid
# (`https://medium.com/p/900ce1f65ece`) and is the reliable way to match the two.
_MEDIUM_POST_ID = re.compile(r"-([0-9a-f]{6,20})/?$")


def _medium_feed_url(canonical_url: str) -> str | None:
    """The author or publication feed that should contain this post, if it is a Medium URL."""
    parsed = urlparse(canonical_url)
    netloc = parsed.netloc.lower()
    if not (netloc == "medium.com" or netloc.endswith(".medium.com")):
        return None

    # Custom-domain Medium blogs are indistinguishable from any other site without
    # fetching first, so they are left to the normal HTML path.
    if netloc.endswith(".medium.com"):
        subdomain = netloc[: -len(".medium.com")]
        if subdomain and subdomain != "www":
            return f"https://{netloc}/feed"

    segments = [s for s in (parsed.path or "").split("/") if s]
    if not segments:
        return None
    if segments[0].startswith("@"):
        return f"https://medium.com/feed/{segments[0]}"
    # `medium.com/<publication>/<slug>` -- a bare `/<slug>` has no feed to consult.
    if len(segments) >= 2:
        return f"https://medium.com/feed/{segments[0]}"
    return None


def _looks_blocked(html: str, title: str | None) -> bool:
    if title and title.strip().lower().strip(" .|-") in _JUNK_TITLES:
        return True
    lowered = html[:20_000].lower()
    return any(marker in lowered for marker in _BLOCK_MARKERS)


class OpenGraphExtractor(Extractor):
    tier = ExtractionTier.OPEN_GRAPH
    name = "open-graph"

    def extract(self, target: ExtractionTarget) -> ContentEnvelope | None:
        envelope: ContentEnvelope | None = None
        notes: list[str] = []

        if target.platform is Platform.REDDIT:
            envelope = self._from_reddit_json(target)
            if envelope:
                notes.append("reddit .json")

        medium_fields = self._from_medium_feed(target)
        if medium_fields:
            notes.append("medium rss")
            candidate = target.envelope(self.tier, **medium_fields)
            envelope = candidate if envelope is None else envelope.merged_with(candidate)

        oembed_fields = self._from_oembed(target)
        if oembed_fields:
            notes.append("oembed")
            candidate = target.envelope(self.tier, **oembed_fields)
            envelope = candidate if envelope is None else envelope.merged_with(candidate)

        html_fields = self._from_html(target)
        if html_fields:
            notes.append("og/html")
            candidate = target.envelope(self.tier, **html_fields)
            envelope = candidate if envelope is None else envelope.merged_with(candidate)

        if envelope is None or not envelope.is_usable():
            return None

        envelope.tier = self.tier
        envelope.extractor_note = " + ".join(notes) or "og/html"
        if envelope.media_kind is MediaKind.UNKNOWN:
            envelope.media_kind = _guess_kind(envelope, target.platform)
        return envelope

    # ------------------------------------------------------------------ oEmbed
    def _from_oembed(self, target: ExtractionTarget) -> dict[str, Any]:
        endpoint = _OEMBED_ENDPOINTS.get(target.platform)
        if not endpoint:
            return {}
        try:
            with httpx.Client(
                timeout=self.config.http_timeout,
                follow_redirects=True,
                headers={"User-Agent": self.config.user_agent},
            ) as client:
                response = client.get(
                    endpoint,
                    params={"url": target.canonical_url, "format": "json"},
                )
            if response.status_code != 200:
                return {}
            data = response.json()
        except Exception:
            return {}

        if not isinstance(data, dict):
            return {}

        fields: dict[str, Any] = {
            "title": clean_text(to_str(data.get("title"))),
            "author": to_str(data.get("author_name")),
            "author_url": to_str(data.get("author_url")),
            "thumbnail_url": to_str(data.get("thumbnail_url")),
        }
        # X/Twitter oEmbed returns the tweet body inside an HTML blockquote.
        if target.platform is Platform.TWITTER and data.get("html"):
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(str(data["html"]), "html.parser")
            block = soup.find("blockquote")
            text = clean_text(block.get_text(" ", strip=True)) if block else None
            if text:
                fields["caption"] = text
        return {k: v for k, v in fields.items() if v}

    # ------------------------------------------------------------------ Medium
    def _from_medium_feed(self, target: ExtractionTarget) -> dict[str, Any]:
        """Recover a Medium post from its author/publication RSS feed.

        The feed carries only the most recent posts -- 7 for the account this was verified
        against -- so an older article still falls through to the HTML path and degrades.
        That is a real limit, not an oversight: there is no free, unauthenticated route to
        Medium's archive, and a partial win on recently-shared links is the common case.
        """
        feed_url = _medium_feed_url(target.canonical_url)
        if not feed_url:
            return {}

        match = _MEDIUM_POST_ID.search(urlparse(target.canonical_url).path)
        post_id = match.group(1) if match else None

        try:
            assert_public_url(feed_url)
        except UnsafeUrl:
            return {}

        try:
            with httpx.Client(
                timeout=self.config.http_timeout,
                follow_redirects=True,
                headers={"User-Agent": self.config.user_agent},
            ) as client:
                response = client.get(feed_url)
            if response.status_code != 200:
                return {}
            root = ET.fromstring(response.content)
        except Exception:
            # Malformed XML, network failure, or a feed that does not exist for this path.
            return {}

        item = self._match_medium_item(root, post_id, target.canonical_url)
        if item is None:
            return {}

        def text_of(tag: str) -> str | None:
            node = item.find(tag, _MEDIUM_NS)
            return to_str(node.text) if node is not None else None

        fields: dict[str, Any] = {}
        title = clean_text(text_of("title"))
        if title:
            fields["title"] = title
        author = to_str(text_of("dc:creator"))
        if author:
            fields["author"] = author

        published = to_datetime(text_of("pubDate"))
        if published:
            fields["published_at"] = published

        body_html = text_of("content:encoded")
        if body_html:
            article = extract_article_text(body_html, self.config.max_article_chars)
            # `content:encoded` is a bare HTML fragment with no <article> wrapper, so the
            # main-content heuristic can come back empty. Falling back to the fragment's own
            # text is correct here: unlike a full page there is no chrome to strip out.
            if not article:
                from bs4 import BeautifulSoup

                article = clean_text(
                    BeautifulSoup(body_html, "html.parser").get_text(" ", strip=True)
                )
            if article:
                fields["article_text"] = article[: self.config.max_article_chars]
            thumbnail = self._first_image(body_html)
            if thumbnail:
                fields["thumbnail_url"] = thumbnail

        tags = to_tags(
            [to_str(node.text) for node in item.findall("category") if node.text]
        )
        if tags:
            fields["platform_tags"] = tags
        return fields

    @staticmethod
    def _match_medium_item(
        root: ET.Element, post_id: str | None, canonical_url: str
    ) -> ET.Element | None:
        """Find the feed entry for this specific post.

        Matched on the post id rather than the slug: Medium rewrites slugs when a title is
        edited, but the trailing id is stable and appears in every item's guid.
        """
        items = root.findall("./channel/item")
        wanted_path = urlparse(canonical_url).path.rstrip("/").lower()
        for item in items:
            for tag in ("guid", "link"):
                node = item.find(tag)
                value = (node.text or "").lower() if node is not None else ""
                if not value:
                    continue
                if post_id and post_id in value:
                    return item
                if wanted_path and urlparse(value).path.rstrip("/").lower() == wanted_path:
                    return item
        return None

    @staticmethod
    def _first_image(html_fragment: str) -> str | None:
        from bs4 import BeautifulSoup

        img = BeautifulSoup(html_fragment, "html.parser").find("img")
        if img is None:
            return None
        src = img.get("src")
        return to_str(src) if isinstance(src, str) and src.startswith("http") else None

    # ------------------------------------------------------------------ Reddit
    def _from_reddit_json(self, target: ExtractionTarget) -> ContentEnvelope | None:
        """Fetch a post via Reddit's `.json`, preferring the `old.` host.

        Verified against live Reddit: `www.reddit.com/<path>/.json` now returns 403
        for every User-Agent tried, including a polite descriptive one, while
        `old.reddit.com/<path>/.json` still returns the same JSON with 200. So the
        old host is tried first and www is kept only as a fallback in case that
        flips back.
        """
        path = target.canonical_url.split("reddit.com", 1)[-1].rstrip("/")
        candidates = [
            f"https://old.reddit.com{path}/.json",
            f"https://www.reddit.com{path}/.json",
        ]

        data: Any = None
        for url in candidates:
            try:
                with httpx.Client(
                    timeout=self.config.http_timeout,
                    follow_redirects=True,
                    headers={
                        "User-Agent": self.config.user_agent,
                        "Accept": "application/json",
                    },
                ) as client:
                    response = client.get(url, params={"limit": 10, "raw_json": 1})
                if response.status_code != 200:
                    continue
                data = response.json()
            except Exception:
                continue
            if isinstance(data, list) and data:
                break
            data = None

        if not isinstance(data, list) or not data:
            return None

        try:
            post = data[0]["data"]["children"][0]["data"]
        except (KeyError, IndexError, TypeError):
            return None

        selftext = clean_text(to_str(post.get("selftext")))
        comments = _reddit_top_comments(data[1] if len(data) > 1 else None)

        body_parts = []
        if selftext:
            body_parts.append(selftext)
        if comments:
            # Top comments are high-signal for recall -- often how someone remembers
            # a thread ("the one where someone explained X").
            body_parts.append("Top comments:\n" + "\n".join(f"- {c}" for c in comments))
        article_text = "\n\n".join(body_parts) or None

        thumbnail = to_str(post.get("thumbnail"))
        if thumbnail in {"self", "default", "nsfw", "spoiler", "image"}:
            thumbnail = None
        if not thumbnail:
            try:
                thumbnail = to_str(
                    post["preview"]["images"][0]["source"]["url"]
                )
            except (KeyError, IndexError, TypeError):
                thumbnail = None

        subreddit = to_str(post.get("subreddit"))
        tags = to_tags([subreddit] if subreddit else [])
        flair = to_str(post.get("link_flair_text"))
        if flair:
            tags += to_tags(flair)

        fields: dict[str, Any] = {
            "title": clean_text(to_str(post.get("title"))),
            "author": to_str(post.get("author")),
            "author_url": (
                f"https://www.reddit.com/user/{post['author']}"
                if post.get("author")
                else None
            ),
            "caption": selftext[:500] if selftext else None,
            "article_text": article_text,
            "thumbnail_url": thumbnail,
            "published_at": to_datetime(post.get("created_utc")),
            "like_count": to_int(post.get("ups")),
            "comment_count": to_int(post.get("num_comments")),
            "platform_tags": tags,
            "media_kind": MediaKind.VIDEO if post.get("is_video") else MediaKind.TEXT,
        }
        if post.get("is_video"):
            try:
                fields["media_url"] = to_str(
                    post["media"]["reddit_video"]["fallback_url"]
                )
                fields["duration_s"] = to_float(
                    post["media"]["reddit_video"].get("duration")
                )
            except (KeyError, TypeError):
                pass
        # A link post points somewhere else; keep the destination as media_url so a
        # later pass can follow it.
        outbound = to_str(post.get("url_overridden_by_dest"))
        if outbound and not fields.get("media_url"):
            fields["media_url"] = outbound

        fields = {k: v for k, v in fields.items() if v not in (None, "", [])}
        envelope = target.envelope(self.tier, **fields)
        return envelope if envelope.is_usable() else None

    # -------------------------------------------------------------------- HTML
    def _from_html(self, target: ExtractionTarget) -> dict[str, Any]:
        try:
            assert_public_url(target.canonical_url)
        except UnsafeUrl:
            # Refuse to fetch internal addresses on behalf of a caller-supplied URL.
            return {}

        try:
            with httpx.Client(
                timeout=self.config.http_timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": self.config.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            ) as client:
                response = client.get(target.canonical_url)
                if response.status_code >= 400:
                    return {}
                content_type = response.headers.get("content-type", "")
                if "html" not in content_type and "xml" not in content_type:
                    return {}
                html = response.text[:_MAX_BYTES]
                final_url = str(response.url)
        except Exception:
            return {}

        meta = parse_meta(html, base_url=final_url)
        if _looks_blocked(html, to_str(meta.get("title"))):
            # Report nothing rather than a bot wall's title. The cascade then records
            # this tier as a genuine failure, which is what you want to see in stats.
            return {}

        fields: dict[str, Any] = {
            key: meta[key]
            for key in ("title", "caption", "author", "thumbnail_url", "media_url")
            if meta.get(key)
        }
        if meta.get("duration_s"):
            fields["duration_s"] = to_float(meta["duration_s"])
        if meta.get("published_at_raw"):
            published = to_datetime(meta["published_at_raw"])
            if published:
                fields["published_at"] = published

        article = meta.get("article_text") or extract_article_text(
            html, self.config.max_article_chars
        )
        if article:
            fields["article_text"] = article

        tags = to_tags(extract_hashtags(fields.get("caption"), fields.get("title")))
        if tags:
            fields["platform_tags"] = tags

        return fields


def _reddit_top_comments(listing: Any, limit: int = 5) -> list[str]:
    if not isinstance(listing, dict):
        return []
    children = (listing.get("data") or {}).get("children") or []
    comments: list[str] = []
    for child in children:
        if not isinstance(child, dict) or child.get("kind") != "t1":
            continue
        body = clean_text(to_str((child.get("data") or {}).get("body")))
        if not body or body in {"[deleted]", "[removed]"}:
            continue
        comments.append(body[:600])
        if len(comments) >= limit:
            break
    return comments


def _guess_kind(envelope: ContentEnvelope, platform: Platform) -> MediaKind:
    if envelope.media_url or envelope.duration_s:
        return MediaKind.VIDEO
    if platform in {Platform.INSTAGRAM, Platform.TIKTOK, Platform.YOUTUBE}:
        return MediaKind.VIDEO
    if envelope.article_text:
        return MediaKind.ARTICLE
    if envelope.thumbnail_url:
        return MediaKind.IMAGE
    return MediaKind.UNKNOWN
