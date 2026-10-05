"""Tier 2 -- yt-dlp. The single highest-leverage extractor in the chain.

One interface covering Instagram, TikTok, YouTube, X, Reddit and roughly a thousand
other sites, returning structured metadata without downloading media. It also
accepts cookies, which is what makes Instagram work during local development
against your own browser profile.

Two things worth knowing:

* Subtitles are fetched here when the platform offers them. That matters more than
  it looks: a real transcript pushes the envelope to STRONG signal, which means the
  enrichment stage skips the multimodal video call entirely. For YouTube this turns
  a ~2 cent item into a ~0.02 cent one.
* Extractor availability is decided by asking yt-dlp itself which of its site
  extractors claim the URL, excluding the catch-all `generic` one. That keeps
  Vimeo, Twitch, Dailymotion and the rest working without maintaining a host list,
  and avoids burning a few seconds running yt-dlp against plain blog posts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from ..base import Extractor, ExtractorUnavailable
from ..envelope import (
    ContentEnvelope,
    ExtractionTarget,
    ExtractionTier,
    MediaKind,
)
from ..html_parse import clean_text, extract_hashtags
from ..normalize import to_datetime, to_float, to_int, to_str, to_tags

log = logging.getLogger("extractor.ytdlp")

_SUBTITLE_LANG_PREFERENCE = ("en", "en-US", "en-GB", "en-orig", "en-auto")
_SUBTITLE_EXT_PREFERENCE = ("json3", "vtt", "srv1", "ttml")
_VTT_TIMESTAMP = re.compile(r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->")
_VTT_TAG = re.compile(r"<[^>]+>")


class _SilentLogger:
    """Swallow yt-dlp's own output; the cascade records failures itself."""

    def debug(self, message: str) -> None:
        log.debug("yt-dlp: %s", message)

    def info(self, message: str) -> None:
        log.debug("yt-dlp: %s", message)

    def warning(self, message: str) -> None:
        log.debug("yt-dlp warning: %s", message)

    def error(self, message: str) -> None:
        log.debug("yt-dlp error: %s", message)


class YtDlpExtractor(Extractor):
    tier = ExtractionTier.YT_DLP
    name = "yt-dlp"

    def availability(self) -> tuple[bool, str | None]:
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            return False, "yt-dlp not installed"
        return True, None

    def can_handle(self, target: ExtractionTarget) -> bool:
        return _resolvable_url(target) is not None

    def extract(self, target: ExtractionTarget) -> ContentEnvelope | None:
        try:
            import yt_dlp
        except ImportError as exc:  # pragma: no cover - guarded by availability()
            raise ExtractorUnavailable("yt-dlp not installed") from exc

        fetch_url = _resolvable_url(target)
        if fetch_url is None:
            return None

        options: dict[str, Any] = {
            # `quiet` alone does not stop yt-dlp writing extractor errors to stderr.
            # Routing it at a logger keeps tier failures inside the cascade's
            # telemetry instead of scribbling on the caller's output.
            "logger": _SilentLogger(),
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noplaylist": True,
            "socket_timeout": self.config.http_timeout,
            "retries": 1,
            "extractor_retries": 1,
            "ignoreerrors": False,
            "nocheckcertificate": False,
            "http_headers": {"User-Agent": self.config.user_agent},
        }
        cookie_copy: str | None = None
        if self.config.cookie_file:
            cookie_copy = _private_cookie_copy(self.config.cookie_file)
            if cookie_copy:
                options["cookiefile"] = cookie_copy
        elif self.config.cookies_from_browser:
            options["cookiesfrombrowser"] = _parse_browser_spec(
                self.config.cookies_from_browser
            )

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(fetch_url, download=False)
        finally:
            if cookie_copy:
                _remove_quietly(cookie_copy)

        if not info:
            return None
        info = ydl.sanitize_info(info)

        # Carousels and playlists arrive as `entries`. Take the first item for media
        # fields but keep top-level text, which is usually the shared caption.
        entry = info
        entries = info.get("entries") or []
        if entries:
            first = next((e for e in entries if isinstance(e, dict)), None)
            if first:
                entry = {**first, **{k: v for k, v in info.items() if v and k != "entries"}}

        caption = clean_text(to_str(entry.get("description")))
        title = clean_text(to_str(entry.get("title")))
        if title and caption and _title_is_echoed_caption(title, caption):
            title = None

        tags = to_tags(entry.get("tags") or entry.get("categories"))
        tags += [t for t in extract_hashtags(caption, title) if t not in tags]

        transcript = self._fetch_transcript(entry)

        fields: dict[str, Any] = {
            "title": title,
            "caption": caption,
            "author": to_str(
                entry.get("uploader") or entry.get("channel") or entry.get("creator")
            ),
            "author_url": to_str(entry.get("uploader_url") or entry.get("channel_url")),
            "thumbnail_url": to_str(entry.get("thumbnail")),
            "media_url": _pick_media_url(entry),
            "duration_s": to_float(entry.get("duration")),
            "published_at": to_datetime(
                entry.get("timestamp") or entry.get("upload_date")
            ),
            "platform_tags": tags,
            "like_count": to_int(entry.get("like_count")),
            "view_count": to_int(entry.get("view_count")),
            "comment_count": to_int(entry.get("comment_count")),
            "transcript": transcript,
            "media_kind": _media_kind(entry),
        }
        fields = {k: v for k, v in fields.items() if v not in (None, "", [])}

        envelope = target.envelope(self.tier, **fields)
        envelope.extractor_note = f"yt-dlp:{entry.get('extractor_key') or 'unknown'}"
        # Keep a trimmed raw record. Useful when adding a platform, and it is the
        # only way to debug a mapping that silently produced nothing.
        envelope.raw = {
            k: v
            for k, v in entry.items()
            if k
            in {
                "id", "extractor_key", "webpage_url", "ext", "width", "height",
                "fps", "resolution", "availability", "live_status", "language",
            }
        }
        return envelope if envelope.is_usable() else None

    # ---------------------------------------------------------------- subtitles
    def _fetch_transcript(self, entry: dict[str, Any]) -> str | None:
        """Download and flatten the best available subtitle track.

        Manual subtitles beat auto-generated ones. Failures are swallowed: a
        missing transcript is normal, not an error.
        """
        for key in ("subtitles", "automatic_captions"):
            tracks = entry.get(key) or {}
            if not isinstance(tracks, dict):
                continue
            track = _pick_subtitle_track(tracks)
            if not track:
                continue
            url = to_str(track.get("url"))
            if not url:
                continue
            try:
                with httpx.Client(
                    timeout=self.config.http_timeout,
                    follow_redirects=True,
                    headers={"User-Agent": self.config.user_agent},
                ) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    body = response.text
            except Exception:
                continue
            text = _parse_subtitles(body, to_str(track.get("ext")))
            if text:
                return text
        return None


@lru_cache(maxsize=1)
def _site_extractor_classes() -> tuple[Any, ...]:
    """yt-dlp's site extractors, minus the catch-all generic one."""
    from yt_dlp.extractor import gen_extractor_classes

    return tuple(
        ie
        for ie in gen_extractor_classes()
        if getattr(ie, "IE_NAME", "").lower() not in {"generic", "unsupported"}
    )


def _has_site_extractor(url: str) -> bool:
    try:
        classes = _site_extractor_classes()
    except Exception:
        return True  # if introspection fails, let the tier try anyway
    for extractor_class in classes:
        try:
            if extractor_class.suitable(url):
                return True
        except Exception:
            continue
    return False


def _resolvable_url(target: ExtractionTarget) -> str | None:
    """Pick a URL form that one of yt-dlp's site extractors actually recognizes.

    Canonicalization strips `www.` for clean deduplication, but a lot of yt-dlp's
    extractor patterns require it -- `ted.com/talks/...` matches nothing while
    `www.ted.com/talks/...` matches fine. Without this the tier silently reports
    "not applicable" for whole sites, which looks like a deliberate skip in the
    telemetry rather than the bug it is.

    So dedup keeps the canonical form and extraction gets a form that works.
    """
    seen: list[str] = []
    for candidate in (
        target.canonical_url,
        target.original_url,
        _with_www(target.canonical_url),
    ):
        if not candidate or candidate in seen:
            continue
        seen.append(candidate)
        if _has_site_extractor(candidate):
            return candidate
    return None


def _with_www(url: str) -> str | None:
    parsed = urlparse(url)
    if not parsed.netloc or parsed.netloc.startswith("www."):
        return None
    return urlunparse(parsed._replace(netloc=f"www.{parsed.netloc}"))


_warned_cookie_paths: set[str] = set()


def _private_cookie_copy(path: str) -> str | None:
    """Copy the configured cookies file to a private temp file for one yt-dlp run.

    yt-dlp writes its cookie jar back to `cookiefile` when it closes. Pointing it at
    the original breaks in two ways: a Render secret file lives under /etc/secrets,
    which the app may not be able to write, so the write-back raises after the
    extraction already succeeded and the whole tier is recorded as failed; and
    concurrent requests would race on one file. A fresh owner-only copy per run
    avoids both. Cookie updates yt-dlp would write back are discarded; the login
    session itself stays valid until it expires or is logged out.

    A missing or unreadable file degrades to an anonymous run instead of failing.
    """
    try:
        fd, copy_path = tempfile.mkstemp(prefix="ytdlp-cookies-", suffix=".txt")
    except OSError as exc:
        log.warning("cannot create temp cookie copy (%s); continuing without cookies", exc)
        return None
    try:
        with os.fdopen(fd, "wb") as out, open(path, "rb") as src:
            shutil.copyfileobj(src, out)
    except OSError as exc:
        _remove_quietly(copy_path)
        if path not in _warned_cookie_paths:
            _warned_cookie_paths.add(path)
            log.warning("cookie file %s unreadable (%s); continuing without cookies",
                        path, exc)
        return None
    return copy_path


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _parse_browser_spec(spec: str) -> tuple[str | None, ...]:
    """Turn `chrome:Profile 1` into yt-dlp's (browser, profile, keyring, container)."""
    browser, _, profile = spec.partition(":")
    keyring = None
    if "+" in browser:
        browser, _, keyring = browser.partition("+")
    return (
        browser.strip().lower() or None,
        profile.strip() or None,
        keyring.strip().upper() if keyring else None,
        None,
    )


def _pick_subtitle_track(tracks: dict[str, Any]) -> dict[str, Any] | None:
    languages = list(tracks.keys())
    ordered = [lang for lang in _SUBTITLE_LANG_PREFERENCE if lang in tracks]
    ordered += [lang for lang in languages if lang.startswith("en") and lang not in ordered]
    ordered += [lang for lang in languages if lang not in ordered]
    for language in ordered:
        formats = tracks.get(language) or []
        if not isinstance(formats, list):
            continue
        for ext in _SUBTITLE_EXT_PREFERENCE:
            for fmt in formats:
                if isinstance(fmt, dict) and fmt.get("ext") == ext:
                    return fmt
        for fmt in formats:
            if isinstance(fmt, dict) and fmt.get("url"):
                return fmt
    return None


def _parse_subtitles(body: str, ext: str | None) -> str | None:
    """Flatten json3 or WebVTT/SRT into deduplicated plain text."""
    if ext == "json3" or body.lstrip().startswith("{"):
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            lines: list[str] = []
            for event in data.get("events") or []:
                segments = event.get("segs") or []
                text = "".join(seg.get("utf8", "") for seg in segments)
                text = text.replace("\n", " ").strip()
                if text:
                    lines.append(text)
            return _dedupe_lines(lines)

    lines = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line in {"WEBVTT"}:
            continue
        if line.isdigit() or "-->" in line or _VTT_TIMESTAMP.match(line):
            continue
        if line.startswith(("NOTE", "Kind:", "Language:", "STYLE")):
            continue
        cleaned = _VTT_TAG.sub("", line).strip()
        if cleaned:
            lines.append(cleaned)
    return _dedupe_lines(lines)


def _dedupe_lines(lines: list[str]) -> str | None:
    """Auto-captions repeat rolling text constantly; collapse consecutive repeats."""
    out: list[str] = []
    for line in lines:
        if out and (line == out[-1] or line in out[-1]):
            continue
        out.append(line)
    text = clean_text(" ".join(out))
    return text if text and len(text) > 20 else None


def _title_is_echoed_caption(title: str, caption: str) -> bool:
    """Whether the title is just the caption again, which is what Instagram and
    TikTok produce -- yt-dlp synthesizes a title by truncating the description.

    The test has to be narrow. "Is the title a prefix of the caption" is not enough:
    plenty of legitimate posts open the description with the title, and dropping the
    title there loses a genuinely useful field. Only treat it as an echo when the
    title covers most of the caption, or is visibly a truncation.
    """
    stripped = title.rstrip(". ").rstrip("\u2026")
    if not stripped or not caption.startswith(stripped):
        return False
    return title.endswith(("\u2026", "...")) or len(stripped) >= 0.8 * len(caption)


def _pick_media_url(entry: dict[str, Any]) -> str | None:
    """Find a genuinely playable media URL, or None.

    This used to fall back to `webpage_url`, which is actively harmful rather than
    merely useless. For Instagram, yt-dlp puts no top-level `url` on the entry, so the
    fallback handed downstream code the reel's *page* URL as though it were a video
    file. The enrichment stage then dutifully downloaded Instagram's HTML -- a login
    wall full of inline JavaScript -- uploaded it to a multimodal model, and billed
    292,000 input tokens to "analyze" a web page. Nothing errored; the tags just
    quietly came from page metadata instead of footage.

    So: only ever return something that came from a real format entry, and never the
    page URL.
    """
    webpage = to_str(entry.get("webpage_url"))
    best: str | None = None
    best_score = -1

    for fmt in entry.get("formats") or []:
        if not isinstance(fmt, dict):
            continue
        url = to_str(fmt.get("url"))
        if not url or url == webpage:
            continue
        # Manifests need a player to resolve; a plain GET returns XML or a playlist.
        if fmt.get("protocol") in {"m3u8", "m3u8_native", "http_dash_segments"}:
            continue
        if fmt.get("vcodec") in (None, "none"):
            continue

        score = int(fmt.get("height") or 0)
        if fmt.get("acodec") not in (None, "none"):
            score += 10_000  # muxed audio+video beats a video-only stream
        if fmt.get("ext") == "mp4":
            score += 1_000
        if score > best_score:
            best, best_score = url, score

    if best:
        return best

    direct = to_str(entry.get("url"))
    if direct and direct != webpage and "://" in direct:
        return direct
    return None


def _media_kind(entry: dict[str, Any]) -> MediaKind:
    if entry.get("duration") or entry.get("fps") or entry.get("vcodec") not in (
        None,
        "none",
    ):
        return MediaKind.VIDEO
    if entry.get("acodec") not in (None, "none"):
        return MediaKind.VIDEO
    if entry.get("thumbnail"):
        return MediaKind.IMAGE
    return MediaKind.UNKNOWN
