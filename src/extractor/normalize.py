"""Small, forgiving coercion helpers.

Every extractor receives data from a source that formats things its own way:
epoch seconds, ISO strings, `YYYYMMDD`, numbers as strings, nulls as empty
strings. These helpers absorb that variance and never raise -- a malformed
timestamp must not be able to fail an extraction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, Mapping


def to_datetime(value: Any) -> datetime | None:
    """Parse epoch numbers, ISO-8601 strings, RFC 822 dates and yt-dlp's YYYYMMDD."""
    if value in (None, "", 0):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None

    if text.isdigit():
        if len(text) == 8:  # YYYYMMDD
            try:
                return datetime.strptime(text, "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return to_datetime(int(text))

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            # RFC 822, e.g. "Mon, 31 Aug 2026 10:42:27 GMT". This is the date format RSS
            # mandates, so every feed source depends on it -- without it a Medium post
            # parsed cleanly in every field except its publication date.
            try:
                parsed = parsedate_to_datetime(text)
            except (TypeError, ValueError):
                return None
            if parsed is None:
                return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    return int(number) if number is not None else None


def to_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        parts = [to_str(v) for v in value]
        joined = " ".join(p for p in parts if p)
        return joined or None
    text = str(value).strip()
    return text or None


def to_tags(value: Any) -> list[str]:
    """Normalize a tag-ish value into a lowercase, de-duplicated list.

    Accepts a list, a comma-separated string, or a space-separated hashtag blob.
    Normalizing here rather than at the tagging stage is what prevents the
    'cooking'/'Cooking'/'#cooking' drift problem later.
    """
    if not value:
        return []
    if isinstance(value, str):
        raw: Iterable[Any] = value.replace("#", " ").replace(",", " ").split()
    elif isinstance(value, (list, tuple, set)):
        raw = value
    else:
        return []

    seen: dict[str, None] = {}
    for item in raw:
        text = to_str(item)
        if not text:
            continue
        slug = text.strip().lstrip("#").strip().lower()
        if slug:
            seen.setdefault(slug, None)
    return list(seen)


def pick(source: Mapping[str, Any], *keys: str) -> Any:
    """First present, non-empty value among `keys`. Case-insensitive on keys."""
    lowered = {str(k).lower(): v for k, v in source.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value not in (None, "", [], {}):
            return value
    return None
