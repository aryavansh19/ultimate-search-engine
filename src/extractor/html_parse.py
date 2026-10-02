"""Shared HTML parsing helpers.

Used by two tiers for the same reason: both end up holding a page's HTML. The
OpenGraph tier fetched it over the network; the client-payload tier was handed it
by a WKWebView that already had the user's session. Same parsing either way,
which is exactly the symmetry the envelope contract is meant to preserve.

The article extraction here is intentionally naive -- enough for the common case,
not a Readability reimplementation. `trafilatura` is the upgrade when generic web
article quality starts to matter.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

_HASHTAG = re.compile(r"#([A-Za-z0-9_]{2,60})")
_WHITESPACE = re.compile(r"[ \t\xa0]+")
_BLANK_LINES = re.compile(r"\n{3,}")

_NOISE_TAGS = (
    "script", "style", "noscript", "template", "svg", "form", "nav",
    "footer", "header", "aside", "iframe", "button",
    # `sup` is where inline citation markers live. Leaving them in litters extracted
    # prose with "[ 1 ] [ 2 ]", which then reads as content to the tagging stage.
    "sup",
)

# og/twitter keys mapped onto envelope field names.
_META_MAP: dict[str, str] = {
    "og:title": "title",
    "twitter:title": "title",
    "og:description": "caption",
    "twitter:description": "caption",
    "description": "caption",
    "og:image": "thumbnail_url",
    "og:image:secure_url": "thumbnail_url",
    "twitter:image": "thumbnail_url",
    "twitter:image:src": "thumbnail_url",
    "og:video": "media_url",
    "og:video:url": "media_url",
    "og:video:secure_url": "media_url",
    "twitter:player:stream": "media_url",
    "og:video:duration": "duration_s",
    "video:duration": "duration_s",
    "article:published_time": "published_at_raw",
    "og:article:published_time": "published_at_raw",
    "article:author": "author",
    "og:site_name": "site_name",
    "og:type": "og_type",
}


def extract_hashtags(*texts: str | None) -> list[str]:
    """Hashtags, lowercased and de-duplicated, preserving first-seen order."""
    seen: dict[str, None] = {}
    for text in texts:
        if not text:
            continue
        for tag in _HASHTAG.findall(text):
            seen.setdefault(tag.lower(), None)
    return list(seen)


def clean_text(value: str | None) -> str | None:
    if not value:
        return None
    value = _WHITESPACE.sub(" ", value.replace("\r\n", "\n").replace("\r", "\n"))
    value = "\n".join(line.strip() for line in value.split("\n"))
    value = _BLANK_LINES.sub("\n\n", value).strip()
    return value or None


def parse_meta(html: str, base_url: str | None = None) -> dict[str, Any]:
    """Pull OpenGraph / Twitter Card / JSON-LD metadata out of a page.

    Returns a loose dict of envelope-ish field names. Callers decide what to keep;
    nothing here raises on malformed markup.
    """
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, Any] = {}

    for tag in soup.find_all("meta"):
        key = (tag.get("property") or tag.get("name") or "").strip().lower()
        content = (tag.get("content") or "").strip()
        if not key or not content:
            continue
        field = _META_MAP.get(key)
        if field and field not in found:
            found[field] = content

    if "title" not in found and soup.title and soup.title.string:
        found["title"] = soup.title.string.strip()

    canonical = soup.find("link", rel=lambda v: v and "canonical" in v)
    if canonical and canonical.get("href"):
        found["canonical_hint"] = canonical["href"].strip()

    jsonld = _parse_jsonld(soup)
    for key, value in jsonld.items():
        found.setdefault(key, value)

    if base_url:
        for field in ("thumbnail_url", "media_url", "canonical_hint"):
            if found.get(field):
                found[field] = urljoin(base_url, str(found[field]))

    if found.get("duration_s") is not None:
        found["duration_s"] = _coerce_duration(found["duration_s"])

    for field in ("title", "caption", "author"):
        if found.get(field):
            found[field] = clean_text(str(found[field]))

    return {k: v for k, v in found.items() if v not in (None, "")}


def _parse_jsonld(soup: BeautifulSoup) -> dict[str, Any]:
    """Best-effort JSON-LD scrape. Often the only place a real author name lives."""
    out: dict[str, Any] = {}
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text() or ""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for node in _iter_nodes(data):
            if not isinstance(node, dict):
                continue
            if node.get("headline") and "title" not in out:
                out["title"] = str(node["headline"])
            if node.get("datePublished") and "published_at_raw" not in out:
                out["published_at_raw"] = str(node["datePublished"])
            if node.get("uploadDate") and "published_at_raw" not in out:
                out["published_at_raw"] = str(node["uploadDate"])
            author = node.get("author")
            if author and "author" not in out:
                if isinstance(author, dict):
                    name = author.get("name")
                elif isinstance(author, list) and author:
                    first = author[0]
                    name = first.get("name") if isinstance(first, dict) else first
                else:
                    name = author
                if name:
                    out["author"] = str(name)
            if node.get("duration") and "duration_s" not in out:
                out["duration_s"] = node["duration"]
            body = node.get("articleBody")
            if body and "article_text" not in out:
                out["article_text"] = str(body)
    return out


def _iter_nodes(data: Any):
    if isinstance(data, dict):
        yield data
        for value in data.values():
            yield from _iter_nodes(value)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_nodes(item)


_ISO_DURATION = re.compile(
    r"^P(?:\d+D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?$", re.IGNORECASE
)


def _coerce_duration(value: Any) -> float | None:
    """Accept seconds-as-number, seconds-as-string, or ISO-8601 (PT1M30S)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    match = _ISO_DURATION.match(text)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    total = 0.0
    if hours:
        total += float(hours) * 3600
    if minutes:
        total += float(minutes) * 60
    if seconds:
        total += float(seconds)
    return total or None


def extract_article_text(html: str, max_chars: int = 20_000) -> str | None:
    """Rough main-content extraction: strip chrome, then take the best-scoring block.

    Candidates are scored on prose length discounted by link density, which is the
    one readability heuristic worth having: navigation sidebars, tag clouds and
    related-post rails are long but almost entirely anchor text, and a pure
    length score happily picks them over the actual article. Wikipedia is the
    clearest case -- its navboxes beat the body on raw character count.

    Good enough for blogs and news. Swap in trafilatura when it isn't.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    # Only ever strip container-level elements. Matching class names against every
    # node will hit <html> itself on real sites -- Wikipedia's root element carries
    # classes like `vector-feature-toc-pinned-clientpref-1` and
    # `vector-feature-main-menu-pinned-disabled`, and decomposing the root deletes
    # the entire document while leaving the code looking like it worked.
    for tag in soup.find_all(attrs={"role": ["navigation", "banner", "complementary"]}):
        if tag.name in _CONTAINER_TAGS:
            tag.decompose()
    for tag in soup.find_all(class_=_NOISE_CLASS):
        if tag.name in _CONTAINER_TAGS:
            tag.decompose()

    best_text: str | None = None
    best_score = 0.0

    for container in _candidate_containers(soup):
        text = _paragraph_text(container)
        if not text:
            continue
        score = _score(container, text)
        if score > best_score:
            best_text, best_score = text, score

    text = clean_text(best_text)
    if not text or len(text) < 200:
        return None
    return text[:max_chars]


# Elements safe to remove wholesale when they look like chrome. Deliberately excludes
# html, body, article and main.
_CONTAINER_TAGS = {
    "div", "section", "aside", "nav", "ul", "ol", "table", "td", "tr", "tbody",
    "span", "footer", "header", "form", "figure", "dl", "menu",
}

# Class/id fragments that reliably mark page chrome rather than content.
_NOISE_CLASS = re.compile(
    r"(navbox|sidebar|side-bar|menu|breadcrumb|toc\b|table-of-contents|"
    r"related|recommend|promo|advert|newsletter|subscribe|cookie|banner|"
    r"comment|social|share|footer|masthead|pagination|skip-link|hatnote|"
    r"metadata|catlinks|mw-editsection|reflist|references|citation|"
    r"navigation|infobox|authority-control)",
    re.IGNORECASE,
)


def _candidate_containers(soup: BeautifulSoup):
    """Semantic containers first, then any block big enough to plausibly hold prose."""
    for selector in ("article", "main"):
        for node in soup.find_all(selector):
            yield node
    body = soup.body or soup
    for node in body.find_all(["div", "section"], recursive=True):
        yield node
    yield body


def _link_density(node: Any, text_length: int) -> float:
    if not text_length:
        return 1.0
    anchor_chars = sum(
        len(a.get_text(" ", strip=True)) for a in node.find_all("a")
    )
    return min(anchor_chars / text_length, 1.0)


def _score(node: Any, text: str) -> float:
    density = _link_density(node, len(text))
    if density > 0.5:
        return 0.0
    # Paragraph count matters as well as raw length: a long block made of one-line
    # list items is usually a menu, not an article.
    paragraph_bonus = 1.0 + 0.02 * text.count("\n\n")
    return len(text) * (1.0 - density) * paragraph_bonus


def _paragraph_text(node: Any) -> str | None:
    """Collect prose blocks, skipping any individual block that is mostly links.

    Scoring whole containers is not enough. A parent div holding the real article *and*
    a "recent posts" list has acceptable overall link density, so it wins, and the link
    titles ride along into the extracted text. They then dominate keyword frequency:
    on a blog post about AI predictions this pulled in a sidebar of pelican-themed post
    titles and the generated tags came out as `brown-pelican` and `pelican-roost`
    instead of anything about AI.

    Filtering per block catches those lists wherever they sit in the tree, while
    leaving normal prose that happens to contain an inline link or two.
    """
    if node is None:
        return None

    paragraphs: list[str] = []
    for element in node.find_all(["p", "h1", "h2", "h3", "li", "blockquote"]):
        text = element.get_text(" ", strip=True)
        if len(text) <= 30:
            continue
        if _link_density(element, len(text)) > 0.5:
            continue
        paragraphs.append(text)

    if not paragraphs:
        return None
    return "\n\n".join(paragraphs)
