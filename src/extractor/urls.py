"""URL canonicalization, hashing and platform detection.

Canonicalization is the cheapest money-saving code in the whole system. The same
reel shared from three different apps arrives as three different URLs, all with
different tracking junk stapled on. Collapsing them to one canonical form before
anything else runs means a given piece of content is fetched, transcribed and
embedded exactly once, ever.

Get this wrong and you pay a multimodal LLM repeatedly for content you already
have.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from .envelope import Platform, content_hash

# Params that carry no meaning about *which* content this is. Anything matching
# is dropped before hashing.
_TRACKING_PARAMS: set[str] = {
    # generic analytics
    "fbclid", "gclid", "dclid", "msclkid", "twclid", "ttclid", "yclid",
    "mc_cid", "mc_eid", "_ga", "_gl", "icid", "ncid", "cmpid", "campaign_id",
    "spm", "scm", "trk", "trkinfo", "li_fat_id", "guccounter",
    # instagram / meta
    "igsh", "igshid", "ig_rid", "img_index", "epik",
    # tiktok
    "is_from_webapp", "sender_device", "sender_web_id", "web_id", "_r", "_d",
    "checksum", "share_app_id", "share_item_id", "share_link_id", "tt_from",
    "source", "enter_from", "referer_url", "referer_video_id",
    # x / twitter
    "s", "t", "ref_src", "ref_url", "refsrc", "__twitter_impression",
    # youtube (feature/attribution noise; `v` is preserved separately)
    "feature", "ab_channel", "pp", "themerefresh", "embeds_referring_euri",
    "si", "kw", "app",
    # reddit
    "share_id", "utm_name", "rdt", "correlation_id", "post_fullname",
    "context", "chainedposts",
    # linkedin
    "originalsubdomain", "trackingid", "lipi", "licu",
    # misc
    "ref", "ref_source", "referrer", "share_source", "from", "hl", "gi",
}

_TRACKING_PREFIXES: tuple[str, ...] = ("utm_", "pk_", "at_", "ito_", "_hs")

_SHORTENER_HOSTS: set[str] = {
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "buff.ly", "is.gd",
    "cutt.ly", "rb.gy", "s.id", "shorturl.at", "dlvr.it", "trib.al", "lnkd.in",
    "redd.it", "vm.tiktok.com", "vt.tiktok.com", "instagr.am", "ig.me",
    "fb.watch", "amzn.to", "spoti.fi", "wa.me", "t.ly", "shrtco.de",
    "youtu.be",  # resolved structurally below rather than over the network
}

_PLATFORM_HOSTS: dict[str, Platform] = {
    "instagram.com": Platform.INSTAGRAM,
    "instagr.am": Platform.INSTAGRAM,
    "youtube.com": Platform.YOUTUBE,
    "youtu.be": Platform.YOUTUBE,
    "tiktok.com": Platform.TIKTOK,
    "twitter.com": Platform.TWITTER,
    "x.com": Platform.TWITTER,
    "reddit.com": Platform.REDDIT,
    "redd.it": Platform.REDDIT,
    "linkedin.com": Platform.LINKEDIN,
    "lnkd.in": Platform.LINKEDIN,
}

_STRIPPABLE_SUBDOMAINS: tuple[str, ...] = ("www.", "m.", "mobile.", "amp.", "old.")

_IG_PATH = re.compile(r"^/(?:[^/]+/)?(reel|reels|p|tv)/([A-Za-z0-9_-]+)")
_YT_SHORTS = re.compile(r"^/shorts/([A-Za-z0-9_-]{6,})")
_YT_EMBED = re.compile(r"^/(?:embed|v|live)/([A-Za-z0-9_-]{6,})")
_TT_VIDEO = re.compile(r"^/@([^/]+)/video/(\d+)")
_TW_STATUS = re.compile(r"^/([^/]+)/status(?:es)?/(\d+)")
_RD_COMMENTS = re.compile(r"^/r/([^/]+)/comments/([A-Za-z0-9]+)")
_RD_SHORT = re.compile(r"^/(?:r/[^/]+/)?s/([A-Za-z0-9]+)$")


def registrable_host(host: str) -> str:
    """Host with noise subdomains removed. Not a public-suffix-accurate eTLD+1."""
    host = host.lower().strip(".")
    for prefix in _STRIPPABLE_SUBDOMAINS:
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host


def detect_platform(url: str) -> Platform:
    host = registrable_host(urlparse(url).netloc)
    if host in _PLATFORM_HOSTS:
        return _PLATFORM_HOSTS[host]
    # match parent domains too (e.g. gaming.youtube.com -> youtube.com)
    parts = host.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in _PLATFORM_HOSTS:
            return _PLATFORM_HOSTS[candidate]
    return Platform.WEB


def is_shortener(url: str) -> bool:
    return registrable_host(urlparse(url).netloc) in _SHORTENER_HOSTS or (
        urlparse(url).netloc.lower() in _SHORTENER_HOSTS
    )


def resolve_redirects(url: str, timeout: float = 10.0) -> str:
    """Follow a shortener to its destination. Network call; failures are non-fatal.

    Only invoked for known shortener hosts -- there is no reason to spend a
    request resolving a URL that is already canonical.
    """
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": _BROWSER_UA},
        ) as client:
            # HEAD first; several shorteners answer it and it saves the body.
            response = client.head(url)
            if response.status_code >= 400 or not response.url:
                response = client.get(url)
            return str(response.url)
    except Exception:
        return url


_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _clean_query(query: str, keep: set[str] | None = None) -> str:
    """Drop tracking params, keep the rest, sort for a stable hash."""
    pairs = parse_qsl(query, keep_blank_values=False)
    kept = []
    for key, value in pairs:
        lowered = key.lower()
        if keep is not None:
            if lowered in keep:
                kept.append((key, value))
            continue
        if lowered in _TRACKING_PARAMS:
            continue
        if lowered.startswith(_TRACKING_PREFIXES):
            continue
        kept.append((key, value))
    kept.sort()
    return urlencode(kept)


_HOSTNAME = re.compile(
    r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)"
    r"(?:\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*\.[A-Za-z]{2,63}$"
)


def looks_like_url(text: str) -> bool:
    """Whether this text is plausibly a web address rather than a phrase.

    Needed because bare text used to be accepted silently. `canonicalize` prepends a
    scheme when one is missing, so a search term typed into the wrong box became a real
    saved item: the words "salt", "shoe" and "ramen" were stored as `https://salt/`,
    `https://shoe/` and `https://ramen/`, each one costing an extraction attempt and a
    tagging call before failing.

    The test is a registrable hostname -- at least one dot and a plausible TLD -- which
    is what distinguishes `example.com/x` from `ramen`.
    """
    candidate = (text or "").strip()
    if not candidate or any(ch.isspace() for ch in candidate):
        return False

    if "://" in candidate:
        parsed = urlparse(candidate)
        if parsed.scheme.lower() not in {"http", "https"}:
            return False
        host = parsed.hostname or ""
    else:
        host = urlparse("https://" + candidate.lstrip("/")).hostname or ""

    if not host:
        return False
    if host == "localhost":
        return True
    # Bare IPv4 is a valid address but never a link a person means to save, and it is the
    # shape SSRF probes take.
    if all(part.isdigit() for part in host.split(".")) and host.count(".") == 3:
        return False
    return bool(_HOSTNAME.match(host))


def _unwrap_redirect(url: str, *, max_depth: int = 3) -> str:
    """Replace a redirect wrapper with the destination carried in its query string.

    Google results, YouTube's outbound link gate and Facebook's `l.php` all wrap the real
    address in a parameter, and links copied out of those places arrive wrapped. This is
    purely structural, so unlike `resolve_redirects` it costs no network request.

    Worth doing because leaving it wrapped is silently destructive: the wrapper becomes the
    canonical URL, so the fetch returns an interstitial with no content and the link is
    saved as degraded and empty. Verified live -- a Google `/url?...&url=<youtube link>`
    saved this way produced an envelope with no title at all, while the same video added
    directly extracted fine.
    """
    for _ in range(max_depth):
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        host = registrable_host(netloc)
        path = parsed.path or "/"

        # `registrable_host` only strips `www.`, so subdomained wrappers such as
        # `l.facebook.com` have to be matched on the suffix rather than compared directly.
        def on(*suffixes: str) -> bool:
            return any(host == s or host.endswith("." + s) for s in suffixes)

        params: tuple[str, ...] = ()
        # google.com plus country domains (google.co.in, google.de, ...).
        if (host == "google.com" or host.startswith("google.")) and path.startswith("/url"):
            params = ("url", "q")
        elif on("youtube.com") and path.startswith("/redirect"):
            params = ("q",)
        elif on("facebook.com", "messenger.com") and path.startswith("/l."):
            params = ("u",)
        elif host == "out.reddit.com":
            params = ("url",)
        if not params:
            return url

        # parse_qsl percent-decodes, so `url=https%3A%2F%2F...` arrives usable.
        values = dict(parse_qsl(parsed.query, keep_blank_values=False))
        target = next((values[name] for name in params if values.get(name)), None)
        # Require a real absolute URL. Google reuses `q` for ordinary search terms, and a
        # phrase like "how to cook rice" must not be mistaken for a destination.
        if not target or "://" not in target or not looks_like_url(target):
            return url
        url = target
    return url


def canonicalize(url: str, *, follow_shorteners: bool = True) -> str:
    """Reduce a URL to one stable form per piece of content.

    Applies, in order: shortener resolution, scheme/host normalization,
    platform-specific path rewriting, tracking-param removal, fragment removal.

    Raises ValueError for anything that is not plausibly a URL, so stray text cannot
    become a saved item.
    """
    url = (url or "").strip()
    if not url:
        raise ValueError("empty URL")
    if not looks_like_url(url):
        raise ValueError(f"not a URL: {url[:80]!r}")
    if "://" not in url:
        url = "https://" + url.lstrip("/")

    # Before anything else: a wrapper's destination may itself be a shortener or a
    # platform URL needing path rewriting, so unwrap first and let the rest apply.
    url = _unwrap_redirect(url)

    if follow_shorteners and is_shortener(url):
        parsed_short = urlparse(url)
        host = registrable_host(parsed_short.netloc)
        if host == "youtu.be":
            # Structural, no network needed.
            video_id = parsed_short.path.strip("/").split("/")[0]
            if video_id:
                url = f"https://www.youtube.com/watch?v={video_id}"
        else:
            url = resolve_redirects(url)

    parsed = urlparse(url)
    host = registrable_host(parsed.netloc)
    path = parsed.path or "/"
    query = parsed.query
    platform = detect_platform(url)

    if platform is Platform.INSTAGRAM:
        match = _IG_PATH.match(path)
        if match:
            kind = "reel" if match.group(1) in {"reel", "reels"} else match.group(1)
            host, path, query = "instagram.com", f"/{kind}/{match.group(2)}/", ""
        else:
            query = ""

    elif platform is Platform.YOUTUBE:
        host = "youtube.com"
        shorts = _YT_SHORTS.match(path)
        embed = _YT_EMBED.match(path)
        video_id = None
        if shorts:
            video_id = shorts.group(1)
        elif embed:
            video_id = embed.group(1)
        elif path.rstrip("/") == "/watch":
            video_id = dict(parse_qsl(query)).get("v")
        if video_id:
            path, query = "/watch", urlencode({"v": video_id})
        else:
            query = _clean_query(query)

    elif platform is Platform.TIKTOK:
        host = "tiktok.com"
        match = _TT_VIDEO.match(path)
        if match:
            path, query = f"/@{match.group(1).lower()}/video/{match.group(2)}", ""
        else:
            query = _clean_query(query)

    elif platform is Platform.TWITTER:
        host = "x.com"
        match = _TW_STATUS.match(path)
        if match:
            path, query = f"/{match.group(1).lower()}/status/{match.group(2)}", ""
        else:
            query = _clean_query(query)

    elif platform is Platform.REDDIT:
        host = "reddit.com"
        match = _RD_COMMENTS.match(path)
        if match:
            path, query = f"/r/{match.group(1)}/comments/{match.group(2)}/", ""
        else:
            query = _clean_query(query)

    else:
        query = _clean_query(query)

    if path != "/" and path.endswith("/") and platform not in {
        Platform.INSTAGRAM,
        Platform.REDDIT,
    }:
        path = path.rstrip("/")
    if not path:
        path = "/"

    return urlunparse(("https", host, path, "", query, ""))


def url_hash(canonical: str) -> str:
    """Primary key for a piece of content. Hash the canonical form, never the raw."""
    return content_hash(canonical)
