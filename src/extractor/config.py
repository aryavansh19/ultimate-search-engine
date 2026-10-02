"""Runtime configuration, read from the environment with working defaults.

Everything here has a default that lets the extractor run with no .env file, so a
fresh clone works immediately and configuration is purely additive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .envelope import ExtractionTier

# See the note in enrichment/config.py: project-local .env values take precedence over
# whatever is already exported in the shell.
load_dotenv(override=True)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _csv(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class ManagedApiConfig:
    """Adapter settings for a third-party scraper API (tier 3).

    Left unconfigured, `enabled` is False and the cascade *skips* the tier rather
    than recording a failure -- a missing optional provider is not an error.
    """

    url: str | None = None
    api_key: str | None = None
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer"
    url_param: str = "url"
    method: str = "POST"
    platforms: list[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def handles(self, platform: str) -> bool:
        if not self.platforms:
            return True
        return platform.lower() in self.platforms


@dataclass(slots=True)
class ExtractorConfig:
    cache_path: Path = Path("data/extractor_cache.sqlite3")
    http_timeout: float = 15.0
    tier_timeout: float = 45.0
    user_agent: str = _BROWSER_UA
    cookies_from_browser: str | None = None
    cookie_file: str | None = None
    disabled_tiers: set[ExtractionTier] = field(default_factory=set)
    managed_api: ManagedApiConfig = field(default_factory=ManagedApiConfig)
    max_article_chars: int = 20_000

    @classmethod
    def from_env(cls) -> ExtractorConfig:
        disabled: set[ExtractionTier] = set()
        for name in _csv("EXTRACTOR_DISABLED_TIERS"):
            try:
                disabled.add(ExtractionTier(name))
            except ValueError:
                continue

        return cls(
            cache_path=Path(
                os.getenv("EXTRACTOR_CACHE_PATH", "data/extractor_cache.sqlite3")
            ),
            http_timeout=_float("EXTRACTOR_HTTP_TIMEOUT", 15.0),
            tier_timeout=_float("EXTRACTOR_TIER_TIMEOUT", 45.0),
            user_agent=os.getenv("EXTRACTOR_USER_AGENT", _BROWSER_UA),
            cookies_from_browser=os.getenv("EXTRACTOR_COOKIES_FROM_BROWSER") or None,
            cookie_file=os.getenv("EXTRACTOR_COOKIE_FILE") or None,
            disabled_tiers=disabled,
            managed_api=ManagedApiConfig(
                url=os.getenv("MANAGED_API_URL") or None,
                api_key=os.getenv("MANAGED_API_KEY") or None,
                auth_header=os.getenv("MANAGED_API_AUTH_HEADER", "Authorization"),
                auth_prefix=os.getenv("MANAGED_API_AUTH_PREFIX", "Bearer"),
                url_param=os.getenv("MANAGED_API_URL_PARAM", "url"),
                method=os.getenv("MANAGED_API_METHOD", "POST").upper(),
                platforms=_csv("MANAGED_API_PLATFORMS"),
            ),
        )
