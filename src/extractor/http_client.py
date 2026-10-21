"""Shared, connection-pooled HTTP client for the extractor cascade.

Eliminates repetitive TCP connection setup and TLS handshakes across shortener unrolling,
oEmbed fetching, and HTML scraping.

Features:
1. Persistent connection pooling with configurable keep-alive limits.
2. Standard browser-grade headers (Accept, Accept-Language, Sec-Fetch-*) to bypass basic bot-walls.
3. Integration with scraper_governor to prevent burst 429s on third-party endpoints.
4. Clean lifecycle management (thread-safe, graceful close).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import httpx

from quota import scraper_governor

log = logging.getLogger("extractor.http")

DEFAULT_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

WIKIMEDIA_UA = "LinQBot/1.0 (https://linq.app; support@linq.app)"

DEFAULT_HEADERS = {
    "User-Agent": DEFAULT_BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


def resolve_headers(url: str, custom_headers: dict[str, str] | None = None) -> dict[str, str]:
    """Ensure platform compliance policies (e.g. Wikimedia bot policy) are enforced."""
    headers = dict(custom_headers) if custom_headers else {}
    url_lower = url.lower()
    if "wikipedia.org" in url_lower or "wikimedia.org" in url_lower:
        headers["User-Agent"] = WIKIMEDIA_UA
    elif "User-Agent" not in headers:
        headers["User-Agent"] = DEFAULT_BROWSER_UA
    return headers


class PooledHttpClient:
    """Thread-safe persistent HTTP client wrapper with connection pooling."""

    def __init__(
        self,
        timeout: float = 12.0,
        max_connections: int = 50,
        max_keepalive: int = 20,
        keepalive_expiry: float = 30.0,
    ) -> None:
        self.timeout = httpx.Timeout(timeout, connect=5.0)
        self.limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
            keepalive_expiry=keepalive_expiry,
        )
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()

    def _get_or_create_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            with self._lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.Client(
                        timeout=self.timeout,
                        limits=self.limits,
                        follow_redirects=True,
                        headers=DEFAULT_HEADERS,
                    )
        return self._client

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
        pace: bool = True,
    ) -> httpx.Response:
        if pace:
            scraper_governor.acquire()
        client = self._get_or_create_client()
        req_timeout = httpx.Timeout(timeout, connect=5.0) if timeout else self.timeout
        effective_headers = resolve_headers(url, headers)
        try:
            resp = client.get(url, headers=effective_headers, params=params, timeout=req_timeout)
            if resp.status_code == 429:
                scraper_governor.record_error(status_code=429, error_body=resp.text[:100])
            else:
                scraper_governor.record_success()
            return resp
        except Exception:
            scraper_governor.record_error(status_code=500)
            raise

    def head(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        pace: bool = True,
    ) -> httpx.Response:
        if pace:
            scraper_governor.acquire()
        client = self._get_or_create_client()
        req_timeout = httpx.Timeout(timeout, connect=5.0) if timeout else self.timeout
        effective_headers = resolve_headers(url, headers)
        try:
            resp = client.head(url, headers=effective_headers, timeout=req_timeout)
            if resp.status_code == 429:
                scraper_governor.record_error(status_code=429, error_body=resp.text[:100])
            else:
                scraper_governor.record_success()
            return resp
        except Exception:
            scraper_governor.record_error(status_code=500)
            raise

    def close(self) -> None:
        with self._lock:
            if self._client is not None and not self._client.is_closed:
                self._client.close()
                self._client = None


_shared_client: PooledHttpClient | None = None
_client_lock = threading.Lock()


def get_shared_client() -> PooledHttpClient:
    """Access the application-wide connection pooled HTTP client."""
    global _shared_client
    if _shared_client is None:
        with _client_lock:
            if _shared_client is None:
                _shared_client = PooledHttpClient()
    return _shared_client
