"""Access control and abuse limits for the API.

Needed the moment this stops being loopback-only. Without it, sharing a tunnel URL hands
anyone who has the link three things: your entire saved library, the ability to spend your
Gemini and OpenRouter quota by adding links, and a server-side URL fetcher running on your
home machine.

Deliberately minimal. This is a shared-secret gate for letting a friend try the app, not an
identity system -- there are no accounts, no sessions and no per-user data. If this ever
needs real users, replace it with Supabase auth rather than growing it.

Two protections:

* A shared token, accepted as a header or a `?k=` query parameter. The query form exists so a
  single link can be pasted to someone; the frontend moves it into local storage and sends it
  as a header from then on, so it stops appearing in the address bar.
* A per-IP hourly cap on writes. A token stops strangers; it does not stop an enthusiastic
  friend from adding two hundred videos and exhausting a daily quota that is shared with
  nothing else.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

log = logging.getLogger("api.security")

TOKEN_HEADER = "x-access-token"
TOKEN_QUERY = "k"

# Paths that stay open regardless: the page itself is inert HTML, and a health probe with no
# data in it is useful for checking a tunnel is alive.
PUBLIC_PATHS = frozenset({"/", "/health", "/favicon.ico"})
PUBLIC_PREFIXES = ("/static/",)


class AccessPolicy:
    """Shared-token gate plus a write rate limit."""

    def __init__(
        self,
        token: str | None = None,
        *,
        writes_per_hour: int = 40,
        allow_writes: bool = True,
    ) -> None:
        self.token = token or None
        self.writes_per_hour = writes_per_hour
        self.allow_writes = allow_writes
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> AccessPolicy:
        raw = (os.getenv("APP_ACCESS_TOKEN") or "").strip()
        return cls(
            token=raw or None,
            writes_per_hour=_int("APP_WRITES_PER_HOUR", 40),
            allow_writes=_bool("APP_ALLOW_WRITES", True),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    # ------------------------------------------------------------------ checking
    def check(self, request: Request) -> None:
        """Raise HTTPException if this request is not allowed through."""
        path = request.url.path
        if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
            return
        if not self.enabled:
            # No token configured: local-only operation, nothing to enforce.
            return

        supplied = request.headers.get(TOKEN_HEADER) or request.query_params.get(TOKEN_QUERY)
        if not supplied or not hmac.compare_digest(str(supplied), str(self.token)):
            # compare_digest rather than == so the comparison does not leak the token's
            # length or matching prefix through response timing.
            raise HTTPException(status_code=401, detail="Access token required or invalid.")

        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            self._check_writes(request)

    def _check_writes(self, request: Request) -> None:
        if not self.allow_writes:
            raise HTTPException(
                status_code=403,
                detail="This instance is shared read-only. Searching works; adding does not.",
            )
        if self.writes_per_hour <= 0:
            return

        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = 3600.0
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= self.writes_per_hour:
                oldest = hits[0]
                wait = int((window - (now - oldest)) / 60) + 1
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Rate limit: {self.writes_per_hour} writes per hour. "
                        f"Try again in about {wait} minute(s)."
                    ),
                )
            hits.append(now)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def generate_token(length: int = 24) -> str:
    """A URL-safe token suitable for pasting into a link."""
    import secrets

    return secrets.token_urlsafe(length)
