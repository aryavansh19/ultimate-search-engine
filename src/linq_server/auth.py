"""Authentication for the LinQ service.

Two accepted credentials, checked in this order:

1. **A Supabase access token.** LinQ already signs users in with Supabase, and the session
   is stored in an App Group so both the app and the share extension have it. That token is
   per-user, short-lived, and revocable by signing the user out — which is everything a
   shared secret is not. Verified locally with HMAC-SHA256 against the project's JWT secret,
   so there is no round trip to Supabase on every request.
2. **A shared token** (`LINQ_API_TOKEN`). For development, for curl, and as the fallback if
   the JWT secret is not configured.

Why not ship the Gemini key and skip all this: an API key in an IPA is public. `strings` on
the binary is enough. The failure mode is someone else spending the quota, and with Gemini
that is a billing problem rather than an inconvenience.

Why verify the JWT locally rather than calling `auth.getUser()`: that would add a network
round trip to Google *plus* one to Supabase on every enrichment, and it would make this
service unavailable whenever Supabase is. The signature is self-validating; only revocation
needs the round trip, and a 1-hour token lifetime bounds that risk.

HS256 is implemented on stdlib `hmac` on purpose — PyJWT is not currently installed, and
pulling in a dependency for thirty lines of well-specified hashing is a poor trade. If you
later switch the project to asymmetric (RS256/ES256) keys, replace this with PyJWT rather
than hand-rolling the curve maths.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import HTTPException, Request

log = logging.getLogger("linq.auth")

PUBLIC_PATHS = frozenset({"/health", "/", "/favicon.ico", "/docs", "/openapi.json", "/redoc"})


class AuthError(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=401, detail=detail)


@dataclass(frozen=True)
class Caller:
    """Who is making the request. `user_id` is None for the shared-token path."""

    user_id: str | None
    kind: str  # "supabase" | "shared"

    @property
    def rate_key(self) -> str:
        # Per-user when we know the user, so one heavy user cannot exhaust everyone's
        # budget, and the limit follows the account rather than a shared NAT address.
        return f"user:{self.user_id}" if self.user_id else "shared"


def _b64url(segment: str) -> bytes:
    """Decode base64url without padding, as JWT emits it."""
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def verify_supabase_jwt(token: str, secret: str, *, leeway: int = 30) -> str | None:
    """Return the user id (`sub`) if the token is a valid Supabase JWT, else None.

    Returns None rather than raising so the caller can fall through to the shared-token
    path. Anything malformed, mis-signed or expired is simply "not a valid JWT".
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    header_b64, payload_b64, signature_b64 = parts

    try:
        header = json.loads(_b64url(header_b64))
    except Exception:
        return None
    if header.get("alg") != "HS256":
        # Never honour `alg: none`, and do not silently accept an algorithm we are not
        # actually checking — that is the classic JWT confusion vulnerability.
        log.warning("rejecting token with unexpected alg=%r", header.get("alg"))
        return None

    expected = hmac.new(
        secret.encode("utf-8"), f"{header_b64}.{payload_b64}".encode("utf-8"), hashlib.sha256
    ).digest()
    try:
        supplied = _b64url(signature_b64)
    except Exception:
        return None
    if not hmac.compare_digest(expected, supplied):
        return None

    try:
        payload = json.loads(_b64url(payload_b64))
    except Exception:
        return None

    now = time.time()
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and now > exp + leeway:
        return None
    nbf = payload.get("nbf")
    if isinstance(nbf, (int, float)) and now + leeway < nbf:
        return None

    subject = payload.get("sub")
    return str(subject) if subject else None


class AccessPolicy:
    """Credential check plus a per-caller hourly cap on expensive calls."""

    def __init__(
        self,
        *,
        shared_token: str | None = None,
        supabase_jwt_secret: str | None = None,
        analyses_per_hour: int = 60,
        require_auth: bool = True,
    ) -> None:
        self.shared_token = shared_token or None
        self.supabase_jwt_secret = supabase_jwt_secret or None
        self.analyses_per_hour = analyses_per_hour
        self.require_auth = require_auth
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> AccessPolicy:
        shared = (os.getenv("LINQ_API_TOKEN") or os.getenv("APP_ACCESS_TOKEN") or "").strip()
        secret = (os.getenv("SUPABASE_JWT_SECRET") or "").strip()
        require = (os.getenv("LINQ_REQUIRE_AUTH", "1").strip().lower()
                   in {"1", "true", "yes", "on"})
        return cls(
            shared_token=shared or None,
            supabase_jwt_secret=secret or None,
            analyses_per_hour=_int("LINQ_ANALYSES_PER_HOUR", 60),
            require_auth=require,
        )

    @property
    def configured(self) -> bool:
        return bool(self.shared_token or self.supabase_jwt_secret)

    def describe(self) -> dict[str, object]:
        return {
            "require_auth": self.require_auth,
            "supabase_jwt": bool(self.supabase_jwt_secret),
            "shared_token": bool(self.shared_token),
            "analyses_per_hour": self.analyses_per_hour,
        }

    # ---------------------------------------------------------------------- checking
    def identify(self, request: Request) -> Caller:
        """Authenticate the request, or raise 401."""
        if not self.require_auth:
            return Caller(user_id=None, kind="shared")

        if not self.configured:
            # Refuse to run wide open. An unauthenticated instance with a Gemini key
            # attached is a bill waiting to happen, so fail loudly at request time
            # rather than quietly serving everyone.
            raise HTTPException(
                status_code=503,
                detail=(
                    "Server has no credentials configured. Set SUPABASE_JWT_SECRET "
                    "(preferred) or LINQ_API_TOKEN, or set LINQ_REQUIRE_AUTH=0 for "
                    "loopback-only development."
                ),
            )

        token = self._bearer(request) or request.headers.get("x-access-token")
        if not token:
            raise AuthError("Missing credentials. Send 'Authorization: Bearer <token>'.")

        if self.supabase_jwt_secret:
            user_id = verify_supabase_jwt(token, self.supabase_jwt_secret)
            if user_id:
                return Caller(user_id=user_id, kind="supabase")

        if self.shared_token and hmac.compare_digest(token, self.shared_token):
            return Caller(user_id=None, kind="shared")

        raise AuthError("Invalid or expired credentials.")

    @staticmethod
    def _bearer(request: Request) -> str | None:
        raw = request.headers.get("authorization") or ""
        if raw.lower().startswith("bearer "):
            return raw[7:].strip() or None
        return None

    def charge(self, caller: Caller) -> None:
        """Count one expensive call against the caller's hourly budget."""
        if self.analyses_per_hour <= 0:
            return
        now = time.monotonic()
        window = 3600.0
        with self._lock:
            hits = self._hits[caller.rate_key]
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= self.analyses_per_hour:
                wait = int((window - (now - hits[0])) / 60) + 1
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Rate limit: {self.analyses_per_hour} analyses per hour. "
                        f"Try again in about {wait} minute(s)."
                    ),
                )
            hits.append(now)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
