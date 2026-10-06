"""Authentication for the LinQ service.

Two accepted credentials, checked in this order:

1. **A Supabase access token.** LinQ already signs users in with Supabase, and the session
   is stored in an App Group so both the app and the share extension have it. That token is
   per-user, short-lived, and revocable by signing the user out — which is everything a
   shared secret is not. Asymmetric tokens are verified locally against Supabase's public
   JWKS endpoint, including issuer, audience, role, expiry, and subject validation. Legacy
   HS256 tokens remain supported when ``SUPABASE_JWT_SECRET`` is configured.
2. **A shared token** (`LINQ_API_TOKEN`). For development, for curl, and as the fallback if
   Supabase authentication is not configured.

Why not ship the Gemini key and skip all this: an API key in an IPA is public. `strings` on
the binary is enough. The failure mode is someone else spending the quota, and with Gemini
that is a billing problem rather than an inconvenience.

The JWKS client caches signing keys, so verification does not put Supabase Auth in the hot
path for every request. Key IDs are resolved through the discovery endpoint, allowing
Supabase key rotation without shipping new app or server secrets.
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
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from fastapi import HTTPException, Request
import jwt
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientError

log = logging.getLogger("linq.auth")

PUBLIC_PATHS = frozenset({"/health", "/", "/favicon.ico", "/docs", "/openapi.json", "/redoc"})
ASYMMETRIC_ALGORITHMS = frozenset({"ES256", "RS256"})


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


class SupabaseTokenVerifier:
    """Verify Supabase session JWTs without sending them back to Auth.

    ``SUPABASE_URL`` enables asymmetric ES256/RS256 verification through the project's
    public JWKS endpoint. ``SUPABASE_JWT_SECRET`` is only for legacy HS256 projects.
    Algorithm families are deliberately handled in separate branches to prevent public
    keys from ever being interpreted as HMAC secrets.
    """

    def __init__(
        self,
        *,
        supabase_url: str | None = None,
        jwt_secret: str | None = None,
        audience: str = "authenticated",
        leeway: int = 30,
        jwks_client: Any | None = None,
    ) -> None:
        self.supabase_url = (supabase_url or "").strip().rstrip("/") or None
        self.jwt_secret = (jwt_secret or "").strip() or None
        self.audience = audience
        self.leeway = leeway
        self.issuer: str | None = None
        self.jwks_url: str | None = None
        self.jwks_client: Any | None = None

        if self.supabase_url:
            parsed = urlparse(self.supabase_url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("SUPABASE_URL must be an absolute HTTPS URL")
            self.issuer = f"{self.supabase_url}/auth/v1"
            self.jwks_url = f"{self.issuer}/.well-known/jwks.json"
            self.jwks_client = jwks_client or PyJWKClient(self.jwks_url)

    @property
    def configured(self) -> bool:
        return bool(self.jwks_client or self.jwt_secret)

    @property
    def uses_jwks(self) -> bool:
        return self.jwks_client is not None

    def verify(self, token: str) -> str | None:
        try:
            header = jwt.get_unverified_header(token)
            algorithm = header.get("alg")

            if algorithm in ASYMMETRIC_ALGORITHMS and self.jwks_client and self.issuer:
                signing_key = self.jwks_client.get_signing_key_from_jwt(token)
                claims = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=[algorithm],
                    audience=self.audience,
                    issuer=self.issuer,
                    leeway=self.leeway,
                    options={"require": ["aud", "exp", "iss", "role", "sub"]},
                )
            elif algorithm == "HS256" and self.jwt_secret:
                if self.issuer:
                    claims = jwt.decode(
                        token,
                        self.jwt_secret,
                        algorithms=["HS256"],
                        audience=self.audience,
                        issuer=self.issuer,
                        leeway=self.leeway,
                        options={"require": ["aud", "exp", "iss", "role", "sub"]},
                    )
                else:
                    # Preserve compatibility for legacy deployments which only set the
                    # JWT secret and predate SUPABASE_URL configuration.
                    return verify_supabase_jwt(token, self.jwt_secret, leeway=self.leeway)
            else:
                return None
        except (jwt.PyJWTError, PyJWKClientError, ValueError, TypeError) as exc:
            log.debug("Supabase JWT rejected: %s", exc)
            return None

        if claims.get("role") != "authenticated":
            return None
        subject = claims.get("sub")
        if not isinstance(subject, str):
            return None
        try:
            UUID(subject)
        except ValueError:
            return None
        return subject


class AccessPolicy:
    """Credential check plus a per-caller hourly cap on expensive calls."""

    def __init__(
        self,
        *,
        shared_token: str | None = None,
        supabase_url: str | None = None,
        supabase_jwt_secret: str | None = None,
        analyses_per_hour: int = 60,
        require_auth: bool = True,
    ) -> None:
        self.shared_token = shared_token or None
        self.supabase_jwt_secret = supabase_jwt_secret or None
        self.supabase_url = supabase_url or None
        self.analyses_per_hour = analyses_per_hour
        self.require_auth = require_auth
        self.supabase = SupabaseTokenVerifier(
            supabase_url=self.supabase_url, jwt_secret=self.supabase_jwt_secret
        )
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> AccessPolicy:
        shared = (os.getenv("LINQ_API_TOKEN") or os.getenv("APP_ACCESS_TOKEN") or "").strip()
        secret = (os.getenv("SUPABASE_JWT_SECRET") or "").strip()
        supabase_url = (os.getenv("SUPABASE_URL") or "").strip()
        require = (os.getenv("LINQ_REQUIRE_AUTH", "1").strip().lower()
                   in {"1", "true", "yes", "on"})
        return cls(
            shared_token=shared or None,
            supabase_url=supabase_url or None,
            supabase_jwt_secret=secret or None,
            analyses_per_hour=_int("LINQ_ANALYSES_PER_HOUR", 60),
            require_auth=require,
        )

    @property
    def configured(self) -> bool:
        return bool(self.shared_token or self.supabase.configured)

    def describe(self) -> dict[str, object]:
        return {
            "require_auth": self.require_auth,
            "supabase_jwt": self.supabase.configured,
            "supabase_jwks": self.supabase.uses_jwks,
            "supabase_hs256": bool(self.supabase_jwt_secret),
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
                    "Server has no credentials configured. Set SUPABASE_URL "
                    "(preferred), SUPABASE_JWT_SECRET, or LINQ_API_TOKEN; or set LINQ_REQUIRE_AUTH=0 for "
                    "loopback-only development."
                ),
            )

        token = self._bearer(request) or request.headers.get("x-access-token")
        if not token:
            raise AuthError("Missing credentials. Send 'Authorization: Bearer <token>'.")

        if self.supabase.configured:
            user_id = self.supabase.verify(token)
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
