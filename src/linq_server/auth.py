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
from collections.abc import Iterable, Mapping
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
    issuer: str | None = None

    @property
    def rate_key(self) -> str:
        # Per-user when we know the user, so one heavy user cannot exhaust everyone's
        # budget, and the limit follows the account rather than a shared NAT address.
        if not self.user_id:
            return "shared"
        # Supabase user UUIDs are unique inside one project, not across projects. Keep
        # staging and production budgets isolated even if they happen to contain the
        # same UUID.
        return f"user:{self.issuer or 'legacy'}:{self.user_id}"


@dataclass(frozen=True)
class VerifiedSupabaseIdentity:
    user_id: str
    issuer: str | None


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

    ``SUPABASE_URL`` or ``SUPABASE_URLS`` enables asymmetric ES256/RS256 verification
    through each project's public JWKS endpoint. The unverified JWT issuer is used only
    to select an exact match from this configured allowlist; all claims and the signature
    are then verified against that project's key. ``SUPABASE_JWT_SECRET`` is only for a
    single legacy HS256 project.
    Algorithm families are deliberately handled in separate branches to prevent public
    keys from ever being interpreted as HMAC secrets.
    """

    def __init__(
        self,
        *,
        supabase_url: str | None = None,
        supabase_urls: Iterable[str] | None = None,
        jwt_secret: str | None = None,
        audience: str = "authenticated",
        leeway: int = 30,
        jwks_client: Any | None = None,
        jwks_clients: Mapping[str, Any] | None = None,
    ) -> None:
        self.jwt_secret = (jwt_secret or "").strip() or None
        self.audience = audience
        self.leeway = leeway

        supplied_urls: list[str] = []
        if supabase_url:
            supplied_urls.append(supabase_url)
        if isinstance(supabase_urls, str):
            supplied_urls.extend(supabase_urls.split(","))
        elif supabase_urls:
            supplied_urls.extend(supabase_urls)

        normalized_urls: list[str] = []
        for candidate in supplied_urls:
            normalized = candidate.strip().rstrip("/")
            if not normalized or normalized in normalized_urls:
                continue
            parsed = urlparse(normalized)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("Every Supabase URL must be an absolute HTTPS URL")
            normalized_urls.append(normalized)

        if self.jwt_secret and len(normalized_urls) > 1:
            raise ValueError(
                "SUPABASE_JWT_SECRET cannot be shared across multiple Supabase projects; "
                "use their public JWKS keys instead"
            )
        if jwks_client and len(normalized_urls) > 1:
            raise ValueError("jwks_client is only valid for one Supabase URL")

        self.supabase_urls = tuple(normalized_urls)
        self.supabase_url = self.supabase_urls[0] if len(self.supabase_urls) == 1 else None
        self.issuers = tuple(f"{url}/auth/v1" for url in self.supabase_urls)
        self.issuer = self.issuers[0] if len(self.issuers) == 1 else None
        self.jwks_url = (
            f"{self.issuer}/.well-known/jwks.json" if self.issuer else None
        )
        supplied_clients = dict(jwks_clients or {})
        self._jwks_clients: dict[str, Any] = {}
        for url, issuer in zip(self.supabase_urls, self.issuers, strict=True):
            client = supplied_clients.get(issuer) or supplied_clients.get(url)
            if client is None and jwks_client is not None:
                client = jwks_client
            self._jwks_clients[issuer] = client or PyJWKClient(
                f"{issuer}/.well-known/jwks.json"
            )

        # Retain the singular attribute for callers that only need a yes/no answer.
        self.jwks_client = (
            next(iter(self._jwks_clients.values())) if len(self._jwks_clients) == 1 else None
        )

    @property
    def configured(self) -> bool:
        return bool(self._jwks_clients or self.jwt_secret)

    @property
    def uses_jwks(self) -> bool:
        return bool(self._jwks_clients)

    @property
    def issuer_count(self) -> int:
        return len(self._jwks_clients)

    def verify(self, token: str) -> str | None:
        identity = self.verify_identity(token)
        return identity.user_id if identity else None

    def verify_identity(self, token: str) -> VerifiedSupabaseIdentity | None:
        try:
            header = jwt.get_unverified_header(token)
            algorithm = header.get("alg")

            verified_issuer: str | None = None
            if algorithm in ASYMMETRIC_ALGORITHMS and self._jwks_clients:
                unverified_claims = jwt.decode(
                    token,
                    options={"verify_signature": False},
                )
                candidate_issuer = unverified_claims.get("iss")
                if not isinstance(candidate_issuer, str):
                    return None
                selected_client = self._jwks_clients.get(candidate_issuer)
                if selected_client is None:
                    # The token controls this string, so it can only select an exact
                    # preconfigured issuer. Never discover keys from an arbitrary URL.
                    return None
                signing_key = selected_client.get_signing_key_from_jwt(token)
                claims = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=[algorithm],
                    audience=self.audience,
                    issuer=candidate_issuer,
                    leeway=self.leeway,
                    options={"require": ["aud", "exp", "iss", "role", "sub"]},
                )
                verified_issuer = candidate_issuer
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
                    verified_issuer = self.issuer
                else:
                    # Preserve compatibility for legacy deployments which only set the
                    # JWT secret and predate SUPABASE_URL configuration.
                    subject = verify_supabase_jwt(
                        token, self.jwt_secret, leeway=self.leeway
                    )
                    return (
                        VerifiedSupabaseIdentity(subject, None) if subject else None
                    )
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
        return VerifiedSupabaseIdentity(subject, verified_issuer)


class AccessPolicy:
    """Credential check plus a per-caller hourly cap on expensive calls."""

    def __init__(
        self,
        *,
        shared_token: str | None = None,
        supabase_url: str | None = None,
        supabase_urls: Iterable[str] | None = None,
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
            supabase_url=self.supabase_url,
            supabase_urls=supabase_urls,
            jwt_secret=self.supabase_jwt_secret,
        )
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> AccessPolicy:
        shared = (os.getenv("LINQ_API_TOKEN") or os.getenv("APP_ACCESS_TOKEN") or "").strip()
        secret = (os.getenv("SUPABASE_JWT_SECRET") or "").strip()
        supabase_url = (os.getenv("SUPABASE_URL") or "").strip()
        supabase_urls = [
            url.strip()
            for url in (os.getenv("SUPABASE_URLS") or "").split(",")
            if url.strip()
        ]
        require = (os.getenv("LINQ_REQUIRE_AUTH", "1").strip().lower()
                   in {"1", "true", "yes", "on"})
        return cls(
            shared_token=shared or None,
            supabase_url=supabase_url or None,
            supabase_urls=supabase_urls,
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
            "supabase_issuers": self.supabase.issuer_count,
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
                    "Server has no credentials configured. Set SUPABASE_URLS or SUPABASE_URL "
                    "(preferred), SUPABASE_JWT_SECRET, or LINQ_API_TOKEN; or set LINQ_REQUIRE_AUTH=0 for "
                    "loopback-only development."
                ),
            )

        token = self._bearer(request) or request.headers.get("x-access-token")
        if not token:
            raise AuthError("Missing credentials. Send 'Authorization: Bearer <token>'.")

        if self.supabase.configured:
            identity = self.supabase.verify_identity(token)
            if identity:
                return Caller(
                    user_id=identity.user_id,
                    kind="supabase",
                    issuer=identity.issuer,
                )

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
