from __future__ import annotations

import base64
import time
import unittest
from uuid import uuid4

import jwt
from cryptography.hazmat.primitives.asymmetric import ec

from linq_server.auth import AccessPolicy, SupabaseTokenVerifier


ISSUER = "https://example.supabase.co/auth/v1"


def _b64url_uint(value: int) -> str:
    raw = value.to_bytes(32, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


class StaticJWKClient:
    def __init__(self, public_key: ec.EllipticCurvePublicKey) -> None:
        numbers = public_key.public_numbers()
        self.signing_key = jwt.PyJWK.from_dict(
            {
                "kty": "EC",
                "crv": "P-256",
                "x": _b64url_uint(numbers.x),
                "y": _b64url_uint(numbers.y),
                "use": "sig",
                "key_ops": ["verify"],
                "alg": "ES256",
                "kid": "test-key",
            }
        )

    def get_signing_key_from_jwt(self, token: str) -> jwt.PyJWK:
        return self.signing_key


class SupabaseTokenVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.user_id = str(uuid4())
        self.verifier = SupabaseTokenVerifier(
            supabase_url="https://example.supabase.co",
            jwks_client=StaticJWKClient(self.private_key.public_key()),
        )

    def token(self, **overrides: object) -> str:
        now = int(time.time())
        claims: dict[str, object] = {
            "sub": self.user_id,
            "role": "authenticated",
            "aud": "authenticated",
            "iss": ISSUER,
            "iat": now,
            "exp": now + 300,
        }
        claims.update(overrides)
        return jwt.encode(
            claims,
            self.private_key,
            algorithm="ES256",
            headers={"kid": "test-key"},
        )

    def test_accepts_valid_es256_supabase_session(self) -> None:
        self.assertEqual(self.verifier.verify(self.token()), self.user_id)

    def test_rejects_wrong_issuer(self) -> None:
        self.assertIsNone(self.verifier.verify(self.token(iss="https://attacker.invalid/auth/v1")))

    def test_rejects_wrong_audience(self) -> None:
        self.assertIsNone(self.verifier.verify(self.token(aud="anon")))

    def test_rejects_non_authenticated_role(self) -> None:
        self.assertIsNone(self.verifier.verify(self.token(role="service_role")))

    def test_rejects_expired_token(self) -> None:
        self.assertIsNone(self.verifier.verify(self.token(exp=int(time.time()) - 120)))

    def test_rejects_non_uuid_subject(self) -> None:
        self.assertIsNone(self.verifier.verify(self.token(sub="not-a-user-id")))

    def test_rejects_unsigned_or_malformed_token(self) -> None:
        self.assertIsNone(self.verifier.verify("not.a.jwt"))

    def test_rejects_hs256_when_no_legacy_secret_is_configured(self) -> None:
        token = jwt.encode(
            {
                "sub": self.user_id,
                "role": "authenticated",
                "aud": "authenticated",
                "iss": ISSUER,
                "exp": int(time.time()) + 300,
            },
            "attacker-controlled-secret",
            algorithm="HS256",
        )
        self.assertIsNone(self.verifier.verify(token))

    def test_health_description_reports_jwks_without_secrets(self) -> None:
        policy = AccessPolicy(
            supabase_url="https://example.supabase.co",
            shared_token="fallback",
        )
        description = policy.describe()
        self.assertEqual(
            description,
            {
                "require_auth": True,
                "supabase_jwt": True,
                "supabase_jwks": True,
                "supabase_hs256": False,
                "shared_token": True,
                "analyses_per_hour": 60,
            },
        )

    def test_invalid_supabase_url_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            SupabaseTokenVerifier(supabase_url="http://example.supabase.co")


if __name__ == "__main__":
    unittest.main()
