from __future__ import annotations

import base64
import os
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

import jwt
from cryptography.hazmat.primitives.asymmetric import ec

from linq_server.auth import AccessPolicy, Caller, SupabaseTokenVerifier


ISSUER = "https://example.supabase.co/auth/v1"
TESTING_URL = "https://testing.supabase.co"
TESTING_ISSUER = f"{TESTING_URL}/auth/v1"
PRODUCTION_URL = "https://production.supabase.co"
PRODUCTION_ISSUER = f"{PRODUCTION_URL}/auth/v1"


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
                "supabase_issuers": 1,
                "supabase_hs256": False,
                "shared_token": True,
                "analyses_per_hour": 60,
            },
        )

    def test_invalid_supabase_url_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            SupabaseTokenVerifier(supabase_url="http://example.supabase.co")

    def test_accepts_two_allowlisted_supabase_projects(self) -> None:
        testing_key = ec.generate_private_key(ec.SECP256R1())
        production_key = ec.generate_private_key(ec.SECP256R1())
        verifier = SupabaseTokenVerifier(
            supabase_urls=[TESTING_URL, PRODUCTION_URL],
            jwks_clients={
                TESTING_ISSUER: StaticJWKClient(testing_key.public_key()),
                PRODUCTION_ISSUER: StaticJWKClient(production_key.public_key()),
            },
        )

        testing_identity = verifier.verify_identity(
            self.token_for(testing_key, TESTING_ISSUER)
        )
        production_identity = verifier.verify_identity(
            self.token_for(production_key, PRODUCTION_ISSUER)
        )

        self.assertEqual(testing_identity.user_id, self.user_id)
        self.assertEqual(testing_identity.issuer, TESTING_ISSUER)
        self.assertEqual(production_identity.user_id, self.user_id)
        self.assertEqual(production_identity.issuer, PRODUCTION_ISSUER)
        self.assertEqual(verifier.issuer_count, 2)

    def test_rejects_issuer_outside_the_configured_allowlist(self) -> None:
        testing_key = ec.generate_private_key(ec.SECP256R1())
        attacker_key = ec.generate_private_key(ec.SECP256R1())
        verifier = SupabaseTokenVerifier(
            supabase_urls=[TESTING_URL],
            jwks_clients={
                TESTING_ISSUER: StaticJWKClient(testing_key.public_key()),
            },
        )

        self.assertIsNone(
            verifier.verify_identity(
                self.token_for(attacker_key, "https://attacker.invalid/auth/v1")
            )
        )

    def test_rejects_token_signed_by_the_other_allowlisted_project(self) -> None:
        testing_key = ec.generate_private_key(ec.SECP256R1())
        production_key = ec.generate_private_key(ec.SECP256R1())
        verifier = SupabaseTokenVerifier(
            supabase_urls=[TESTING_URL, PRODUCTION_URL],
            jwks_clients={
                TESTING_ISSUER: StaticJWKClient(testing_key.public_key()),
                PRODUCTION_ISSUER: StaticJWKClient(production_key.public_key()),
            },
        )

        self.assertIsNone(
            verifier.verify_identity(
                self.token_for(production_key, TESTING_ISSUER)
            )
        )

    def test_rate_limit_identity_includes_supabase_issuer(self) -> None:
        testing = Caller(self.user_id, "supabase", TESTING_ISSUER)
        production = Caller(self.user_id, "supabase", PRODUCTION_ISSUER)
        self.assertNotEqual(testing.rate_key, production.rate_key)

    def test_plural_environment_variable_configures_both_projects(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SUPABASE_URL": TESTING_URL,
                "SUPABASE_URLS": f"{TESTING_URL}, {PRODUCTION_URL}",
                "SUPABASE_JWT_SECRET": "",
            },
            clear=False,
        ):
            policy = AccessPolicy.from_env()

        self.assertEqual(policy.supabase.issuer_count, 2)
        self.assertEqual(
            set(policy.supabase.issuers),
            {TESTING_ISSUER, PRODUCTION_ISSUER},
        )

    def test_legacy_secret_cannot_be_reused_for_multiple_projects(self) -> None:
        with self.assertRaises(ValueError):
            SupabaseTokenVerifier(
                supabase_urls=[TESTING_URL, PRODUCTION_URL],
                jwt_secret="one-secret-cannot-identify-two-projects",
            )

    def token_for(
        self,
        private_key: ec.EllipticCurvePrivateKey,
        issuer: str,
    ) -> str:
        now = int(time.time())
        return jwt.encode(
            {
                "sub": self.user_id,
                "role": "authenticated",
                "aud": "authenticated",
                "iss": issuer,
                "iat": now,
                "exp": now + 300,
            },
            private_key,
            algorithm="ES256",
            headers={"kid": "test-key"},
        )


if __name__ == "__main__":
    unittest.main()
