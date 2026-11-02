from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import UUID

import httpx
from fastapi.testclient import TestClient

from enrichment.config import EnrichmentConfig
from enrichment.providers.base import ProviderError
from linq_server.auth import Caller
from linq_server.collections import (
    CollectionLookupError,
    ExistingCollection,
    GeminiCollectionSelector,
    SupabaseCollectionCatalog,
)
from linq_server.models import AnalyzeRequest, AnalyzeResponse
from linq_server import main


PROJECT = "https://testing.supabase.co"
OTHER_PROJECT = "https://production.supabase.co"
OWNER = "00000000-0000-0000-0000-000000000001"
OTHER_OWNER = "00000000-0000-0000-0000-000000000002"
COLLECTION = "00000000-0000-0000-0000-000000000010"
SECOND_COLLECTION = "00000000-0000-0000-0000-000000000020"
CALLER = Caller(user_id=OWNER, kind="supabase", issuer=f"{PROJECT}/auth/v1")


def analysis(**kwargs) -> AnalyzeResponse:
    return AnalyzeResponse(
        url="https://example.com/kyoto", url_hash="same-shared-content",
        summary="A guide to visiting Kyoto in Japan.", **kwargs,
    )


def model_response(selected: str) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps({"collection_id": selected})}]}}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10, "thoughtsTokenCount": 2},
    }


class CatalogTests(unittest.TestCase):
    def fetch(self, handler, caller=CALLER, keys=None):
        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
        catalog = SupabaseCollectionCatalog(keys or {PROJECT: "sb_publishable_testing"})
        with patch("linq_server.collections.httpx.Client", return_value=client):
            return catalog.fetch(caller, "private-user-jwt")

    def test_owner_scoping_project_routing_and_keyset_pagination(self):
        calls = []

        def handler(request):
            calls.append(request)
            self.assertEqual(request.url.host, "testing.supabase.co")
            self.assertEqual(request.url.params["user_id"], f"eq.{OWNER}")
            self.assertEqual(request.url.params["select"], "id,name,user_id")
            self.assertEqual(request.headers["authorization"], "Bearer private-user-jwt")
            self.assertEqual(request.headers["apikey"], "sb_publishable_testing")
            cursor = request.url.params.get("id")
            rows = {
                None: [{"id": COLLECTION, "name": "Travel", "user_id": OWNER}],
                f"gt.{COLLECTION}": [{"id": SECOND_COLLECTION, "name": "Food", "user_id": OWNER}],
                f"gt.{SECOND_COLLECTION}": [],
            }[cursor]
            return httpx.Response(200, json=rows)

        result = self.fetch(handler, keys={PROJECT: "sb_publishable_testing", OTHER_PROJECT: "sb_publishable_production"})
        self.assertEqual([item.name for item in result], ["Travel", "Food"])
        # Even a server-imposed page cap below PAGE_SIZE must not truncate the list.
        self.assertEqual(len(calls), 3)

    def test_unknown_issuer_or_shared_token_never_sends_credentials(self):
        for caller in (
            Caller(user_id=OWNER, kind="supabase", issuer="https://attacker.example/auth/v1"),
            Caller(user_id=None, kind="shared"),
        ):
            with self.subTest(caller=caller), self.assertRaises(CollectionLookupError):
                self.fetch(lambda _: self.fail("network must not be called"), caller)

    def test_other_owner_rows_are_rejected(self):
        with self.assertRaises(CollectionLookupError):
            self.fetch(lambda _: httpx.Response(200, json=[{"id": COLLECTION, "name": "Private", "user_id": OTHER_OWNER}]))

    def test_redirects_errors_and_invalid_rows_are_rejected(self):
        for response in (
            httpx.Response(302, headers={"location": "https://attacker.example"}),
            httpx.Response(401, json={"message": "private details"}),
            httpx.Response(200, json={"not": "rows"}),
            httpx.Response(200, json=[{"id": "not-a-uuid", "name": "Travel", "user_id": OWNER}]),
            httpx.Response(200, json=[{"id": COLLECTION, "name": "", "user_id": OWNER}]),
        ):
            with self.subTest(status=response.status_code), self.assertRaises(CollectionLookupError):
                self.fetch(lambda _, response=response: response)

    def test_prompt_limit_never_silently_uses_partial_collections(self):
        rows = [
            {"id": str(UUID(int=100 + i)), "name": f"Collection {i}", "user_id": OWNER}
            for i in range(201)
        ]

        def handler(request):
            cursor = int(UUID(request.url.params["id"][3:])) if "id" in request.url.params else 0
            remaining = [row for row in rows if int(UUID(row["id"])) > cursor]
            return httpx.Response(200, json=remaining[:int(request.url.params["limit"])])

        with self.assertRaises(CollectionLookupError):
            self.fetch(handler)

    def test_multi_project_config_never_reuses_single_key(self):
        with patch.dict(os.environ, {"SUPABASE_PUBLISHABLE_KEY": "sb_publishable_testing"}, clear=True):
            catalog = SupabaseCollectionCatalog.from_env([PROJECT, OTHER_PROJECT])
        self.assertEqual(catalog.project_keys, {})
        with patch.dict(os.environ, {"LINQ_SUPABASE_PUBLISHABLE_KEYS": json.dumps({PROJECT: "sb_publishable_testing", "https://attacker.example": "sb_publishable_bad"})}, clear=True):
            catalog = SupabaseCollectionCatalog.from_env([PROJECT, OTHER_PROJECT])
        self.assertEqual(catalog.project_keys, {PROJECT: "sb_publishable_testing"})

    def test_secret_key_is_not_accepted(self):
        with self.assertRaises(CollectionLookupError):
            self.fetch(lambda _: self.fail("network must not be called"), keys={PROJECT: "sb_secret_private"})

    def test_invalid_key_config_fails_closed(self):
        for value in ("not-json", "[]"):
            with self.subTest(value=value), patch.dict(os.environ, {"LINQ_SUPABASE_PUBLISHABLE_KEYS": value}, clear=True):
                with self.assertRaises(CollectionLookupError):
                    SupabaseCollectionCatalog.from_env([PROJECT])

    def test_single_project_key_and_empty_catalog(self):
        with patch.dict(os.environ, {"SUPABASE_PUBLISHABLE_KEY": "sb_publishable_testing"}, clear=True):
            catalog = SupabaseCollectionCatalog.from_env([PROJECT])
        self.assertEqual(catalog.project_keys, {PROJECT: "sb_publishable_testing"})
        self.assertEqual(self.fetch(lambda _: httpx.Response(200, json=[])), [])

    def test_repeated_cursor_is_rejected_instead_of_looping(self):
        row = {"id": COLLECTION, "name": "Travel", "user_id": OWNER}
        with self.assertRaises(CollectionLookupError):
            self.fetch(lambda _: httpx.Response(200, json=[row]))


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.selector = GeminiCollectionSelector(EnrichmentConfig(api_key="fake-test-key"))
        self.choices = [ExistingCollection(UUID(COLLECTION), "Travel")]

    def test_returns_existing_id_and_server_name_with_usage(self):
        with patch.object(self.selector, "_generate", return_value=("gemini-3.5-flash-lite", model_response(COLLECTION))) as generate:
            result = self.selector.select(analysis(), self.choices)
        self.assertEqual(result.status, "selected")
        self.assertEqual(str(result.collection_id), COLLECTION)
        self.assertEqual(result.collection_name, "Travel")
        self.assertEqual((result.input_tokens, result.output_tokens), (100, 12))
        payload = generate.call_args.args[1]
        allowed = payload["generationConfig"]["responseSchema"]["properties"]["collection_id"]["enum"]
        self.assertEqual(allowed, ["NONE", COLLECTION])
        self.assertNotIn("fileData", json.dumps(payload))

    def test_no_match_and_invented_id_are_distinct(self):
        for selected, status in (("NONE", "no_match"), (SECOND_COLLECTION, "unavailable")):
            with self.subTest(selected=selected), patch.object(self.selector, "_generate", return_value=("test", model_response(selected))):
                result = self.selector.select(analysis(), self.choices)
                self.assertEqual(result.status, status)
                self.assertIsNone(result.collection_id)

    def test_empty_degraded_or_unconfigured_inputs_do_not_spend_quota(self):
        with patch.object(self.selector, "_generate") as generate:
            self.assertEqual(self.selector.select(analysis(), []).status, "no_collections")
            self.assertEqual(self.selector.select(analysis(degraded=True), self.choices).status, "unavailable")
            self.selector.config.api_key = None
            self.assertEqual(self.selector.select(analysis(), self.choices).status, "unavailable")
        generate.assert_not_called()

    def test_provider_failures_do_not_discard_enrichment(self):
        with patch.object(self.selector, "_generate", side_effect=ProviderError("private failure")):
            result = self.selector.select(analysis(), self.choices)
        self.assertEqual(result.status, "unavailable")
        self.assertNotIn("private failure", result.model_dump_json())

    def test_malformed_model_output_is_unavailable(self):
        data = {"candidates": [{"content": {"parts": [{"text": "not-json"}]}}]}
        with patch.object(self.selector, "_generate", return_value=("test", data)):
            self.assertEqual(self.selector.select(analysis(), self.choices).status, "unavailable")

    def test_classification_timeout_does_not_mutate_video_configuration(self):
        config = EnrichmentConfig(request_timeout=120)
        selector = GeminiCollectionSelector(config)
        self.assertEqual(selector.config.request_timeout, 20)
        self.assertEqual(config.request_timeout, 120)
        with patch("enrichment.providers.gemini.GeminiProvider._post", return_value={}) as post:
            selector._post("models/test:generateContent", {}, attempts=3)
        post.assert_called_once_with("models/test:generateContent", {}, attempts=1)

    def test_same_content_is_classified_independently_for_each_user(self):
        original = analysis()
        other_choices = [ExistingCollection(UUID(SECOND_COLLECTION), "Holidays")]
        with patch.object(self.selector, "_generate", side_effect=[("test", model_response(COLLECTION)), ("test", model_response(SECOND_COLLECTION))]):
            first = self.selector.select(original, self.choices)
            second = self.selector.select(original, other_choices)
        self.assertNotEqual(first.collection_id, second.collection_id)
        self.assertIsNone(original.collection_suggestion)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.policy = Mock()
        self.policy.identify.return_value = CALLER
        self.policy.supabase.supabase_urls = (PROJECT,)
        self.policy._bearer.return_value = "private-user-jwt"
        self.service = SimpleNamespace(
            analyze=Mock(side_effect=lambda *args, **kwargs: analysis()),
            enrichment_config=EnrichmentConfig(api_key="fake-test-key"),
        )
        self.client = TestClient(main.app)

    def test_old_request_does_not_fetch_or_classify(self):
        self.assertFalse(AnalyzeRequest(url="https://example.com").suggest_collection)
        with patch.object(main, "policy", self.policy), patch.object(main, "service", self.service), patch.object(main, "SupabaseCollectionCatalog") as catalog:
            response = self.client.post("/v1/analyze", json={"url": "https://example.com"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["collection_suggestion"])
        catalog.from_env.assert_not_called()

    def test_opt_in_looks_up_verified_user_and_returns_choice(self):
        choices = [ExistingCollection(UUID(COLLECTION), "Travel")]
        with patch.object(main, "policy", self.policy), patch.object(main, "service", self.service), patch.object(main, "SupabaseCollectionCatalog") as catalog, patch.object(GeminiCollectionSelector, "_generate", return_value=("test", model_response(COLLECTION))):
            catalog.from_env.return_value.fetch.return_value = choices
            response = self.client.post("/v1/analyze", json={"url": "https://example.com", "suggest_collection": True, "user_id": OTHER_OWNER, "collections": [{"id": SECOND_COLLECTION}]})
            catalog.from_env.return_value.fetch.assert_called_once_with(CALLER, "private-user-jwt")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["collection_suggestion"]["collection_id"], COLLECTION)

    def test_lookup_failure_keeps_successful_summary(self):
        with patch.object(main, "policy", self.policy), patch.object(main, "service", self.service), patch.object(main, "SupabaseCollectionCatalog") as catalog:
            catalog.from_env.side_effect = CollectionLookupError("private failure")
            response = self.client.post("/v1/analyze", json={"url": "https://example.com", "suggest_collection": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["summary"], analysis().summary)
        self.assertEqual(response.json()["collection_suggestion"]["status"], "unavailable")
        self.assertNotIn("private failure", response.text)

    def test_shared_credentials_cannot_request_personal_collections(self):
        self.policy.identify.return_value = Caller(user_id=None, kind="shared")
        with patch.object(main, "policy", self.policy), patch.object(main, "service", self.service):
            response = self.client.post("/v1/analyze", json={"url": "https://example.com", "suggest_collection": True})
        self.assertEqual(response.status_code, 403)
        self.service.analyze.assert_not_called()
        self.policy.charge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
