from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from extractor.base import Extractor
from extractor.cascade import ExtractionCascade
from extractor.config import ExtractorConfig
from extractor.envelope import (
    ExtractionTarget,
    ExtractionTier,
    MediaKind,
    Platform,
)
from extractor.extractors.client_payload import ClientPayloadExtractor
from extractor.extractors.ytdlp import _safe_log_message
from linq_server.models import AnalyzeRequest


class AnalyzeRequestClientPayloadTests(unittest.TestCase):
    def test_description_is_forwarded_as_caption(self) -> None:
        request = AnalyzeRequest(
            url="https://www.instagram.com/reel/example/",
            title="Creator on Instagram",
            description="A signed-in iPhone extracted this useful reel caption.",
            keywords=["reel", "creator"],
        )

        self.assertTrue(request.has_substantial_content())
        self.assertEqual(
            request.client_payload(),
            {
                "title": "Creator on Instagram",
                "caption": "A signed-in iPhone extracted this useful reel caption.",
                "tags": ["reel", "creator"],
            },
        )

    def test_title_and_keywords_alone_remain_thin_metadata(self) -> None:
        request = AnalyzeRequest(
            url="https://example.com/item",
            title="A title that must not short-circuit extraction",
            keywords=["example"],
        )

        self.assertFalse(request.has_substantial_content())
        self.assertEqual(request.client_payload(), {})

    def test_whitespace_description_is_not_content(self) -> None:
        request = AnalyzeRequest(
            url="https://example.com/item",
            description="  \n\t ",
        )

        self.assertFalse(request.has_substantial_content())
        self.assertEqual(request.client_payload(), {})

    def test_blank_caption_falls_back_to_description(self) -> None:
        request = AnalyzeRequest(
            url="https://www.instagram.com/reel/example/",
            caption="  ",
            description="  A useful caption from Link Presentation.  ",
        )

        self.assertEqual(
            request.client_payload()["caption"],
            "A useful caption from Link Presentation.",
        )


class YtDlpLoggingTests(unittest.TestCase):
    def test_signed_query_values_are_redacted(self) -> None:
        message = (
            "failed https://cdn.example/video.mp4?token=secret&expires=12345 "
            "after a remote error"
        )

        safe = _safe_log_message(message)

        self.assertNotIn("secret", safe)
        self.assertNotIn("12345", safe)
        self.assertIn("token=<redacted>", safe)
        self.assertIn("expires=<redacted>", safe)


class _RichVideoExtractor(Extractor):
    tier = ExtractionTier.YT_DLP

    def __init__(self, config: ExtractorConfig) -> None:
        super().__init__(config)
        self.called = False

    def extract(self, target: ExtractionTarget):
        self.called = True
        return target.envelope(
            self.tier,
            media_url="https://cdn.example/reel.mp4",
            media_kind=MediaKind.VIDEO,
            transcript=(
                "This transcript contains enough useful words to prove that the richer "
                "video extraction tier ran after receiving the client caption fallback."
            ),
        )


class _FailingVideoExtractor(Extractor):
    tier = ExtractionTier.YT_DLP

    def __init__(self, config: ExtractorConfig) -> None:
        super().__init__(config)
        self.called = False

    def extract(self, target: ExtractionTarget):
        self.called = True
        return None


class ClientPayloadCascadeTests(unittest.TestCase):
    @staticmethod
    def _target() -> ExtractionTarget:
        return ExtractionTarget(
            canonical_url="https://www.instagram.com/reel/example/",
            url_hash="example-hash",
            platform=Platform.INSTAGRAM,
            original_url="https://www.instagram.com/reel/example/",
            client_payload={
                "caption": "A useful caption extracted on the signed-in iPhone client."
            },
        )

    def test_weak_client_video_metadata_does_not_block_richer_extraction(self) -> None:
        with TemporaryDirectory() as directory:
            config = ExtractorConfig(cache_path=Path(directory) / "cache.sqlite3")
            richer = _RichVideoExtractor(config)
            cascade = ExtractionCascade(
                config=config,
                extractors=[ClientPayloadExtractor(config), richer],
            )
            try:
                result = cascade._run_cascade(self._target())
            finally:
                cascade.close()

        self.assertTrue(richer.called)
        self.assertEqual(
            result.envelope.caption,
            "A useful caption extracted on the signed-in iPhone client.",
        )
        self.assertIn("richer video extraction", result.attempts[0].reason or "")
        self.assertFalse(result.envelope.degraded)

    def test_client_caption_remains_a_non_degraded_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            config = ExtractorConfig(cache_path=Path(directory) / "cache.sqlite3")
            failing = _FailingVideoExtractor(config)
            cascade = ExtractionCascade(
                config=config,
                extractors=[ClientPayloadExtractor(config), failing],
            )
            try:
                result = cascade._run_cascade(self._target())
            finally:
                cascade.close()

        self.assertTrue(failing.called)
        self.assertFalse(result.envelope.degraded)
        self.assertEqual(result.envelope.tier, ExtractionTier.CLIENT_PAYLOAD)
        self.assertTrue(result.envelope.reprocessable)


if __name__ == "__main__":
    unittest.main()
