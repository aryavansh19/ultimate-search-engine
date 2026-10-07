from __future__ import annotations

import logging
import os
import unittest
from unittest.mock import patch

from linq_server.diagnostics import analyze_request_payload, log_json
from linq_server.models import AnalyzeRequest


class DiagnosticsTests(unittest.TestCase):
    def test_request_payload_excludes_rendered_html(self) -> None:
        payload = analyze_request_payload(
            AnalyzeRequest(
                url="https://example.com/post",
                title="Example",
                html="<html>private page</html>",
                embed=False,
            )
        )
        self.assertNotIn("html", payload)
        self.assertEqual(payload["html_chars"], 25)
        self.assertEqual(payload["title"], "Example")
        self.assertFalse(payload["embed"])

    def test_payload_logging_is_disabled_by_default(self) -> None:
        logger = logging.getLogger("linq.test.disabled")
        with patch.dict(os.environ, {}, clear=True), self.assertNoLogs(logger):
            log_json(logger, "trace", "payload", {"title": "private"})

    def test_payload_logging_pretty_prints_when_enabled(self) -> None:
        logger = logging.getLogger("linq.test.enabled")
        with patch.dict(os.environ, {"LINQ_LOG_PAYLOADS": "1"}, clear=True):
            with self.assertLogs(logger, level="INFO") as captured:
                log_json(logger, "abc123", "response", {"ok": True})
        output = "\n".join(captured.output)
        self.assertIn("[search-pipeline:abc123] response", output)
        self.assertIn('"ok": true', output)


if __name__ == "__main__":
    unittest.main()
