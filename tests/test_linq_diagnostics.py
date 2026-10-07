from __future__ import annotations

import logging
import os
import unittest
from unittest.mock import patch

from linq_server.diagnostics import analyze_request_payload, configure_logging, log_json
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

    def test_configure_logging_reuses_uvicorn_handlers(self) -> None:
        uvicorn_logger = logging.getLogger("uvicorn.error")
        linq_logger = logging.getLogger("linq")
        old_uvicorn_handlers = list(uvicorn_logger.handlers)
        old_linq_handlers = list(linq_logger.handlers)
        old_propagate = linq_logger.propagate
        handler = logging.NullHandler()
        try:
            uvicorn_logger.handlers = [handler]
            with patch.dict(os.environ, {"LINQ_LOG_LEVEL": "INFO"}, clear=True):
                configured_level = configure_logging()
            self.assertEqual(configured_level, logging.INFO)
            self.assertEqual(linq_logger.handlers, [handler])
            self.assertFalse(linq_logger.propagate)
        finally:
            uvicorn_logger.handlers = old_uvicorn_handlers
            linq_logger.handlers = old_linq_handlers
            linq_logger.propagate = old_propagate


if __name__ == "__main__":
    unittest.main()
