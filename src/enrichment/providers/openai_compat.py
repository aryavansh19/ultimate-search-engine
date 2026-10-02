"""Provider for any OpenAI-compatible chat endpoint. Used for the Ashna gateway.

Exists because the Gemini free tier's daily quotas are the binding constraint on this
project, and a gateway with higher limits can carry the text tagging load. The Gemini
provider is untouched and still handles everything this one cannot.

Verified against the live Ashna gateway (`api.ashna.ai/v1/api`) before writing this:

* `GET /models` lists 68 chat models, including Gemini, GPT, Claude and Llama families.
* `response_format: {"type": "json_schema"}` works and returns clean JSON with no code
  fences, and it honours the `enum` constraints on category and content_type.
* Image input via a base64 data URL works.
* **No `/embeddings` endpoint** -- it returns 404 with an HTML body. Search vectors have to
  stay on Gemini.
* **No video input.** A `video_url` content part is accepted with HTTP 200 and then
  silently dropped; the model replies that it cannot watch video. Worse, given only the URL
  as text it produced a confident, detailed description of a well-known video from training
  knowledge alone. For an arbitrary reel that same path fabricates, so this provider
  declines MEDIA mode outright rather than risk inventing what is in someone's video.

One practical detail: `max_tokens` has to be generous. At 400 the response came back HTTP
200 with empty content, having spent the whole budget on reasoning tokens -- a failure that
looks like a working call returning nothing.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from extractor import ContentEnvelope

from ..prompts import PROMPT_VERSION, SYSTEM_INSTRUCTION, build_text_prompt
from ..schema import Enrichment, Entities, EnrichmentMode, openai_response_format
from ..taxonomy import coerce_category, coerce_content_type, normalize_tags
from .base import EnrichmentProvider, ProviderError, ProviderUnavailable

log = logging.getLogger("enrichment.openai_compat")

_RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_BEARER = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}")


def _redact(text: str, api_key: str | None) -> str:
    cleaned = _BEARER.sub(r"\1...REDACTED", text)
    if api_key and len(api_key) > 6:
        cleaned = cleaned.replace(api_key, "...REDACTED")
    return cleaned


class OpenAICompatProvider(EnrichmentProvider):
    """Text enrichment through an OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(self, config) -> None:  # noqa: ANN001 - EnrichmentConfig, avoids a cycle
        super().__init__(config)
        self.name = config.compat_name or "openai-compat"

    # ---------------------------------------------------------------- availability
    def availability(self) -> tuple[bool, str | None]:
        if not self.config.compat_api_key:
            return False, f"{self.name.upper()}_API_KEY not set"
        if not self.config.compat_base_url:
            return False, f"{self.name} base URL not configured"
        return True, None

    def supports(self, mode: EnrichmentMode) -> bool:
        # Text only, on purpose. See the module docstring: this endpoint accepts video
        # parts and silently ignores them, which would produce invented descriptions.
        return mode is EnrichmentMode.METADATA

    # ------------------------------------------------------------------- entrypoint
    def enrich(
        self, envelope: ContentEnvelope, mode: EnrichmentMode, document: str
    ) -> Enrichment:
        available, reason = self.availability()
        if not available:
            raise ProviderUnavailable(reason or "unavailable")
        if mode is not EnrichmentMode.METADATA:
            raise ProviderError(f"{self.name} cannot analyse media; text only")

        started = time.perf_counter()
        model = self.config.compat_model
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {
                    "role": "user",
                    "content": build_text_prompt(
                        document, envelope.platform.value, envelope.canonical_url
                    ),
                },
            ],
            "response_format": openai_response_format(),
            "max_tokens": self.config.compat_max_tokens,
            "temperature": 0.2,
        }

        data = self._post("chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("no choices returned")

        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        finish = choices[0].get("finish_reason")
        if not content:
            raise ProviderError(
                f"empty content (finish_reason={finish}); try raising "
                f"COMPAT_MAX_TOKENS above {self.config.compat_max_tokens}"
            )

        try:
            parsed = json.loads(_FENCE.sub("", content))
        except json.JSONDecodeError as exc:
            raise ProviderError(f"unparseable JSON (finish_reason={finish}): {exc}") from exc

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )

        details = [
            " ".join(str(item).split())
            for item in (parsed.get("details") or [])
            if str(item).strip()
        ][:30]
        tags = normalize_tags(
            list(parsed.get("tags") or []) + list(envelope.platform_tags), limit=15
        )

        return Enrichment(
            url_hash=envelope.url_hash,
            summary=" ".join(str(parsed.get("summary") or "").split()),
            description=" ".join(str(parsed.get("description") or "").split()),
            details=details,
            tags=tags,
            category=coerce_category(parsed.get("category")),
            content_type=coerce_content_type(parsed.get("content_type")),
            entities=Entities(
                people=[str(v) for v in (parsed.get("people") or [])][:8],
                organizations=[str(v) for v in (parsed.get("organizations") or [])][:8],
                places=[str(v) for v in (parsed.get("places") or [])][:8],
                products=[str(v) for v in (parsed.get("products") or [])][:10],
            ),
            language=(str(parsed["language"]) if parsed.get("language") else None),
            transcript_excerpt=envelope.transcript[:400] if envelope.transcript else None,
            mode=EnrichmentMode.METADATA,
            provider=self.name,
            model=model,
            prompt_version=PROMPT_VERSION,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            # No public per-token rate table for a third-party gateway, so cost is left at
            # zero and labelled rather than guessed. A fabricated number in a cost report
            # is worse than an honest gap.
            cost_usd=0.0,
            duration_ms=int((time.perf_counter() - started) * 1000),
            degraded=False,
            note=f"text via {self.name}:{model}; cost not tracked (no published rates)",
        )

    # --------------------------------------------------------------------- requests
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{str(self.config.compat_base_url).rstrip('/')}/{path}"
        headers = {
            "Authorization": f"Bearer {self.config.compat_api_key}",
            "Content-Type": "application/json",
        }
        last_error = "unknown"
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.config.request_timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    return response.json()
                body = " ".join(response.text.split())[:240]
                last_error = f"HTTP {response.status_code}: {body}"
                if response.status_code not in _RETRY_STATUSES:
                    break
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
        raise ProviderError(_redact(last_error, self.config.compat_api_key))
