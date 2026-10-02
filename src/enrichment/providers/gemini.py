"""Gemini provider: structured tagging, text or full media.

Talks to the REST API directly over httpx rather than pulling in the `google-genai`
SDK. Two reasons: no extra dependency to keep pinned, and the request bodies stay
visible in this file, which matters when you are debugging why a schema constraint was
ignored.

Three media paths, cheapest first:

* YouTube URLs are passed straight through as `fileData.fileUri`. Google fetches the
  video itself -- nothing is downloaded, nothing is uploaded, and it works even though
  yt-dlp is currently bot-walled on YouTube.
* Any other remote video is streamed down under a size cap and pushed through the
  Files API, then referenced by URI.
* A local path is uploaded the same way, for the case where a client already has the
  bytes.

`mediaResolution` is set to LOW by default. Per Gemini's media resolution docs, LOW and
MEDIUM are billed identically for video at 70 tokens per frame, so LOW is the correct
default and HIGH (280 tokens per frame) is only worth it for text-dense video -- slides,
code walkthroughs, dense on-screen captions -- where reading the frame is the point.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
import time
from pathlib import Path
from typing import Any

import httpx

from extractor import ContentEnvelope

from ..pricing import estimate_cost, estimate_video_tokens, is_priced
from ..prompts import PROMPT_VERSION, SYSTEM_INSTRUCTION, build_media_prompt, build_text_prompt
from ..schema import RESPONSE_SCHEMA, Enrichment, Entities, EnrichmentMode
from ..taxonomy import coerce_category, coerce_content_type, normalize_tags
from ..video import compress_video
from .base import EnrichmentProvider, ProviderError, ProviderUnavailable

log = logging.getLogger("enrichment.gemini")

_YOUTUBE_HOST = re.compile(r"(?:^|\.)(?:youtube\.com|youtu\.be)$", re.IGNORECASE)
_RETRY_STATUSES = {429, 500, 502, 504}
# 404 means the model name is retired for this key; 503 means it is overloaded. Both are
# answered by switching models, not by retrying the same one.
_MODEL_UNAVAILABLE_STATUSES = {404, 503}


class ModelUnavailable(ProviderError):
    """This model cannot serve the request; a different model might."""

# Google's error bodies quote the offending API key back at you verbatim, e.g.
# "Consumer 'api_key:AIza...' has been suspended". Those bodies get stored in the
# enrichment note and printed to the console, so the key has to be scrubbed before it
# is ever attached to an error. Never let a credential travel with a message.
# Two key shapes in the wild: the classic `AIzaSy...` (39 chars) and the newer
# `AQ.Ab8...` AI Studio format (53 chars). The literal-value replacement below is the
# real safety net, but pattern matching also catches keys other than our own appearing
# in a shared error body.
_API_KEY_PATTERN = re.compile(r"\b(?:AIza[0-9A-Za-z_\-]{10,}|AQ\.[0-9A-Za-z_\-]{10,})")


def _redact(text: str, api_key: str | None = None) -> str:
    cleaned = _API_KEY_PATTERN.sub("AIza...REDACTED", text)
    if api_key and len(api_key) > 6:
        cleaned = cleaned.replace(api_key, "...REDACTED")
    return cleaned


class GeminiProvider(EnrichmentProvider):
    name = "gemini"

    def availability(self) -> tuple[bool, str | None]:
        if not self.config.has_api_key:
            return False, "GEMINI_API_KEY not set"
        return True, None

    def supports(self, mode: EnrichmentMode) -> bool:
        if mode is EnrichmentMode.MEDIA:
            return self.config.allow_media
        return mode is EnrichmentMode.METADATA

    # ------------------------------------------------------------------- entrypoint
    def enrich(
        self, envelope: ContentEnvelope, mode: EnrichmentMode, document: str
    ) -> Enrichment:
        available, reason = self.availability()
        if not available:
            raise ProviderUnavailable(reason or "unavailable")

        started = time.perf_counter()
        if mode is EnrichmentMode.MEDIA:
            candidates = [self.config.media_model, self.config.media_fallback_model]
            parts, note = self._media_parts(envelope)
            prompt = build_media_prompt(
                document, envelope.platform.value, envelope.canonical_url,
                envelope.duration_s,
            )
            parts = [{"text": prompt}, *parts]
        else:
            candidates = [self.config.text_model, self.config.text_fallback_model]
            note = "text only"
            prompt = build_text_prompt(
                document, envelope.platform.value, envelope.canonical_url
            )
            parts = [{"text": prompt}]

        payload = self._request_body(parts, mode)
        model, data = self._generate(candidates, payload)
        if model != candidates[0]:
            note = f"{note}; fell back from {candidates[0]}"

        text = _collect_text(data)
        if not text:
            raise ProviderError(f"empty response ({_finish_reason(data)})")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"unparseable JSON ({_finish_reason(data)}): {exc}"
            ) from exc

        usage = data.get("usageMetadata") or {}
        input_tokens = int(usage.get("promptTokenCount") or 0)
        # Thinking tokens bill as output, so fold them in or the cost is understated.
        output_tokens = int(usage.get("candidatesTokenCount") or 0) + int(
            usage.get("thoughtsTokenCount") or 0
        )

        if not is_priced(model):
            # Surface the gap rather than reporting a confident $0.00.
            note = f"{note}; cost unknown: no published rate for {model}"

        return _to_enrichment(
            parsed,
            envelope=envelope,
            mode=mode,
            provider=self.name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(model, input_tokens, output_tokens),
            duration_ms=int((time.perf_counter() - started) * 1000),
            note=note,
        )

    def _generate(
        self, candidates: list[str | None], payload: dict
    ) -> tuple[str, dict]:
        """Call the first model that is actually reachable.

        Model availability is not static and not knowable from the model list: a name
        can be advertised and retired (404 for new keys), or present and temporarily
        overloaded (503). Both are worth rolling past rather than failing the item,
        because the alternative is discarding extraction work that already succeeded.
        """
        models = [m for m in candidates if m]
        last: ProviderError | None = None
        for index, model in enumerate(models):
            is_last = index == len(models) - 1
            # One shot for every model except the last. Retrying a model that just
            # timed out on video is close to pointless and extremely expensive in wall
            # clock: three attempts at a 120s timeout burned six minutes before the
            # fallback -- which then answered in nine seconds -- was even tried.
            attempts = 3 if is_last else 1
            try:
                return model, self._post(
                    f"models/{model}:generateContent", payload, attempts=attempts
                )
            except ModelUnavailable as exc:
                last = exc
                log.debug("model %s unavailable: %s", model, exc)
                if is_last:
                    raise
                continue
        raise last or ProviderError("no usable model configured")

    # ------------------------------------------------------------------- inspection
    def build_request(
        self, envelope: ContentEnvelope, mode: EnrichmentMode, document: str
    ) -> tuple[str, dict[str, Any]]:
        """Build the exact request body without sending it.

        Exists so the request can be inspected, diffed, or replayed by hand with curl
        when the API itself is unavailable -- which is the situation whenever a key is
        missing, suspended, or rate-limited. Media that would require an upload is
        represented by a placeholder rather than being fetched.
        """
        if mode is EnrichmentMode.MEDIA:
            model = self.config.media_model
            if _is_youtube(envelope.canonical_url):
                parts: list[dict[str, Any]] = [
                    {"fileData": {"fileUri": _youtube_uri(envelope.canonical_url)}}
                ]
            else:
                parts = [
                    {
                        "fileData": {
                            "fileUri": "<uploaded via Files API at call time>",
                            "mimeType": "video/mp4",
                        }
                    }
                ]
            prompt = build_media_prompt(
                document, envelope.platform.value, envelope.canonical_url,
                envelope.duration_s,
            )
            parts = [{"text": prompt}, *parts]
        else:
            model = self.config.text_model
            parts = [
                {
                    "text": build_text_prompt(
                        document, envelope.platform.value, envelope.canonical_url
                    )
                }
            ]
        return model, self._request_body(parts, mode)

    # ---------------------------------------------------------------- budget guard
    def _clip_window(self, envelope: ContentEnvelope) -> dict[str, str] | None:
        """Bound the analyzed portion of a video, when needed.

        Returns None when the whole video already fits inside the ceiling, so short
        clips -- reels, shorts, most of what gets saved -- are analyzed in full.
        """
        limit = self.config.max_analyze_seconds
        if limit <= 0:
            return None
        duration = envelope.duration_s or 0.0
        if 0 < duration <= limit:
            return None
        return {"startOffset": "0s", "endOffset": f"{int(limit)}s"}

    def analyzed_seconds(self, envelope: ContentEnvelope) -> float:
        """How much video will actually be sent, accounting for the clip window."""
        limit = self.config.max_analyze_seconds
        duration = envelope.duration_s or 0.0
        if duration <= 0:
            # Unknown duration: the clip window is what we will actually be billed for.
            return limit if limit > 0 else 0.0
        if limit > 0:
            return min(duration, limit)
        return duration

    def preflight_cost(self, envelope: ContentEnvelope) -> float:
        """Predicted cost of a media call, before committing to it.

        Uses the clipped length rather than the raw duration, so the prediction matches
        what is actually sent -- and stays meaningful when duration is unknown.
        """
        input_tokens = estimate_video_tokens(
            self.analyzed_seconds(envelope),
            high_resolution=self.config.high_resolution,
        )
        model = self.config.media_model
        if not is_priced(model):
            model = self.config.media_fallback_model or model
        return estimate_cost(model, input_tokens, 500)

    # -------------------------------------------------------------------- requests
    def _request_body(self, parts: list[dict[str, Any]], mode: EnrichmentMode) -> dict:
        generation: dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            "temperature": 0.2,
            "maxOutputTokens": 2048,
        }
        if mode is EnrichmentMode.MEDIA:
            generation["mediaResolution"] = (
                "MEDIA_RESOLUTION_HIGH"
                if self.config.high_resolution
                else "MEDIA_RESOLUTION_LOW"
            )
        return {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation,
        }

    def _post(self, path: str, payload: dict, attempts: int = 3) -> dict:
        url = f"{self.config.api_base}/{path}"
        headers = {
            "x-goog-api-key": str(self.config.api_key),
            "Content-Type": "application/json",
        }
        last_error: str = "unknown"
        transport_only = True
        for attempt in range(max(1, attempts)):
            try:
                with httpx.Client(timeout=self.config.request_timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    return response.json()
                body = " ".join(response.text.split())[:240]
                last_error = f"HTTP {response.status_code}: {body}"
                transport_only = False
                if response.status_code in _MODEL_UNAVAILABLE_STATUSES:
                    # A retired or overloaded model. Retrying the same name will not
                    # help; the caller should try a different model.
                    raise ModelUnavailable(_redact(last_error, self.config.api_key))
                if response.status_code not in _RETRY_STATUSES:
                    break
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            # Rate limits and 5xx are worth a short backoff; anything else is not.
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))

        message = _redact(last_error, self.config.api_key)
        if transport_only:
            # Every attempt died in transport -- almost always a read timeout while the
            # model chews on video. A model that cannot answer within the timeout is
            # unavailable in practice, so let the caller switch models rather than
            # burning the item.
            raise ModelUnavailable(message)
        raise ProviderError(message)

    # ----------------------------------------------------------------------- media
    def _media_parts(self, envelope: ContentEnvelope) -> tuple[list[dict], str]:
        source = envelope.media_url or envelope.canonical_url

        if _is_youtube(envelope.canonical_url) or _is_youtube(source):
            # Google fetches it directly. No download, no upload, no expiry problem.
            part: dict[str, Any] = {
                "fileData": {"fileUri": _youtube_uri(envelope.canonical_url)}
            }
            clip = self._clip_window(envelope)
            if clip:
                part["videoMetadata"] = clip
            return [part], "youtube uri" + (" (clipped)" if clip else "")

        if not envelope.media_url:
            raise ProviderError("no media URL available for media-mode enrichment")

        local = Path(envelope.media_url)
        if local.exists() and local.is_file():
            blob, mime = local.read_bytes(), _guess_mime(local.name)
            label = "local file"
        else:
            blob, mime = self._download(envelope.media_url)
            label = "downloaded"

        # Shrink before upload. Models sample video at ~1 fps, so full-resolution 30 fps
        # frames are billed detail nobody looks at.
        trimmed_by_ffmpeg = False
        if self.config.compress_video and mime.startswith("video/"):
            result = compress_video(
                blob,
                ffmpeg=self.config.ffmpeg_path,
                max_seconds=self.config.max_analyze_seconds,
                fps=self.config.video_fps,
                height=self.config.video_height,
                crf=self.config.video_crf,
                timeout=self.config.request_timeout,
            )
            blob, mime = result.data, result.mime
            trimmed_by_ffmpeg = result.compressed
            if result.note:
                label = f"{label}, {result.note}"

        uri, mime = self._upload_bytes(blob, mime)
        part: dict[str, Any] = {"fileData": {"fileUri": uri, "mimeType": mime}}

        # ffmpeg already enforced the duration cap, so a clip window would be redundant.
        # Only ask the model to clip when compression did not happen.
        if not trimmed_by_ffmpeg:
            clip = self._clip_window(envelope)
            if clip:
                part["videoMetadata"] = clip
                label = f"{label} (clipped by model)"
        return [part], label

    def _download(self, url: str) -> tuple[bytes, str]:
        """Stream a remote media file under a hard size cap.

        The cap is not politeness -- a single unexpectedly long video is the most
        likely way this pipeline runs away with your money and your disk.
        """
        chunks: list[bytes] = []
        total = 0
        try:
            with httpx.Client(
                timeout=self.config.request_timeout, follow_redirects=True
            ) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    mime = (
                        response.headers.get("content-type", "").split(";")[0].strip()
                        or "video/mp4"
                    )
                    # Refuse anything that is not actually media. Without this check a
                    # bad media URL sends an HTML page to a multimodal model, which
                    # succeeds, costs a fortune in tokens, and returns tags derived from
                    # page furniture instead of footage -- a failure that looks exactly
                    # like success.
                    if not mime.startswith(("video/", "audio/", "image/")) and mime not in {
                        "application/octet-stream"
                    }:
                        raise ProviderError(
                            f"media URL returned {mime!r}, not audio or video: {url[:100]}"
                        )
                    for chunk in response.iter_bytes(1 << 16):
                        total += len(chunk)
                        if total > self.config.max_media_bytes:
                            raise ProviderError(
                                f"media exceeds {self.config.max_media_bytes} bytes"
                            )
                        chunks.append(chunk)
        except httpx.HTTPError as exc:
            raise ProviderError(
                _redact(f"media download failed: {exc}", self.config.api_key)
            ) from exc
        if not chunks:
            raise ProviderError("media download returned no data")
        return b"".join(chunks), mime

    @property
    def _upload_base(self) -> str:
        """Base URL for the Files API resumable upload protocol.

        Uploads do not live under the normal API base. They are served from an `/upload`
        path prefix -- `/upload/v1beta/files` rather than `/v1beta/files` -- and posting
        the start request to the ordinary base returns 200 with no `x-goog-upload-url`
        header, so the failure surfaces as "Files API did not return an upload URL"
        rather than as a routing error.
        """
        base = self.config.api_base.rstrip("/")
        marker = "://"
        index = base.find(marker)
        if index == -1:
            return base
        host_end = base.find("/", index + len(marker))
        if host_end == -1:
            return f"{base}/upload"
        return f"{base[:host_end]}/upload{base[host_end:]}"

    def _upload_bytes(self, blob: bytes, mime: str) -> tuple[str, str]:
        """Files API resumable upload, then wait for the file to become ACTIVE.

        The wait is required, not optional: video is processed asynchronously and
        referencing a PROCESSING file in generateContent fails.
        """
        base = self._upload_base
        key = str(self.config.api_key)
        start_headers = {
            "x-goog-api-key": key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(blob)),
            "X-Goog-Upload-Header-Content-Type": mime,
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.config.request_timeout) as client:
                start = client.post(
                    f"{base}/files",
                    headers=start_headers,
                    json={"file": {"display_name": "link-memory-media"}},
                )
                start.raise_for_status()
                upload_url = start.headers.get("x-goog-upload-url")
                if not upload_url:
                    raise ProviderError("Files API did not return an upload URL")

                finish = client.post(
                    upload_url,
                    headers={
                        "Content-Length": str(len(blob)),
                        "X-Goog-Upload-Offset": "0",
                        "X-Goog-Upload-Command": "upload, finalize",
                    },
                    content=blob,
                )
                finish.raise_for_status()
                info = (finish.json() or {}).get("file") or {}
        except httpx.HTTPError as exc:
            raise ProviderError(
                _redact(f"media upload failed: {exc}", self.config.api_key)
            ) from exc

        uri = info.get("uri")
        name = info.get("name")
        if not uri or not name:
            raise ProviderError("Files API response missing file uri")

        self._await_active(name)
        return str(uri), str(info.get("mimeType") or mime)

    def _await_active(self, name: str) -> None:
        deadline = time.monotonic() + self.config.upload_poll_timeout_s
        headers = {"x-goog-api-key": str(self.config.api_key)}
        delay = 1.0
        while time.monotonic() < deadline:
            try:
                with httpx.Client(timeout=self.config.request_timeout) as client:
                    response = client.get(
                        f"{self.config.api_base}/{name}", headers=headers
                    )
                    response.raise_for_status()
                    state = (response.json() or {}).get("state")
            except httpx.HTTPError as exc:
                raise ProviderError(
                    _redact(f"file state check failed: {exc}", self.config.api_key)
                ) from exc

            if state == "ACTIVE":
                return
            if state == "FAILED":
                raise ProviderError("Files API reported processing FAILED")
            time.sleep(delay)
            delay = min(delay * 1.5, 8.0)
        raise ProviderError("timed out waiting for uploaded media to become ACTIVE")


# --------------------------------------------------------------------------- helpers
def _is_youtube(url: str | None) -> bool:
    if not url:
        return False
    from urllib.parse import urlparse

    return bool(_YOUTUBE_HOST.search(urlparse(url).netloc or ""))


def _youtube_uri(url: str) -> str:
    """Rebuild a canonical `www.youtube.com/watch?v=ID` URL.

    Canonicalization strips `www.` for deduplication, which already turned out to break
    yt-dlp's URL matching. Rather than find out the hard way whether Google's fetcher is
    equally particular, hand it the documented form.
    """
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    video_id = parse_qs(parsed.query).get("v", [""])[0]
    if not video_id:
        video_id = parsed.path.strip("/").split("/")[-1]
    if not video_id:
        return url
    return f"https://www.youtube.com/watch?v={video_id}"


def _guess_mime(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "video/mp4"


def _collect_text(data: dict) -> str:
    """Join every text part. Thinking models emit non-text parts alongside the answer."""
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    return "".join(str(part.get("text", "")) for part in parts if "text" in part).strip()


def _finish_reason(data: dict) -> str:
    candidates = data.get("candidates") or []
    if candidates:
        return str(candidates[0].get("finishReason") or "unknown")
    feedback = data.get("promptFeedback") or {}
    if feedback.get("blockReason"):
        return f"blocked: {feedback['blockReason']}"
    return "no candidates"


def _to_enrichment(
    parsed: dict[str, Any],
    *,
    envelope: ContentEnvelope,
    mode: EnrichmentMode,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    duration_ms: int,
    note: str,
) -> Enrichment:
    """Map raw model JSON onto the enrichment contract.

    Platform hashtags are merged into the model's tags rather than replaced. They are
    free, already human-authored, and often name the specific thing ('#orecchiette')
    that a model describes only generically.
    """
    tags = normalize_tags(
        list(parsed.get("tags") or []) + list(envelope.platform_tags), limit=15
    )
    entities = Entities(
        people=[str(v) for v in (parsed.get("people") or [])][:8],
        organizations=[str(v) for v in (parsed.get("organizations") or [])][:8],
        places=[str(v) for v in (parsed.get("places") or [])][:8],
        products=[str(v) for v in (parsed.get("products") or [])][:10],
    )
    details = [
        " ".join(str(item).split())
        for item in (parsed.get("details") or [])
        if str(item).strip()
    ][:30]

    return Enrichment(
        url_hash=envelope.url_hash,
        summary=" ".join(str(parsed.get("summary") or "").split()),
        description=" ".join(str(parsed.get("description") or "").split()),
        details=details,
        tags=tags,
        category=coerce_category(parsed.get("category")),
        content_type=coerce_content_type(parsed.get("content_type")),
        entities=entities,
        language=(str(parsed["language"]) if parsed.get("language") else None),
        transcript_excerpt=envelope.transcript[:400] if envelope.transcript else None,
        mode=mode,
        provider=provider,
        model=model,
        prompt_version=PROMPT_VERSION,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        duration_ms=duration_ms,
        degraded=False,
        note=note,
    )
