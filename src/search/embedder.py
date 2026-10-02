"""Embedding providers.

Two backends behind one interface: Google's native Gemini endpoint and any
OpenAI-compatible `/embeddings` endpoint (used for OpenRouter). Which one runs is chosen by
`SEARCH_EMBED_PROVIDER`; both stay available so a withdrawn free model or an exhausted quota
is a config change rather than an outage.

Three details here are the difference between search that works and search that looks like
it works.

**Normalization.** Verified against the live APIs: `gemini-embedding-001` truncated to 768
dimensions returns vectors with L2 norm around 0.59, while `nemotron-3-embed-1b` returns
exactly 1.000. Cosine similarity equals a dot product only for unit vectors, so mixing the
two produces rankings that are subtly and silently wrong -- documents winning on magnitude
rather than direction. Everything is normalized here, explicitly, whatever the model claims.

**Asymmetric task types.** Gemini is told `RETRIEVAL_DOCUMENT` for passages and
`RETRIEVAL_QUERY` for queries, because these models are trained to place a short question
near the long passage answering it, and that only happens if you say which side you are
embedding. OpenAI-compatible endpoints have no such parameter, so nothing is sent.

**No offline fallback.** A non-semantic stand-in embedder would return plausible-looking
neighbours that mean nothing, which is worse than an honest "vector search unavailable". With
no key, search degrades to keyword-only and says so.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from abc import ABC, abstractmethod

import httpx
import numpy as np

from .config import SearchConfig

log = logging.getLogger("search.embedder")

_RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
_API_KEY_PATTERN = re.compile(
    r"\b(?:AIza[0-9A-Za-z_\-]{10,}|AQ\.[0-9A-Za-z_\-]{10,}"
    r"|sk-or-v1-[0-9a-f]{16,}|nvapi-[0-9A-Za-z_\-]{16,})"
)

# USD per 1M input tokens, for the cost estimate only. The free providers are 0.
EMBED_RATES: dict[str, float] = {
    "nvidia/nemotron-3-embed-1b": 0.0,
    "nvidia/nemotron-3-embed-1b:free": 0.0,
    "liquid/lfm-2.5-embedding-350m:free": 0.0,
    "openai/text-embedding-3-small": 0.02,
    "baai/bge-m3": 0.01,
    "qwen/qwen3-embedding-8b": 0.01,
    "gemini-embedding-001": 0.15,
    "gemini-embedding-2": 0.15,
}


class EmbedderError(Exception):
    """Embedding failed. Callers degrade to keyword-only rather than failing outright."""


class EmbedderUnavailable(EmbedderError):
    """No API key, so vector search cannot run at all."""


def _redact(text: str, api_key: str | None) -> str:
    cleaned = _API_KEY_PATTERN.sub("...REDACTED", text)
    if api_key and len(api_key) > 6:
        cleaned = cleaned.replace(api_key, "...REDACTED")
    return cleaned


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so cosine similarity reduces to a dot product.

    Zero-length rows are left alone rather than producing NaN, which would poison every
    comparison against them.
    """
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        return matrix if norm == 0 else (matrix / norm).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(1.0, min(float(raw), 120.0))
    except ValueError:
        return None


class BaseEmbedder(ABC):
    """Shared caching, pacing and normalization for every embedding backend."""

    name = "embedder"

    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig.from_env()
        self.estimated_tokens = 0
        # Query cache, keyed by exact text. Search-as-you-type reissues heavily overlapping
        # queries -- typing, pausing, backspacing and retyping one word easily produces a
        # dozen requests for text already embedded.
        self._query_cache: dict[str, np.ndarray] = {}
        self._query_cache_limit = 512
        self.query_cache_hits = 0
        self._last_bulk_request = 0.0
        self._pace_lock = threading.Lock()

    # ---------------------------------------------------------------- availability
    def availability(self) -> tuple[bool, str | None]:
        if not self.config.api_key:
            return False, f"{self.name} API key not set; vector search disabled"
        return True, None

    @property
    def signature(self) -> tuple[str, int]:
        """Model and dimension pair stored vectors must match to be comparable."""
        return self.config.embed_model, self.config.dimensions

    @property
    def rate_usd_per_million(self) -> float:
        return EMBED_RATES.get(self.config.embed_model, 0.0)

    @property
    def estimated_cost_usd(self) -> float:
        return round(self.estimated_tokens * self.rate_usd_per_million / 1_000_000, 8)

    # -------------------------------------------------------------------- embedding
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages for indexing. Returns an (n, dim) float32 matrix, paced."""
        return self._embed(texts, is_query=False, pace=True)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query. Returns a (dim,) float32 vector. Cached by exact text."""
        model, dims = self.signature
        key = f"{model}@{dims}:{text.strip().lower()}"
        cached = self._query_cache.get(key)
        if cached is not None:
            self.query_cache_hits += 1
            return cached

        vector = self._embed([text], is_query=True)[0]
        if len(self._query_cache) >= self._query_cache_limit:
            # Plain FIFO eviction; a true LRU is not worth the bookkeeping here.
            self._query_cache.pop(next(iter(self._query_cache)), None)
        self._query_cache[key] = vector
        return vector

    def _pace(self) -> None:
        rpm = self.config.embed_requests_per_minute
        if rpm <= 0:
            return
        interval = 60.0 / rpm
        with self._pace_lock:
            elapsed = time.monotonic() - self._last_bulk_request
            if self._last_bulk_request and elapsed < interval:
                time.sleep(interval - elapsed)
            self._last_bulk_request = time.monotonic()

    def _embed(self, texts: list[str], *, is_query: bool, pace: bool = False) -> np.ndarray:
        available, reason = self.availability()
        if not available:
            raise EmbedderUnavailable(reason or "unavailable")
        if not texts:
            return np.zeros((0, self.config.dimensions), dtype=np.float32)

        vectors: list[list[float]] = []
        batch = max(1, self.config.embed_batch_size)
        for start in range(0, len(texts), batch):
            window = texts[start : start + batch]
            if pace:
                self._pace()
            vectors.extend(self._embed_batch(window, is_query=is_query))
            self.estimated_tokens += sum(max(1, len(t) // 4) for t in window)

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.config.dimensions:
            got = matrix.shape[1] if matrix.ndim == 2 else "?"
            raise EmbedderError(
                f"{self.config.embed_model} returned {got} dims, expected "
                f"{self.config.dimensions}; set SEARCH_DIMENSIONS to match and re-index"
            )
        return normalize(matrix)

    @abstractmethod
    def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        """One request. Returns raw vectors; normalization happens in `_embed`."""

    # --------------------------------------------------------------------- requests
    def _post(self, url: str, payload: dict, headers: dict, attempts: int = 5) -> dict:
        """POST with retries. `attempts` is deliberately caller-controlled.

        Bulk indexing can afford to wait out a rate limit; an interactive query cannot. The
        free tier's per-minute cap turned a single search into a 100-second stall by
        retrying five times with 20-second backoffs, and because the search endpoint was
        blocking, that stalled every other request too. Queries now use a small budget and
        degrade to keyword-only instead.
        """
        last_error = "unknown"
        attempts = max(1, attempts)
        for attempt in range(attempts):
            try:
                with httpx.Client(timeout=self.config.request_timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    return response.json()
                body = " ".join(response.text.split())[:240]
                last_error = f"HTTP {response.status_code}: {body}"
                if response.status_code not in _RETRY_STATUSES:
                    break

                if response.status_code == 429:
                    quota = _quota_violation(response)
                    if quota:
                        last_error = f"quota exhausted: {quota}"
                        # A daily quota cannot clear by waiting inside this call. Both
                        # providers word it differently -- Google sends
                        # `...PerDayPerProjectPerModel`, OpenRouter sends
                        # `free-models-per-day` -- so match case-insensitively on both. A
                        # missed match here cost 20 seconds per search before falling back
                        # to keyword, which reads as a hang rather than a quota message.
                        flat = quota.replace("-", "").replace("_", "").lower()
                        if "perday" in flat:
                            break
                    delay = _retry_after(response) or min(20.0 * (attempt + 1), 60.0)
                else:
                    delay = 1.5 * (attempt + 1)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                delay = 1.5 * (attempt + 1)

            if attempt < attempts - 1:
                log.debug("embed retry in %.1fs: %s", delay, last_error)
                time.sleep(delay)
        raise EmbedderError(_redact(last_error, self.config.api_key))


def _quota_violation(response: httpx.Response) -> str | None:
    """Extract the quota id and limit from a 429 body.

    Google's 429 message is generic boilerplate pointing at a docs page; the useful part is
    buried in `error.details` as a QuotaFailure. Surfacing it turns "you exceeded your quota"
    into "1000 embed requests per day per model", which is the difference between guessing
    and knowing. OpenRouter instead names its limit inline, e.g. `free-models-per-min`.
    """
    try:
        error = response.json().get("error") or {}
    except Exception:
        return None

    message = str(error.get("message") or "")
    if "per-min" in message or "per-day" in message:
        return message[:120]

    for detail in error.get("details") or []:
        if "QuotaFailure" not in str(detail.get("@type", "")):
            continue
        for violation in detail.get("violations") or []:
            quota_id = violation.get("quotaId") or violation.get("quotaMetric") or ""
            value = violation.get("quotaValue")
            if quota_id:
                return f"{quota_id}" + (f" (limit {value})" if value else "")
    return None


class GeminiEmbedder(BaseEmbedder):
    """Google's native `batchEmbedContents` endpoint."""

    name = "gemini"

    def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        model = self.config.embed_model
        task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        request: dict = {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task,
                    **(
                        {"outputDimensionality": self.config.dimensions}
                        if self.config.supports_dimensions
                        else {}
                    ),
                }
                for text in texts
            ]
        }
        data = self._post(
            f"{self.config.api_base}/models/{model}:batchEmbedContents",
            request,
            {"x-goog-api-key": str(self.config.api_key), "Content-Type": "application/json"},
            attempts=self.config.query_retry_attempts if is_query else 5,
        )
        embeddings = data.get("embeddings") or []
        if len(embeddings) != len(texts):
            raise EmbedderError(f"expected {len(texts)} embeddings, got {len(embeddings)}")
        return [list(item.get("values") or []) for item in embeddings]


class OpenAICompatEmbedder(BaseEmbedder):
    """Any OpenAI-compatible `/embeddings` endpoint. Used for OpenRouter."""

    name = "openrouter"

    def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        payload: dict = {"model": self.config.embed_model, "input": texts}
        # Only send `dimensions` where the model actually supports truncation; models that
        # do not will reject the parameter outright.
        if self.config.supports_dimensions:
            payload["dimensions"] = self.config.dimensions

        data = self._post(
            f"{self.config.api_base}/embeddings",
            payload,
            {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            attempts=self.config.query_retry_attempts if is_query else 5,
        )
        rows = data.get("data") or []
        if len(rows) != len(texts):
            raise EmbedderError(f"expected {len(texts)} embeddings, got {len(rows)}")
        # Order is not guaranteed by the spec, so sort by the returned index.
        rows.sort(key=lambda row: row.get("index", 0))
        return [list(row.get("embedding") or []) for row in rows]


class NvidiaEmbedder(BaseEmbedder):
    """NVIDIA NIM `/v1/embeddings`, which supports proper asymmetric input types.

    The one thing this has that the OpenAI-compatible route does not is `input_type`:
    `passage` when indexing, `query` when searching. NVIDIA warns that getting it wrong causes
    "large drops in retrieval accuracy".

    Measured on the real library, the honest picture is narrower than that warning suggests.
    OpenRouter silently embeds everything in query mode, and query/query benchmarked level
    with correct passage/query -- 9/9 correct either way, noise floor 0.108 vs 0.110. The
    genuinely bad configuration is passage/passage at 0.264, which nothing was doing. So this
    is worth using because it is correct and free, not because it rescues the index.

    The real reason to prefer this endpoint is quota: 45 rapid requests passed with no 429,
    against OpenRouter's free tier cap of 50 per day.
    """

    name = "nvidia"

    def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        payload: dict = {
            "input": texts,
            "model": self.config.embed_model,
            # The asymmetric mode. Indexing produces passages; searching produces queries.
            "input_type": "query" if is_query else "passage",
            # Stated explicitly rather than letting the server infer it. Inference looks for
            # `data:image/...` prefixes, and a caption that happens to contain one would be
            # silently treated as an image.
            "modality": "text",
        }
        if self.config.supports_dimensions:
            payload["dimensions"] = self.config.dimensions

        data = self._post(
            f"{self.config.api_base}/embeddings",
            payload,
            {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            attempts=self.config.query_retry_attempts if is_query else 5,
        )
        rows = data.get("data") or []
        if len(rows) != len(texts):
            raise EmbedderError(f"expected {len(texts)} embeddings, got {len(rows)}")
        rows.sort(key=lambda row: row.get("index", 0))
        return [list(row.get("embedding") or []) for row in rows]


def build_embedder(config: SearchConfig | None = None) -> BaseEmbedder:
    """Construct the embedder for the configured provider."""
    config = config or SearchConfig.from_env()
    if config.embed_kind == "gemini":
        return GeminiEmbedder(config)
    if config.embed_kind == "nvidia":
        return NvidiaEmbedder(config)
    return OpenAICompatEmbedder(config)
