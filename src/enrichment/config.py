"""Enrichment configuration.

Kept separate from the extractor's config so the two stages stay independently
deployable -- the extraction step is going to move onto the iOS client eventually,
and enrichment is not.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# override=True on purpose. python-dotenv defaults to leaving pre-existing OS
# environment variables alone, which is the wrong precedence for project-local config:
# this machine already has a stale GEMINI_API_KEY exported at the user level, so with
# the default a correct key pasted into .env would be silently ignored in favour of the
# broken one. Failures like that read as "the API is down" and cost hours.
load_dotenv(override=True)

# Cheap text tagging vs multimodal video understanding. The whole point of the
# signal gate in the envelope is that most items only ever need the first one.
#
# Both defaults were chosen by calling the API rather than reading the model list.
# That distinction matters: `gemini-2.5-flash-lite` is still advertised by ListModels
# but returns 404 "no longer available to new users" on a real request, so the cheapest
# published rate is not actually reachable on a new key.
DEFAULT_TEXT_MODEL = "gemini-3.5-flash-lite"
# Measured on a real 60-second clipped YouTube request:
#
#   gemini-3.7-flash        timed out / 503 on every attempt
#   gemini-3.5-flash        answered well: "explains how the human brain recognizes
#                           handwritten digits and introduces the challenge"
#   gemini-3.5-flash-lite   answered, but shallower: "a blue-eyed pi character appears"
#
# So 3.7-flash is not usable for video at all. Between the other two, 3.5-flash gave
# the better one-sentence answer on a 60s clip -- but on a full 180s window it timed out
# repeatedly, while flash-lite answered in seconds and still surfaced specifics the
# title never mentions (`mnist`, `perceptron`). Reliability and 5x lower cost beat a
# marginally richer description that often fails to arrive, so lite leads and 3.5-flash
# stands behind it.
DEFAULT_MEDIA_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MEDIA_FALLBACK = "gemini-3.5-flash"

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _providers(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse an ordered provider preference list from a comma-separated value."""
    raw = os.getenv(name, "")
    chosen = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    return chosen or default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class EnrichmentConfig:
    api_key: str | None = None
    api_base: str = GEMINI_API_BASE
    text_model: str = DEFAULT_TEXT_MODEL
    media_model: str = DEFAULT_MEDIA_MODEL
    media_fallback_model: str | None = DEFAULT_MEDIA_FALLBACK
    text_fallback_model: str | None = "gemini-3.1-flash-lite"

    store_path: Path = Path("data/extractor_cache.sqlite3")
    request_timeout: float = 120.0

    # Media handling
    allow_media: bool = True
    high_resolution: bool = False
    # Hard ceiling on how much video is actually analyzed, sent as a clip window on the
    # request. Two reasons this exists:
    #
    # 1. Duration is frequently unknown at this point -- oEmbed does not return it, and
    #    it is oEmbed that carries YouTube now that yt-dlp is bot-walled. An unknown
    #    duration makes the cost predictor return $0 and the budget ceiling
    #    unenforceable, so an 18-minute video sails through as if it were free.
    # 2. For *tagging*, the opening couple of minutes plus the title almost always
    #    settles what something is. Paying to watch the remaining sixteen minutes buys
    #    very little recall.
    max_analyze_seconds: float = 180.0
    max_media_bytes: int = 100_000_000
    max_media_duration_s: float = 900.0
    upload_poll_timeout_s: float = 180.0

    # Re-encode video before upload. Models sample video at ~1 fps, so anything beyond that
    # is billed-for detail nobody looks at. Measured on a real reel: 17.7 MB and 16,760
    # tokens became 0.45 MB and 5,230 tokens, and the compressed clip still read the shop
    # signage correctly. Best-effort -- a missing or failing ffmpeg just sends the original.
    compress_video: bool = True
    ffmpeg_path: str | None = None
    video_fps: float = 1.0
    video_height: int = 480
    video_crf: int = 32

    # Budget guard: refuse a single call predicted to cost more than this. Stops one
    # pathological two-hour video from quietly costing more than everything else.
    max_cost_per_item_usd: float = 0.25

    # Text sent to the model. Long transcripts are trimmed rather than truncating
    # mid-word at an arbitrary byte count.
    max_text_chars: int = 12_000

    # --- provider selection --------------------------------------------------
    # Ordered preference list, e.g. "gemini,ashna" or "ashna,gemini". The first provider
    # that is configured *and* supports the required mode wins, so putting ashna first
    # routes text tagging through the gateway while video still falls to gemini, which is
    # the only one that can actually watch it.
    provider_order: tuple[str, ...] = ("gemini", "ashna")

    # OpenAI-compatible gateway (Ashna by default, but any compatible endpoint works).
    compat_name: str = "ashna"
    compat_base_url: str | None = "https://api.ashna.ai/v1/api"
    compat_api_key: str | None = None
    compat_model: str = "gemini-2.5-Flash"
    # Generous by necessity: at 400 the gateway returned HTTP 200 with empty content,
    # having spent the entire budget on reasoning tokens.
    compat_max_tokens: int = 3000

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> EnrichmentConfig:
        return cls(
            api_key=(
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or None
            ),
            api_base=os.getenv("GEMINI_API_BASE", GEMINI_API_BASE).rstrip("/"),
            text_model=os.getenv("ENRICH_TEXT_MODEL", DEFAULT_TEXT_MODEL),
            media_model=os.getenv("ENRICH_MEDIA_MODEL", DEFAULT_MEDIA_MODEL),
            media_fallback_model=(
                os.getenv("ENRICH_MEDIA_FALLBACK", DEFAULT_MEDIA_FALLBACK) or None
            ),
            text_fallback_model=(
                os.getenv("ENRICH_TEXT_FALLBACK", "gemini-3.1-flash-lite") or None
            ),
            store_path=Path(
                os.getenv("EXTRACTOR_CACHE_PATH", "data/extractor_cache.sqlite3")
            ),
            request_timeout=_float("ENRICH_TIMEOUT", 120.0),
            allow_media=_bool("ENRICH_ALLOW_MEDIA", True),
            high_resolution=_bool("ENRICH_HIGH_RESOLUTION", False),
            max_analyze_seconds=_float("ENRICH_MAX_ANALYZE_SECONDS", 180.0),
            compress_video=_bool("ENRICH_COMPRESS_VIDEO", True),
            ffmpeg_path=os.getenv("ENRICH_FFMPEG_PATH") or None,
            video_fps=_float("ENRICH_VIDEO_FPS", 1.0),
            video_height=_int("ENRICH_VIDEO_HEIGHT", 480),
            video_crf=_int("ENRICH_VIDEO_CRF", 32),
            max_media_bytes=_int("ENRICH_MAX_MEDIA_BYTES", 100_000_000),
            max_media_duration_s=_float("ENRICH_MAX_MEDIA_DURATION", 900.0),
            max_cost_per_item_usd=_float("ENRICH_MAX_COST_PER_ITEM", 0.25),
            max_text_chars=_int("ENRICH_MAX_TEXT_CHARS", 12_000),
            provider_order=_providers("ENRICH_PROVIDER", ("gemini", "ashna")),
            compat_name=os.getenv("COMPAT_NAME", "ashna"),
            compat_base_url=(
                os.getenv("ASHNA_BASE_URL")
                or os.getenv("COMPAT_BASE_URL")
                or "https://api.ashna.ai/v1/api"
            ),
            compat_api_key=(
                os.getenv("ASHNA_API_KEY") or os.getenv("COMPAT_API_KEY") or None
            ),
            compat_model=(
                os.getenv("ASHNA_TEXT_MODEL")
                or os.getenv("COMPAT_MODEL")
                or "gemini-2.5-Flash"
            ),
            compat_max_tokens=_int("COMPAT_MAX_TOKENS", 3000),
        )
