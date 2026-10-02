"""Per-model token pricing and cost estimation.

Tracked from day one on purpose. Without a cost-per-item column you cannot answer
"what does one user cost me" until you already have users, and by then the expensive
decisions are baked in.

Rates below are USD per 1M tokens, taken from Google's official Gemini API pricing
page (ai.google.dev/gemini-api/docs/pricing), which was last updated 2026-09-01 when
these were read. Two things to know:

* Several Gemini 3.x Flash rates are promotional through 2026-12-31 and double on
  2027-01-01. `PRICE_INCREASE_NOTE` records that so it does not surprise you.
* Output pricing includes thinking tokens.

Costs computed here are estimates for your own accounting, not billing figures.
Token counts come from the API's own `usageMetadata`, so they are accurate; the
arithmetic is only as current as this table.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICE_INCREASE_NOTE = (
    "gemini-3.6-flash and gemini-3.7-flash input/output rates double on 2027-01-01"
)


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per 1,000,000 tokens."""

    input_usd: float
    output_usd: float
    note: str = ""


# Standard (non-batch, non-priority) paid-tier rates.
MODEL_PRICES: dict[str, ModelPrice] = {
    # Gemini 3.x Flash family -- promotional pricing through 2026-12-31.
    "gemini-3.7-flash": ModelPrice(0.75, 3.75, "doubles 2027-01-01"),
    "gemini-3.6-flash": ModelPrice(0.75, 3.75, "doubles 2027-01-01"),
    "gemini-3.5-flash": ModelPrice(1.50, 9.00),
    "gemini-3.1-flash": ModelPrice(0.50, 3.00),
    "gemini-3.1-flash-lite": ModelPrice(0.25, 1.50),
    "gemini-3.1-pro": ModelPrice(2.00, 12.00, "input doubles above 200k tokens"),
    "gemini-3.5-flash-lite": ModelPrice(0.30, 2.50),
    # Gemini 2.5 family. Note that 2.5-flash-lite still appears in ListModels but
    # returns 404 "no longer available to new users" when called, so it is not a safe
    # default despite being the cheapest rate here.
    "gemini-2.5-flash": ModelPrice(0.30, 2.50),
    "gemini-2.5-flash-lite": ModelPrice(0.10, 0.40, "retired for new API keys"),
    "gemini-2.5-pro": ModelPrice(1.25, 10.00, "input doubles above 200k tokens"),
    # Embeddings, for the stage after this one.
    "gemini-embedding": ModelPrice(0.15, 0.0),
    "gemini-multimodal-embedding": ModelPrice(0.20, 0.0),
}

# Batch API halves both rates. Worth remembering for backfilling a library of
# already-saved links, where latency is irrelevant.
BATCH_DISCOUNT = 0.5


def is_priced(model: str) -> bool:
    """Whether this table can price the model at all.

    Worth checking explicitly. Google ships new model names faster than any hardcoded
    rate table keeps up, and an unpriced model silently reporting $0.00 per item is
    worse than no cost tracking -- it reads as "this is free" rather than "this is
    unknown".
    """
    return model in MODEL_PRICES or _family(model) in MODEL_PRICES


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    batch: bool = False,
) -> float:
    """USD cost for one call. Unknown models cost 0.0 rather than raising.

    Returning zero for an unrecognized model is deliberate: a model name this table
    has not caught up with must not be able to break the pipeline. Callers should pair
    this with `is_priced()` so the gap is surfaced rather than hidden.
    """
    price = MODEL_PRICES.get(model)
    if price is None:
        price = MODEL_PRICES.get(_family(model))
    if price is None:
        return 0.0

    multiplier = BATCH_DISCOUNT if batch else 1.0
    cost = (
        input_tokens * price.input_usd + output_tokens * price.output_usd
    ) / 1_000_000
    return round(cost * multiplier, 8)


def _family(model: str) -> str:
    """Strip suffixes like `-preview`, `-latest`, `-001` to find a base rate."""
    base = model.strip().lower()
    for suffix in ("-preview", "-latest", "-exp"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    parts = base.split("-")
    while parts and parts[-1].isdigit():
        parts.pop()
    return "-".join(parts)


# ------------------------------------------------------------------- video budgets
# From Gemini's media resolution docs: video frames cost 70 tokens each at low and
# medium resolution (identical for video) and 280 at high, sampled at ~1 fps, plus
# roughly 32 tokens per second of audio.
#
# Use `high` only for text-dense video -- slides, code, dense on-screen captions --
# where reading the frame is the whole point.
VIDEO_TOKENS_PER_FRAME_LOW = 70
VIDEO_TOKENS_PER_FRAME_HIGH = 280
AUDIO_TOKENS_PER_SECOND = 32
DEFAULT_FRAMES_PER_SECOND = 1.0


def estimate_video_tokens(
    duration_s: float,
    *,
    high_resolution: bool = False,
    fps: float = DEFAULT_FRAMES_PER_SECOND,
    with_audio: bool = True,
) -> int:
    """Predicted input tokens for a video, for budgeting before you spend anything."""
    if duration_s <= 0:
        return 0
    per_frame = (
        VIDEO_TOKENS_PER_FRAME_HIGH if high_resolution else VIDEO_TOKENS_PER_FRAME_LOW
    )
    frames = max(1, int(duration_s * fps))
    tokens = frames * per_frame
    if with_audio:
        tokens += int(duration_s * AUDIO_TOKENS_PER_SECOND)
    return tokens


def describe_video_budget(
    duration_s: float, model: str, *, high_resolution: bool = False
) -> dict[str, object]:
    """Cost preview for one video, used by the CLI before committing to a call."""
    input_tokens = estimate_video_tokens(duration_s, high_resolution=high_resolution)
    output_tokens = 500  # a filled-in enrichment schema lands near here
    return {
        "model": model,
        "duration_s": round(duration_s, 1),
        "resolution": "high" if high_resolution else "low",
        "est_input_tokens": input_tokens,
        "est_output_tokens": output_tokens,
        "est_cost_usd": estimate_cost(model, input_tokens, output_tokens),
    }
