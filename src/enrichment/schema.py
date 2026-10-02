"""The enrichment contract and the JSON schema handed to the model.

`Enrichment` is what the tagging stage produces for every item, whatever produced it
-- a multimodal LLM call, a text-only call, or the offline heuristic fallback. The
embedding stage consumes this plus the `ContentEnvelope` and needs neither to know
which path was taken.

The response schema is hand-built rather than generated from the pydantic model.
Pydantic emits JSON Schema with `$defs`, `anyOf` and `$ref`, and Gemini's
`responseSchema` accepts only a restricted OpenAPI subset. Writing it by hand is
duller but avoids a class of failure where the model silently ignores constraints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .taxonomy import CATEGORIES, CONTENT_TYPES


class EnrichmentMode(str, Enum):
    """Which path produced this enrichment. Drives cost accounting and re-runs."""

    METADATA = "metadata"          # text only: caption, title, transcript, article
    MEDIA = "media"                # full video/image sent to a multimodal model
    HEURISTIC = "heuristic"        # no LLM at all; keyword extraction fallback


class Entities(BaseModel):
    """Proper nouns worth indexing separately from topical tags.

    Kept apart from `tags` because these are the tokens keyword search has to match
    exactly -- a person's handle, a product name, a place. Embeddings smear exactly
    this kind of token, which is the reason the search layer stays hybrid.
    """

    model_config = ConfigDict(extra="ignore")

    people: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)

    def flat(self) -> list[str]:
        out: dict[str, None] = {}
        for group in (self.people, self.organizations, self.places, self.products):
            for value in group:
                cleaned = " ".join(str(value).split())
                if cleaned:
                    out.setdefault(cleaned, None)
        return list(out)


class Enrichment(BaseModel):
    """Derived, searchable understanding of one saved link."""

    model_config = ConfigDict(extra="ignore")

    url_hash: str
    summary: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    category: str = "other"
    content_type: str = "other"
    entities: Entities = Field(default_factory=Entities)
    language: str | None = None
    transcript_excerpt: str | None = None
    # Itemized, concrete moments and objects. This is what makes an item findable by a
    # half-remembered detail rather than only by its overall subject.
    #
    # A tidy three-sentence description is the wrong shape for recall. Tested against a
    # comedy reel that is a rapid stream of visual gags, the summary captured five of
    # them and the model simply did not write down the rest -- so queries for the
    # pheromone spray or the Apple Watch worn on a leg had nothing to match, even though
    # the model had plainly seen both. Retrieval cannot recover what enrichment declined
    # to record.
    details: list[str] = Field(default_factory=list)

    # --- provenance and cost -------------------------------------------------
    mode: EnrichmentMode = EnrichmentMode.HEURISTIC
    provider: str = "heuristic"
    model: str | None = None
    prompt_version: int = 1
    source_hash: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    degraded: bool = False
    note: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def search_text(self) -> str:
        """The enrichment's contribution to the document that gets embedded.

        Ordered most-distinctive first so that if a chunker ever truncates, the
        summary and tags survive.
        """
        parts = [
            ("summary", self.summary),
            ("description", self.description),
            ("details", "; ".join(self.details)),
            ("tags", " ".join(self.tags)),
            ("entities", " ".join(self.entities.flat())),
            ("category", self.category),
            ("type", self.content_type),
        ]
        return "\n".join(f"{label}: {value}" for label, value in parts if value)


# --------------------------------------------------------------------------------
# Gemini structured-output schema. OpenAPI subset only: type, properties, items,
# enum, description, required, propertyOrdering. No $ref, no anyOf, no oneOf.
# --------------------------------------------------------------------------------

def _string_array(description: str, max_items: int) -> dict[str, Any]:
    return {
        "type": "array",
        "description": description,
        "items": {"type": "string"},
        "maxItems": max_items,
    }


def openai_response_format(name: str = "enrichment") -> dict[str, Any]:
    """The same contract expressed for an OpenAI-compatible `response_format`.

    Two dialects, one schema. Gemini's `responseSchema` accepts `propertyOrdering`, which
    OpenAI-compatible endpoints do not, and OpenAI expects `additionalProperties: false` on
    every object. Converting here keeps a single source of truth -- maintaining two
    hand-written copies of this schema would guarantee they drift.

    `strict` is left off deliberately. Strict mode requires every property to appear in
    `required`, which would force the model to emit empty arrays for entity groups it found
    nothing for, and the downstream mapping already tolerates missing keys.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": False,
            "schema": _to_openai_schema(RESPONSE_SCHEMA),
        },
    }


def _to_openai_schema(schema: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "propertyOrdering":  # Gemini-only keyword
            continue
        if key == "properties":
            out[key] = {k: _to_openai_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            out[key] = _to_openai_schema(value)
        else:
            out[key] = value
    if out.get("type") == "object":
        out["additionalProperties"] = False
    return out


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "One sentence, under 20 words, stating what this content actually is. "
                "No preamble, no 'this video shows'."
            ),
        },
        "description": {
            "type": "string",
            "description": (
                "Two or three sentences covering the specifics someone would use to "
                "find this again later: concrete steps, named things, numbers, "
                "outcomes. Avoid generic praise."
            ),
        },
        "details": _string_array(
            "Exhaustive list of individually memorable moments. Each entry must describe "
            "an ACTION with its object, not just name the object: write 'pours baking "
            "soda into his penny loafers' rather than 'baking soda'; 'straps an Apple "
            "Watch to his calf' rather than 'Apple Watch'. Include what is done, where "
            "it is placed, and any on-screen text, numbers, brands or spoken claims. "
            "Be complete rather than tidy -- someone will search for exactly the small "
            "thing you leave out, and they will remember the action, not the prop.",
            30,
        ),
        "tags": _string_array(
            "5 to 15 lowercase topical keywords. Prefer specific over generic "
            "('orecchiette' over 'food'). Exclude reach-bait like 'viral' or 'fyp'.",
            15,
        ),
        "category": {
            "type": "string",
            "description": "Single best subject area.",
            "enum": list(CATEGORIES),
        },
        "content_type": {
            "type": "string",
            "description": "What kind of content this is, regardless of subject.",
            "enum": list(CONTENT_TYPES),
        },
        "people": _string_array("Named individuals or handles appearing.", 8),
        "organizations": _string_array("Named brands, companies or publications.", 8),
        "places": _string_array("Named locations.", 8),
        "products": _string_array("Named products, tools, models or libraries.", 10),
        "language": {
            "type": "string",
            "description": "BCP-47 primary language of the content, e.g. 'en', 'hi'.",
        },
    },
    "required": [
        "summary",
        "description",
        "details",
        "tags",
        "category",
        "content_type",
    ],
    "propertyOrdering": [
        "summary",
        "description",
        "details",
        "tags",
        "category",
        "content_type",
        "people",
        "organizations",
        "places",
        "products",
        "language",
    ],
}
