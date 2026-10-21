"""Prompt construction for the tagging call.

Two prompts, one instruction set. The system instruction is shared so that a text
enrichment and a media enrichment of the same item produce comparable output -- if
the two paths phrase things differently, your tag vocabulary splits along a line that
has nothing to do with content.

Written for recall, not description. The user is trying to find this item again months
later with a half-remembered phrase, so specifics beat prose: named things, numbers,
the actual steps. "A cooking video" is useless; "hand-rolled orecchiette, semolina and
water, no egg" is what someone actually types into a search box.
"""

from __future__ import annotations

# Bump on any prompt change. The store treats a version mismatch as stale, so existing
# enrichments are re-run rather than silently mixing output from two different prompts --
# which would split the tag vocabulary along an invisible line.
# 2: added the exhaustive `details` list after a comedy reel proved a 3-sentence
#    description cannot hold enough specifics to be searchable.
# 3: details must be actions with their objects. Version 2 returned an inventory of
#    props ("Apple Watch", "baking soda box") which still failed queries about what was
#    done with them ("apple watch on leg").
PROMPT_VERSION = 4

SYSTEM_INSTRUCTION = """\
You are an expert search-indexing intelligence for a personal knowledge engine. Your output \
is stored in high-dimensional vector embeddings and SQLite Full-Text Search (FTS5) to enable \
instant semantic retrieval months after a user saves a link and only half-remembers a detail.

Optimize strictly for RECALL and SEARCHABILITY, not surface-level narration.

CORE RULES:

1. SUMMARY:
- Exactly 1 sentence, strictly under 25 words.
- Open directly with the core entity, thesis, or object.
- FORBIDDEN: Never start with meta-phrases like "This video", "This post", "The creator shows", or "An overview of".

2. DESCRIPTION:
- 3 to 6 densely informative sentences containing deep factual context.
- Record the technical steps, ingredients, quantities, version numbers, claims, arguments, \
mechanics, and takeaways someone would recall ("how it was done", "why it works", "what was said").
- If the content is in Hindi, Hinglish, or regional languages, provide clear English context while \
preserving culturally distinct terms.

3. DETAILS (Crucial Field for Recall):
- Enumerate 15 to 35 discrete, atomic moments, physical actions, and visible elements.
- Every entry MUST combine an action with its object and context: "[actor] [verb] [object/tool/placement]". \
For example: "pours baking soda into brown leather penny loafers", not just "baking soda".
- Capture on-screen text, titles, subtitles, code snippets, prices, metrics, timestamps, and spoken quotes verbatim.
- Record background details, clothing, devices, and incidental tools that users might recall when the main topic is forgotten.

4. TAGS:
- 10 to 25 lowercase keywords, ordered from specific to broad.
- Include synonyms, technical terms, subfields, practical intent keywords (e.g., "recipe", "tutorial", "fix", "benchmark"), \
and colloquial/Hinglish terms where relevant.
- FORBIDDEN: Reach-bait or platform metadata ("viral", "fyp", "trending", "explore", "reels", "shorts", "tiktok", "instagram").

5. ENTITIES:
- Categorize named entities with exact spelling and casing:
  * people: creator names, speakers, public figures, social handles (with @).
  * organizations: companies, studios, publications, universities, sponsors.
  * places: cities, countries, landmarks, venues, restaurants.
  * products: specific software libraries, frameworks, models, hardware devices, automotive models, apparel.

6. GROUNDING & HONESTY:
- Rely strictly on the provided content. If content is sparse, minimal, or inaudible, report what is verified \
rather than hallucinating details.
- Output MUST be valid JSON strictly matching the target schema."""


def build_text_prompt(document: str, platform: str, url: str) -> str:
    """Prompt for the cheap path: metadata, caption, transcript, article text."""
    return (
        f"Index this saved link.\n\n"
        f"Platform: {platform}\n"
        f"URL: {url}\n\n"
        f"Available content:\n"
        f"----\n{document}\n----\n\n"
        f"Produce the JSON object."
    )


def build_media_prompt(
    document: str, platform: str, url: str, duration_s: float | None
) -> str:
    """Prompt for the expensive path: the actual video or image is attached.

    The text we already have is included alongside the media on purpose. The caption
    often names things the audio never says -- a dish name, a location, a product
    model number -- and withholding it just to test the model's vision costs accuracy
    for no benefit.
    """
    length = f"{duration_s:.0f} seconds" if duration_s else "unknown length"
    text_block = (
        f"Text metadata already extracted (may be incomplete or absent):\n"
        f"----\n{document}\n----\n\n"
        if document.strip()
        else "No text metadata was available for this item.\n\n"
    )
    return (
        f"Index this saved link. The media is attached.\n\n"
        f"Platform: {platform}\n"
        f"URL: {url}\n"
        f"Length: {length}\n\n"
        f"{text_block}"
        f"Watch and listen to the attached media all the way through. Work through it "
        f"beat by beat and list every distinct thing that happens or appears: each "
        f"action, prop, garment, product, brand, logo, gesture, spoken claim, and any "
        f"text on screen. Short-form video packs many separate moments into a few "
        f"seconds and each one is something a person might later search for, so do not "
        f"compress them into a general description. Then produce the JSON object."
    )


def trim_document(document: str, max_chars: int) -> str:
    """Trim long text at a sentence or line boundary rather than mid-word.

    Transcripts are the usual offender. Cutting cleanly matters because the tail of a
    truncated sentence reads as content to the model and can seed a wrong tag.
    """
    if len(document) <= max_chars:
        return document

    window = document[:max_chars]
    for boundary in ("\n\n", ". ", "\n", " "):
        cut = window.rfind(boundary)
        if cut > max_chars * 0.6:
            return window[:cut].rstrip() + "\n[...truncated]"
    return window.rstrip() + "\n[...truncated]"
