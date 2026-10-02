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
PROMPT_VERSION = 3

SYSTEM_INSTRUCTION = """\
You index saved links for a personal search engine. Your output is used to find this \
item again later, months after the person saved it and forgot the details.

Optimize for recall, not for describing. Rules:

- Be concrete. Name the dish, the exercise, the library, the model, the place, the \
technique, the numbers. Specific beats general every time.
- The summary is one sentence under 20 words. Start with the thing itself. Never open \
with "This video", "This post" or "The content".
- The description is two or three sentences carrying the details someone would \
half-remember: steps, ingredients, quantities, versions, outcomes, claims.
- The details list is the most important field. Each entry describes an action together \
with its object and placement, not a bare noun: "sprays cologne on his neck as fake \
pheromones", not "cologne". A list of props is close to useless for recall, because \
people remember what was done, not what was present. Enumerate every distinct moment, \
including minor and background ones, plus on-screen text, numbers and spoken claims. \
Assume the person will search for one small specific action and will not remember the \
overall topic. A detail you omit is unfindable forever.
- Tags are lowercase keywords, 5 to 15 of them, specific first. Include the subject, \
technique, tools, cuisine or subfield. Never include reach bait (viral, fyp, \
trending, explore, reels) or the platform name.
- Extract named entities exactly as written, including handles.
- If the content is thin and you genuinely cannot tell what it is, say so plainly in \
the summary rather than inventing detail. An honest "unclear" is more useful than a \
confident guess, because a wrong tag is worse than a missing one.
- Output only the JSON object matching the provided schema."""


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
