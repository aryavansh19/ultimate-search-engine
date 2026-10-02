"""Document assembly and chunking for the search index.

Two ideas here that matter more than the splitting mechanics.

First, the *digest chunk*. Every item gets a small leading chunk built from its title,
summary, tags and entities. Real queries are short and vague -- "that pasta video" --
and a short query embedded against a 2000-character chunk of transcript matches weakly
even when the item is exactly right, because the chunk's meaning is diluted across
everything else in it. A dense digest gives short queries something equally dense to
match.

Second, most saved links never need splitting at all. A reel with a caption is a few
hundred characters. Chunking only earns its keep on long articles and transcripts, so
the splitter stays out of the way below the threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from enrichment import Enrichment
from extractor import ContentEnvelope

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class Chunk:
    ordinal: int
    text: str
    kind: str  # "digest" or "body"

    @property
    def token_estimate(self) -> int:
        # Rough but adequate: English averages near four characters per token, and this
        # is only used for cost estimation and batch sizing.
        return max(1, len(self.text) // 4)


def build_keyword_fields(
    envelope: ContentEnvelope, enrichment: Enrichment | None
) -> dict[str, str]:
    """The per-column text that BM25 scores, kept separate so columns can be weighted.

    Keyword search is the leg that has to nail exact tokens -- a handle, a model number,
    a library name. Splitting these into columns is what lets a title match outrank an
    incidental mention buried in a transcript.
    """
    tags = " ".join(enrichment.tags) if enrichment else " ".join(envelope.platform_tags)
    entities = " ".join(enrichment.entities.flat()) if enrichment else ""
    # Details lead the body column so their tokens are keyword-searchable too. They are
    # folded in here rather than given a dedicated FTS column because adding a column to
    # an existing FTS5 table requires dropping and repopulating it, and repopulating
    # currently means re-embedding -- a schema tidiness worth less than the API spend.
    details = "; ".join(enrichment.details) if enrichment and enrichment.details else ""
    body_parts = [
        details,
        envelope.caption or "",
        envelope.article_text or "",
        envelope.transcript or "",
        envelope.ocr_text or "",
        envelope.visual_description or "",
    ]
    return {
        "title": envelope.title or "",
        "summary": (enrichment.summary if enrichment else "") or "",
        "description": (enrichment.description if enrichment else "") or "",
        "tags": tags,
        "entities": " ".join(part for part in [entities, envelope.author or ""] if part),
        "body": " ".join(part for part in body_parts if part),
    }


def build_digest(envelope: ContentEnvelope, enrichment: Enrichment | None) -> str:
    """Short, dense description of the item, used as the first chunk."""
    lines: list[str] = []
    if envelope.title:
        lines.append(f"title: {envelope.title}")
    if enrichment and enrichment.summary:
        lines.append(f"summary: {enrichment.summary}")
    if enrichment and enrichment.description:
        lines.append(f"about: {enrichment.description}")
    tags = enrichment.tags if enrichment else envelope.platform_tags
    if tags:
        lines.append(f"tags: {', '.join(tags)}")
    if enrichment:
        entities = enrichment.entities.flat()
        if entities:
            lines.append(f"mentions: {', '.join(entities)}")
        lines.append(f"category: {enrichment.category}; type: {enrichment.content_type}")
    if envelope.author:
        lines.append(f"by: {envelope.author}")
    lines.append(f"source: {envelope.platform.value}")
    if not any(line.startswith(("summary:", "about:")) for line in lines):
        # Nothing enriched yet; fall back to the caption so the digest is not just
        # metadata labels.
        if envelope.caption:
            lines.append(f"caption: {envelope.caption[:500]}")
    return "\n".join(lines)


def build_body(envelope: ContentEnvelope) -> str:
    """Long-form text worth searching inside, in descending order of value."""
    sections = [
        ("caption", envelope.caption),
        ("visual", envelope.visual_description),
        ("on-screen text", envelope.ocr_text),
        ("transcript", envelope.transcript),
        ("article", envelope.article_text),
    ]
    return "\n\n".join(
        f"{label}: {value.strip()}" for label, value in sections if value and value.strip()
    )


def chunk_document(
    envelope: ContentEnvelope,
    enrichment: Enrichment | None,
    *,
    chunk_chars: int = 2000,
    overlap_chars: int = 300,
) -> list[Chunk]:
    """Digest chunk, one chunk per enriched detail, then body chunks.

    Each detail gets its own chunk, and that is the single most important decision in
    this module. Packing 25 details into the digest was measurably worse than leaving
    them out: the digest grew to 2,438 characters and a two-word query like "apple watch
    on leg" scored *lower* against it than against a 114-character caption that had
    nothing to do with the query. Similarity is direction, and a long chunk's direction
    is the average of everything in it, so any one detail inside it is diluted away.

    One short chunk per detail keeps each one's meaning intact, which is exactly what a
    short, specific query needs to match against. The extra vectors are cheap -- a
    detail is roughly fifteen tokens.
    """
    chunks: list[Chunk] = [
        Chunk(ordinal=0, text=build_digest(envelope, enrichment), kind="digest")
    ]
    ordinal = 1

    if enrichment and enrichment.details:
        for detail in enrichment.details:
            text = detail.strip()
            if len(text) < 3:
                continue
            chunks.append(Chunk(ordinal=ordinal, text=text, kind="detail"))
            ordinal += 1

    body = build_body(envelope)
    if not body:
        return chunks

    if len(body) <= chunk_chars:
        chunks.append(Chunk(ordinal=ordinal, text=body, kind="body"))
        return chunks

    for piece in _split(body, chunk_chars, overlap_chars):
        chunks.append(Chunk(ordinal=ordinal, text=piece, kind="body"))
        ordinal += 1
    return chunks


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph boundaries, then sentences, never mid-word.

    Overlap exists so a passage that straddles a boundary is still fully present in at
    least one chunk. Without it, the sentence that answers a query can end up cut in
    half across two chunks and match neither well.
    """
    blocks = [block.strip() for block in _PARAGRAPH.split(text) if block.strip()]
    pieces: list[str] = []
    current = ""

    for block in blocks:
        for unit in _units(block, size):
            if not current:
                current = unit
            elif len(current) + len(unit) + 2 <= size:
                current = f"{current}\n\n{unit}"
            else:
                pieces.append(current)
                current = _tail(current, overlap) + unit if overlap else unit
    if current:
        pieces.append(current)
    return pieces


def _units(block: str, size: int) -> list[str]:
    """Break a block that is itself larger than one chunk into sentence-sized units."""
    if len(block) <= size:
        return [block]
    units: list[str] = []
    current = ""
    for sentence in _SENTENCE.split(block):
        if len(sentence) > size:
            # A single sentence longer than a chunk (transcripts without punctuation do
            # this). Fall back to hard word-boundary slicing.
            if current:
                units.append(current)
                current = ""
            units.extend(_hard_wrap(sentence, size))
            continue
        if not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= size:
            current = f"{current} {sentence}"
        else:
            units.append(current)
            current = sentence
    if current:
        units.append(current)
    return units


def _hard_wrap(text: str, size: int) -> list[str]:
    out: list[str] = []
    words = text.split()
    current = ""
    for word in words:
        if current and len(current) + len(word) + 1 > size:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out


def _tail(text: str, overlap: int) -> str:
    """Trailing slice of the previous chunk, cut at a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    if space > 0:
        tail = tail[space + 1:]
    return tail.strip() + "\n\n"
