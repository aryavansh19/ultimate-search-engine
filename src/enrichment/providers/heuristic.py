"""Offline enrichment: keyword extraction, no LLM, no API key, no cost.

This is not a toy. It does three real jobs:

1. The pipeline runs end to end today, before anyone has signed up for an API key,
   which means every stage after this one is testable now.
2. It is the fallback when the LLM call fails, is rate-limited, or would exceed the
   per-item cost ceiling. A saved link still becomes searchable, just less precisely,
   and the row is flagged `degraded` so a re-run can upgrade it later.
3. It is the honest baseline. Before paying per item for tagging, you should know how
   much better the paid version actually is on your own links -- and that comparison
   only exists if the free version is implemented.

The method is frequency scoring with position weighting and bigram detection. Crude
compared to a language model, and it will never infer that a video shows orecchiette
when nobody wrote the word down. That gap is exactly what you are buying when you
escalate.
"""

from __future__ import annotations

import re
import time
from collections import Counter

from extractor import ContentEnvelope

from ..schema import Enrichment, Entities, EnrichmentMode
from ..taxonomy import coerce_category, coerce_content_type, normalize_tags
from .base import EnrichmentProvider

_WORD = re.compile(r"[a-z][a-z0-9'\-]{1,30}")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PROPER_NOUN = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3})\b")
_HANDLE = re.compile(r"@([A-Za-z0-9._]{2,30})")

_STOPWORDS: frozenset[str] = frozenset(
    """
a about above after again against all also am an and any are aren't as at be because
been before being below between both but by can can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further get got had
hadn't has hasn't have haven't having he her here hers herself him himself his how
i i'm if in into is isn't it it's its itself just let's like make me more most must
my myself no nor not now of off on once only or other ought our ours ourselves out
over own really same shan't she should shouldn't so some such than that that's the
their theirs them themselves then there there's these they this those through to too
under until up very want was wasn't way we well were weren't what what's when where
which while who whom why will with won't would wouldn't you your yours yourself
yourselves thing things lot lots much many one two three new using use used know
going go come came see look looking made take taking give given put says said think
thought back good great best better right need needs first last next time day today
guys people everyone something anything nothing everything video watch follow link
bio comment share subscribe part full check out click swipe

retrieved archived original cite cited citation citations reference references
isbn issn doi arxiv jstor pdf html http https www com org net edu wikipedia
wikimedia commons license licensed edit editing template category categories
external links sources further reading see also notes bibliography accessed
january february march april may june july august september october november
december monday tuesday wednesday thursday friday saturday sunday
""".split()
)

# Subject-area cues. Rough by design: this is a fallback, and a wrong category is
# recoverable because the tags carry most of the search weight.
_CATEGORY_CUES: dict[str, tuple[str, ...]] = {
    "food-cooking": (
        "recipe", "cook", "cooking", "bake", "baking", "dough", "flour", "sauce",
        "pasta", "kitchen", "ingredient", "flavour", "flavor", "roast", "grill",
        "dish", "meal", "eat", "food", "chef", "pizza", "bread", "dessert",
    ),
    "fitness-health": (
        "workout", "exercise", "reps", "sets", "muscle", "training", "gym", "squat",
        "deadlift", "cardio", "protein", "stretch", "mobility", "fitness", "diet",
        "calories", "sleep", "recovery", "injury",
    ),
    "software-engineering": (
        "code", "coding", "function", "api", "database", "query", "server", "deploy",
        "bug", "refactor", "repository", "commit", "typescript", "python", "rust",
        "postgres", "postgresql", "docker", "kubernetes", "index", "latency", "cache",
    ),
    "ai-ml": (
        "model", "embedding", "embeddings", "vector", "neural", "network", "training",
        "dataset", "inference", "transformer", "llm", "prompt", "token", "gradient",
        "classifier", "fine-tune", "agent",
    ),
    "travel": (
        "travel", "trip", "flight", "hotel", "itinerary", "visa", "airport", "city",
        "beach", "hike", "tour", "destination", "backpacking", "village", "island",
    ),
    "technology": (
        "phone", "laptop", "chip", "processor", "gadget", "device", "battery",
        "screen", "hardware", "software", "app", "update", "release", "spec",
    ),
    "business-finance": (
        "revenue", "profit", "startup", "funding", "investor", "market", "stock",
        "tax", "salary", "pricing", "customer", "growth", "budget", "invoice",
    ),
    "design": (
        "design", "typography", "layout", "colour", "color", "font", "spacing",
        "figma", "interface", "wireframe", "brand", "logo", "palette",
    ),
    "music": ("song", "album", "guitar", "piano", "chord", "beat", "melody", "band"),
    "sports": ("match", "goal", "team", "player", "league", "score", "tournament"),
    "gaming": ("game", "gaming", "level", "player", "console", "boss", "speedrun"),
    "science": (
        "research", "study", "experiment", "hypothesis", "physics", "chemistry",
        "biology", "cell", "quantum", "climate", "species",
    ),
    "home-diy": (
        "diy", "build", "wood", "paint", "drill", "repair", "furniture", "garden",
        "install", "tool", "shelf",
    ),
    "fashion-beauty": (
        "outfit", "style", "wear", "skincare", "makeup", "hair", "fabric", "dress",
    ),
    "education-learning": (
        "learn", "learning", "course", "lesson", "study", "exam", "student",
        "teacher", "school", "university", "explain",
    ),
    "productivity": (
        "productivity", "workflow", "habit", "routine", "notes", "planning", "focus",
        "calendar", "task",
    ),
}

_CONTENT_TYPE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("recipe", ("recipe", "ingredient", "ingredients", "tbsp", "tsp", "grams", "bake")),
    ("workout", ("reps", "sets", "workout", "circuit", "warm-up", "superset")),
    ("tutorial", ("tutorial", "how", "step", "steps", "guide", "walkthrough", "setup")),
    ("review", ("review", "verdict", "pros", "cons", "worth", "tested", "rating")),
    ("listicle", ("top", "best", "reasons", "ways", "tips", "things")),
    ("explainer", ("explain", "explained", "why", "understand", "actually", "means")),
    ("interview", ("interview", "asked", "conversation", "talks", "discusses")),
    ("news-report", ("announced", "launch", "released", "report", "breaking")),
    ("product-demo", ("demo", "showcase", "introducing", "features", "hands-on")),
    ("case-study", ("case", "study", "results", "before", "after", "improved")),
    ("travel-guide", ("itinerary", "where", "stay", "visit", "days", "budget")),
    ("discussion", ("thoughts", "opinions", "anyone", "thread", "discussion")),
    ("meme", ("meme", "pov", "when", "nobody", "relatable")),
)


class HeuristicProvider(EnrichmentProvider):
    name = "heuristic"

    def supports(self, mode: EnrichmentMode) -> bool:
        # Cannot see or hear anything; text only.
        return mode is not EnrichmentMode.MEDIA

    def enrich(
        self, envelope: ContentEnvelope, mode: EnrichmentMode, document: str
    ) -> Enrichment:
        started = time.perf_counter()

        raw_text = document or envelope.search_document()
        lowered = _dedupe_sentences(raw_text).lower()
        tokens = [t for t in _WORD.findall(lowered) if t not in _STOPWORDS and len(t) > 2]

        scores = self._score_terms(envelope, tokens)
        tags = normalize_tags(
            list(envelope.platform_tags) + [term for term, _ in scores.most_common(25)],
            limit=15,
        )

        summary = self._summary(envelope)
        description = self._description(envelope, raw_text)

        return Enrichment(
            url_hash=envelope.url_hash,
            summary=summary,
            description=description,
            tags=tags,
            category=coerce_category(self._category(tokens)),
            content_type=coerce_content_type(self._content_type(lowered, envelope)),
            entities=self._entities(envelope, raw_text),
            language="en" if _mostly_ascii(raw_text) else None,
            transcript_excerpt=(
                envelope.transcript[:400] if envelope.transcript else None
            ),
            mode=EnrichmentMode.HEURISTIC,
            provider=self.name,
            model=None,
            source_hash="",
            duration_ms=int((time.perf_counter() - started) * 1000),
            degraded=True,
            note="keyword extraction; no language model was used",
        )

    # ------------------------------------------------------------------ internals
    def _score_terms(
        self, envelope: ContentEnvelope, tokens: list[str]
    ) -> Counter[str]:
        """Frequency, weighted by where a term appears and boosted for bigrams."""
        scores: Counter[str] = Counter()
        for token in tokens:
            scores[token] += 1

        # Terms in the title or the platform's own hashtags are worth far more than
        # terms buried in a transcript.
        for field, weight in (
            (envelope.title, 4),
            (envelope.caption, 2),
            (" ".join(envelope.platform_tags), 5),
        ):
            if not field:
                continue
            for token in _WORD.findall(field.lower()):
                if token in _STOPWORDS or len(token) <= 2:
                    continue
                scores[token] += weight

        # Repeated adjacent pairs usually name the actual subject ("vector database",
        # "partial index"), which single tokens lose.
        #
        # Built from prose only. Running this over the assembled document pairs up
        # neighbouring hashtags, which are adjacent by layout rather than by meaning --
        # `#pasta #orecchiette #puglia` yielded the phantom terms `pasta-orecchiette`
        # and `orecchiette-puglia`.
        prose = " ".join(
            part
            for part in (
                envelope.caption,
                envelope.article_text,
                envelope.transcript,
                envelope.visual_description,
            )
            if part
        ).lower()
        prose_tokens = [
            t for t in _WORD.findall(prose) if t not in _STOPWORDS and len(t) > 2
        ]
        bigrams = Counter(
            f"{a}-{b}" for a, b in zip(prose_tokens, prose_tokens[1:]) if a != b
        )
        for bigram, count in bigrams.items():
            if count >= 2:
                scores[bigram] += count * 3

        for term in list(scores):
            if term in _STOPWORDS:
                del scores[term]
        return scores

    def _category(self, tokens: list[str]) -> str:
        counts = Counter(tokens)
        best, best_score = "other", 0
        for category, cues in _CATEGORY_CUES.items():
            score = sum(counts.get(cue, 0) for cue in cues)
            if score > best_score:
                best, best_score = category, score
        return best if best_score >= 2 else "other"

    def _content_type(self, lowered: str, envelope: ContentEnvelope) -> str:
        for content_type, cues in _CONTENT_TYPE_CUES:
            if sum(1 for cue in cues if cue in lowered) >= 2:
                return content_type
        if envelope.article_text:
            return "article"
        return "other"

    def _summary(self, envelope: ContentEnvelope) -> str:
        for candidate in (envelope.title, envelope.caption, envelope.article_text):
            if not candidate:
                continue
            sentence = _SENTENCE_SPLIT.split(candidate.strip())[0].strip()
            if len(sentence) > 12:
                return _clip(sentence, 140)
        if envelope.author:
            return f"Saved {envelope.media_kind.value} from {envelope.author}"
        return "No description available"

    def _description(self, envelope: ContentEnvelope, raw_text: str) -> str:
        body = envelope.caption or envelope.article_text or envelope.transcript or ""
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(body) if len(s.strip()) > 20]
        if not sentences:
            return _clip(" ".join(raw_text.split()), 300)
        return _clip(" ".join(sentences[:3]), 400)

    def _entities(self, envelope: ContentEnvelope, raw_text: str) -> Entities:
        """Capitalized runs and @handles. Catches proper nouns, also catches
        sentence-initial words -- acceptable for a fallback, and the LLM path does
        this properly."""
        handles = [f"@{h}" for h in _HANDLE.findall(raw_text)[:5]]
        people = handles
        if envelope.author:
            people = [envelope.author, *[h for h in handles if h != envelope.author]][:6]

        candidates: list[str] = []
        for match in _PROPER_NOUN.findall(raw_text):
            phrase = match.strip()
            if len(phrase) < 3 or phrase.lower() in _STOPWORDS:
                continue
            if phrase not in candidates:
                candidates.append(phrase)
            if len(candidates) >= 8:
                break

        return Entities(
            people=people[:6],
            organizations=[c for c in candidates if " " not in c][:6],
            places=[],
            products=[c for c in candidates if " " in c][:6],
        )


def _dedupe_sentences(text: str) -> str:
    """Drop repeated sentences before counting term frequency.

    Frequency scoring has no way to tell a topic from a repeated incidental example. A
    blog post about AI predictions quoted one sample research query -- about brown
    pelican roosts -- and the tool's echoed plan repeated it seventeen times, which was
    enough to make `brown-pelican` the top-scoring tag on an article that is not about
    pelicans at all.

    Collapsing exact repeats fixes that case and helps generally: auto-generated
    captions repeat rolling text, quoted plans and outlines restate themselves, and
    boilerplate recurs. It does not make this a good tagger -- judging what a document
    is *about* rather than what words it contains is what the language model is for.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        normalized = " ".join(sentence.split()).lower()
        if len(normalized) < 25:
            kept.append(sentence)
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        kept.append(sentence)
    return " ".join(kept)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rfind(" ")
    return text[: cut if cut > limit * 0.6 else limit].rstrip() + "..."


def _mostly_ascii(text: str) -> bool:
    if not text:
        return True
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return ascii_chars / len(text) > 0.9
