"""Controlled vocabulary and tag normalization.

The design decision here is the mixed approach: a *fixed* category and content type
the model must choose from, plus *free-form* tags it may invent. Pure free-form
tagging drifts badly -- you end up with `cooking`, `Cooking`, `#cooking`, `cook` and
`cookery` as five distinct tags describing one thing, and your tag facet becomes
useless within a few hundred items.

So tags are normalized aggressively on write:

* lowercased, punctuation stripped, whitespace collapsed to hyphens
* naive singularization
* aliases folded (`js` -> `javascript`)
* platform noise dropped

That last one matters more than it sounds. Real Instagram and TikTok captions are
padded with `#fyp`, `#viral`, `#explorepage`, `#trending` -- reach bait that describes
nothing. Left in, it becomes the most common tag in the entire system and pollutes
every search facet.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------- categories
# Broad subject areas. One per item. Kept deliberately short: a long list makes the
# model equivocate and makes the facet useless for filtering.
CATEGORIES: tuple[str, ...] = (
    "food-cooking",
    "fitness-health",
    "travel",
    "technology",
    "software-engineering",
    "ai-ml",
    "design",
    "business-finance",
    "education-learning",
    "science",
    "news-politics",
    "entertainment",
    "music",
    "art-culture",
    "sports",
    "fashion-beauty",
    "home-diy",
    "nature-outdoors",
    "gaming",
    "lifestyle",
    "productivity",
    "shopping-products",
    "humor-memes",
    "other",
)

# ------------------------------------------------------------------ content types
# What *kind* of thing it is, independent of subject. This is the axis people
# actually remember by: "that recipe", "that tutorial", "that review".
CONTENT_TYPES: tuple[str, ...] = (
    "recipe",
    "workout",
    "tutorial",
    "explainer",
    "review",
    "listicle",
    "news-report",
    "interview",
    "vlog",
    "documentary",
    "comedy",
    "performance",
    "product-demo",
    "announcement",
    "opinion",
    "case-study",
    "tool-showcase",
    "travel-guide",
    "article",
    "discussion",
    "meme",
    "reference",
    "other",
)

# ------------------------------------------------------------------------- aliases
# Fold synonyms and abbreviations onto one canonical tag. Grow this as you see real
# drift in your own data rather than trying to guess it all up front.
TAG_ALIASES: dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "ml": "machine-learning",
    "dl": "deep-learning",
    "llm": "llms",
    "large-language-model": "llms",
    "large-language-models": "llms",
    "ai": "artificial-intelligence",
    "genai": "generative-ai",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "psql": "postgresql",
    "pg": "postgresql",
    "react-js": "react",
    "reactjs": "react",
    "nodejs": "node",
    "node-js": "node",
    "nextjs": "next-js",
    "swiftui": "swift-ui",
    "ios-development": "ios",
    "workout-routine": "workout",
    "gym": "fitness",
    "recipes": "recipe",
    "cooking-tips": "cooking",
    "italian-food": "italian-cuisine",
    "vector-db": "vector-database",
    "vector-databases": "vector-database",
    "rag": "retrieval-augmented-generation",
    "embedding": "embeddings",
}

# --------------------------------------------------------------------------- noise
# Reach bait and platform furniture. Describes nothing, appears everywhere.
NOISE_TAGS: frozenset[str] = frozenset(
    {
        "fyp", "fy", "foryou", "foryoupage", "for-you", "for-you-page",
        "viral", "viralvideo", "viral-video", "trending", "trend", "trendingreels",
        "explore", "explorepage", "explore-page", "instagram", "insta", "instagood",
        "instadaily", "reels", "reel", "reelsinstagram", "reelitfeelit",
        "tiktok", "tiktokviral", "youtube", "youtubeshorts", "shorts", "short",
        "video", "videos", "twitter", "x", "reddit", "linkedin", "facebook",
        "follow", "followme", "follow-for-more", "like", "likes", "comment",
        "share", "subscribe", "linkinbio", "link-in-bio", "dm", "collab",
        "love", "photooftheday", "picoftheday", "bestoftheday", "amazing",
        "beautiful", "cool", "nice", "wow", "omg", "lol", "fun", "funny",
        "new", "news", "today", "daily", "content", "post", "page", "account",
        "algorithm", "boost", "growth", "engagement", "views", "watch",
        "part1", "part2", "part3", "pt1", "pt2", "duet", "stitch", "sound",
        "capcut", "edit", "edits", "editing", "aesthetic", "vibes", "mood",
        "thisorthat", "asmr", "satisfying", "relatable", "storytime",
    }
)

_PUNCT = re.compile(r"[^a-z0-9\s\-+#]+")
_SPACES = re.compile(r"[\s_]+")
_DASHES = re.compile(r"-{2,}")

# Tags this short are almost always noise, with a few real exceptions worth keeping.
_SHORT_ALLOWLIST = frozenset({"ai", "ml", "ui", "ux", "3d", "vr", "ar", "go", "c", "r"})


def slugify_tag(raw: str) -> str | None:
    """Normalize one tag to its canonical slug, or None if it should be dropped."""
    if not raw:
        return None

    text = raw.strip().lower().lstrip("#")
    text = _PUNCT.sub(" ", text)
    text = _SPACES.sub("-", text).strip("-")
    text = _DASHES.sub("-", text)
    if not text or len(text) > 40:
        return None

    # Aliases are consulted *before* the length filter and *before* singularization,
    # because both would otherwise destroy the very inputs the alias table exists to
    # catch: `js` is dropped by the two-character rule, and `postgres` is stemmed to
    # `postgre` and never matches its alias again.
    aliased = TAG_ALIASES.get(text)
    if aliased is not None:
        text = aliased
    else:
        if len(text) < 2:
            return None
        if len(text) <= 2 and text not in _SHORT_ALLOWLIST:
            return None
        text = _singularize(text)
        # Singularization can expose an alias that the plural form hid.
        text = TAG_ALIASES.get(text, text)

    text = TAG_ALIASES.get(text, text)  # one hop for alias -> alias chains

    if text in NOISE_TAGS or text.isdigit():
        return None
    return text


# Words that end in `s` but are already singular. Without this guard, `physics`
# becomes `physic` and `analysis` becomes `analysi`.
_NOT_PLURAL_ENDINGS: tuple[str, ...] = (
    "ss", "us", "is", "sis", "ics", "ous", "ness", "ese", "ies-",
)
_NOT_PLURAL_WORDS: frozenset[str] = frozenset(
    {
        "news", "series", "species", "physics", "mathematics", "maths", "statistics",
        "economics", "politics", "analytics", "graphics", "ethics", "logistics",
        "robotics", "genetics", "diabetes", "means", "lens", "bias", "canvas",
        "gas", "always", "chaos", "cosmos", "iris", "axis", "basis", "thesis",
        "css", "js", "aws", "ios", "macos", "hers", "its",
    }
)

# `-ies` is genuinely ambiguous: `categories` -> `category` but `cookies` -> `cookie`,
# and nothing in the spelling distinguishes them. The `-ie` family is small enough in
# practice to enumerate, so it is listed and everything else takes the `-y` rule.
_IE_STEMS: frozenset[str] = frozenset(
    {
        "cook", "mov", "brown", "smooth", "calor", "self", "food", "newb", "zomb",
        "rook", "hood", "vegg", "beg", "cad", "goal", "prair", "aud", "spec",
    }
)


def _singularize(text: str) -> str:
    """Crude English singularization, applied to the final word only.

    Conservative on purpose. Over-eager stemming merges genuinely distinct tags, and a
    wrong merge is worse than a missed one: two unrelated topics collapsing into one
    facet is invisible corruption, whereas `recipe` and `recipes` sitting side by side
    is merely untidy.
    """
    head, _, last = text.rpartition("-")
    word = last or text

    if len(word) < 4 or not word.endswith("s"):
        return text
    if word in _NOT_PLURAL_WORDS or word.endswith(_NOT_PLURAL_ENDINGS):
        return text

    if word.endswith("ies"):
        stem = word[:-3]
        word = f"{stem}ie" if stem in _IE_STEMS else f"{stem}y"
    elif word.endswith("sses"):
        # glasses -> glass, classes -> class
        word = word[:-2]
    elif word.endswith(("ches", "shes", "xes", "zes")):
        word = word[:-2]
    elif word.endswith("oes"):
        # tomatoes -> tomato, potatoes -> potato
        word = word[:-2]
    else:
        # databases -> database, recipes -> recipe, tags -> tag
        word = word[:-1]

    return f"{head}-{word}" if head else word


def normalize_tags(raw_tags: object, limit: int = 15) -> list[str]:
    """Normalize, de-duplicate and cap a collection of tags, preserving order."""
    if not raw_tags:
        return []
    if isinstance(raw_tags, str):
        candidates: list[str] = re.split(r"[,\n]", raw_tags)
    elif isinstance(raw_tags, (list, tuple, set)):
        candidates = [str(t) for t in raw_tags]
    else:
        return []

    seen: dict[str, None] = {}
    for candidate in candidates:
        slug = slugify_tag(candidate)
        if slug:
            seen.setdefault(slug, None)
        if len(seen) >= limit:
            break
    return list(seen)


def coerce_category(value: object) -> str:
    slug = slugify_tag(str(value)) if value else None
    if slug in CATEGORIES:
        return str(slug)
    # Accept underscore/space variants the model may emit despite the enum.
    if slug:
        compact = slug.replace("-", "")
        for category in CATEGORIES:
            if category.replace("-", "") == compact:
                return category
    return "other"


def coerce_content_type(value: object) -> str:
    slug = slugify_tag(str(value)) if value else None
    if slug in CONTENT_TYPES:
        return str(slug)
    if slug:
        compact = slug.replace("-", "")
        for content_type in CONTENT_TYPES:
            if content_type.replace("-", "") == compact:
                return content_type
    return "other"
