"""Reciprocal Rank Fusion.

The reason fusion happens on *ranks* rather than scores: BM25 returns unbounded values
whose scale shifts with corpus statistics, while cosine similarity sits in [-1, 1]. There
is no honest way to add those together, and normalizing them per query (min-max over the
result set) makes a document's score depend on which other documents happened to come
back. RRF sidesteps the whole problem by discarding magnitudes and keeping only order.

    score(d) = sum over retrievers of  weight / (k + rank(d))

k=60 comes from the original paper. Its practical effect is to flatten the difference
between the top few positions so that one retriever's confident first place cannot
dominate a document the other retriever also liked -- which is exactly the behaviour you
want when neither retriever is reliably better than the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FusedItem:
    key: str
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        return sorted(self.ranks)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[FusedItem]:
    """Fuse named ranked lists of keys into one ranking, best first.

    Ties are broken by the best single rank achieved, then by key, so ordering is stable
    across runs. Non-deterministic result order is miserable to debug and makes any
    before/after comparison untrustworthy.
    """
    weights = weights or {}
    fused: dict[str, FusedItem] = {}

    for source, keys in ranked_lists.items():
        weight = weights.get(source, 1.0)
        if weight == 0:
            continue
        for position, key in enumerate(keys, start=1):
            item = fused.setdefault(key, FusedItem(key=key))
            contribution = weight / (k + position)
            item.score += contribution
            # Keep the best rank if a key somehow appears twice in one list.
            if source not in item.ranks or position < item.ranks[source]:
                item.ranks[source] = position
            item.contributions[source] = contribution

    def sort_key(item: FusedItem) -> tuple[float, int, str]:
        best_rank = min(item.ranks.values()) if item.ranks else 10**9
        return (-item.score, best_rank, item.key)

    return sorted(fused.values(), key=sort_key)
