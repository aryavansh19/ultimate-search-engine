"""CLI for the search stage.

    python -m search add "<url>"              extract + tag + index, one command
    python -m search index                    index everything already extracted
    python -m search query "that pasta video" hybrid search
    python -m search compare "<query>"        keyword vs vector vs hybrid, side by side
    python -m search stats                    index size and vector signatures

`compare` is the one to spend time with. Running the same twenty queries through all
three modes is how you find out where each retriever fails on *your* library, which is
not something anyone can tell you in the abstract.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import SearchConfig
from .indexer import Indexer
from .searcher import SearchMode, Searcher, SearchResponse
from .store import SearchStore


def _print_response(response: SearchResponse, show_text: bool = False) -> None:
    print()
    header = f'  "{response.query}"  [{response.mode.value}]'
    counts = []
    if response.keyword_candidates:
        counts.append(f"{response.keyword_candidates} keyword")
    if response.vector_candidates:
        counts.append(f"{response.vector_candidates} vector")
    if counts:
        header += f"   candidates: {', '.join(counts)}"
    print(header)
    for note in response.notes:
        print(f"  note: {note}")

    if not response.hits:
        print("\n  no results\n")
        return

    print()
    for position, hit in enumerate(response.hits, start=1):
        ranks = []
        if hit.keyword_rank:
            ranks.append(f"kw#{hit.keyword_rank}")
        if hit.vector_rank:
            ranks.append(f"vec#{hit.vector_rank}")
        if hit.vector_similarity is not None:
            ranks.append(f"cos={hit.vector_similarity:.3f}")
        rank_text = " ".join(ranks)

        title = hit.title or hit.summary or hit.canonical_url
        print(f"  {position:>2}. {_clip(title, 78)}")
        print(
            f"      {hit.score:.5f}  {hit.found_by:<8} {rank_text}"
        )
        if hit.summary and hit.summary != title:
            print(f"      {_clip(hit.summary, 96)}")
        meta = [hit.platform]
        if hit.category:
            meta.append(hit.category)
        if hit.content_type:
            meta.append(hit.content_type)
        print(f"      {' | '.join(meta)}")
        if hit.tags:
            print(f"      tags: {', '.join(hit.tags[:8])}")
        if show_text and hit.matched_text:
            print(f"      match: {hit.matched_text}")
        print(f"      {hit.canonical_url}")
        print()


def _clip(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def _cmd_add(args: argparse.Namespace) -> int:
    """Full pipeline for one link: extract, enrich, index."""
    from enrichment import Enricher
    from extractor import ExtractionCascade

    payload = None
    if args.payload:
        from pathlib import Path

        payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))

    config = SearchConfig.from_env()
    cascade = ExtractionCascade()
    enricher = Enricher()
    indexer = Indexer(config)
    try:
        envelope = cascade.extract(args.url, client_payload=payload).envelope
        enrichment = enricher.enrich(envelope, force=args.force)
        indexer.index_one(envelope, enrichment, force=True, embed=not args.no_embed)
        print()
        print(f"  saved      {envelope.canonical_url}")
        print(f"  extracted  {envelope.tier.value}, signal {envelope.signal.value}")
        print(
            f"  tagged     {enrichment.mode.value} via {enrichment.provider}"
            f"  ${enrichment.cost_usd:.6f}"
        )
        print(f"  summary    {_clip(enrichment.summary, 90)}")
        print(f"  tags       {', '.join(enrichment.tags) or '-'}")
        print(f"  indexed    embeddings ${indexer.embedder.estimated_cost_usd:.6f} (est)")
        print()
        return 0
    finally:
        indexer.close()
        enricher.close()
        cascade.close()


def _cmd_index(args: argparse.Namespace) -> int:
    indexer = Indexer(SearchConfig.from_env())
    try:
        report = indexer.index_all(
            limit=args.limit, force=args.force, embed=not args.no_embed
        )
        print()
        print(
            f"  indexed {report.indexed}   skipped {report.skipped}   "
            f"failed {report.failed}"
        )
        print(f"  chunks {report.chunks}   vectors {report.vectors}")
        print(f"  embedding cost ~${report.estimated_cost_usd:.6f} (estimated)")
        for note in report.notes or []:
            print(f"  note: {note}")
        print()
        return 0 if report.failed == 0 else 1
    finally:
        indexer.close()


def _cmd_query(args: argparse.Namespace) -> int:
    searcher = Searcher(SearchConfig.from_env())
    try:
        response = searcher.search(
            args.query,
            limit=args.limit,
            mode=SearchMode(args.mode),
            category=args.category,
            content_type=args.type,
            platform=args.platform,
            tag=args.tag,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "query": response.query,
                        "mode": response.mode.value,
                        "notes": response.notes,
                        "hits": [
                            {
                                "url": h.canonical_url,
                                "title": h.title,
                                "summary": h.summary,
                                "score": h.score,
                                "found_by": h.found_by,
                                "keyword_rank": h.keyword_rank,
                                "vector_rank": h.vector_rank,
                                "cosine": h.vector_similarity,
                                "category": h.category,
                                "tags": h.tags,
                            }
                            for h in response.hits
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        _print_response(response, show_text=args.show_text)
        return 0
    finally:
        searcher.close()


def _cmd_compare(args: argparse.Namespace) -> int:
    """Same query through all three modes, so the difference is visible."""
    searcher = Searcher(SearchConfig.from_env())
    try:
        print()
        print(f'  query: "{args.query}"')
        rows: dict[str, list[str]] = {}
        for mode in (SearchMode.KEYWORD, SearchMode.VECTOR, SearchMode.HYBRID):
            response = searcher.search(args.query, limit=args.limit, mode=mode)
            rows[mode.value] = [
                _clip(hit.title or hit.summary or hit.canonical_url, 46)
                for hit in response.hits
            ]
            for note in response.notes:
                print(f"  note ({mode.value}): {note}")

        width = 48
        print()
        print(
            f"  {'#':<3}{'keyword':<{width}}{'vector':<{width}}{'hybrid':<{width}}"
        )
        print(f"  {'-' * (3 + width * 3)}")
        depth = max((len(v) for v in rows.values()), default=0)
        for index in range(depth):
            cells = []
            for mode in ("keyword", "vector", "hybrid"):
                values = rows.get(mode, [])
                cells.append(values[index] if index < len(values) else "")
            print(
                f"  {index + 1:<3}{cells[0]:<{width}}{cells[1]:<{width}}{cells[2]:<{width}}"
            )
        print()
        return 0
    finally:
        searcher.close()


def _cmd_stats(args: argparse.Namespace) -> int:
    config = SearchConfig.from_env()
    store = SearchStore(config.store_path)
    try:
        stats = store.stats()
        print()
        print(
            f"  documents {stats['documents']}   enriched {stats['enriched']}   "
            f"chunks {stats['chunks']}   vectors {stats['vectors']}"
        )
        print(f"  active model {config.embed_model} @ {config.dimensions} dims")
        signatures = stats["signatures"]
        if isinstance(signatures, list) and signatures:
            print("\n  stored vector signatures")
            for model, dim, count in signatures:
                marker = (
                    "  <- active"
                    if model == config.embed_model and dim == config.dimensions
                    else "  (stale, re-index to use)"
                )
                print(f"    {model}@{dim}  {count} vectors{marker}")
        platforms = stats["platforms"]
        if isinstance(platforms, dict) and platforms:
            print(
                "\n  platforms  "
                + ", ".join(f"{k} {v}" for k, v in platforms.items())
            )
        print()
        return 0
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass

    parser = argparse.ArgumentParser(prog="search", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="extract, tag and index one URL")
    add.add_argument("url")
    add.add_argument("--payload", help="JSON file of client-extracted content")
    add.add_argument("--force", action="store_true", help="re-tag even if cached")
    add.add_argument("--no-embed", action="store_true", help="keyword index only")
    add.set_defaults(func=_cmd_add)

    index = sub.add_parser("index", help="index everything already extracted")
    index.add_argument("--limit", type=int, default=1000)
    index.add_argument("--force", action="store_true", help="re-index unchanged items")
    index.add_argument("--no-embed", action="store_true", help="keyword index only")
    index.set_defaults(func=_cmd_index)

    query = sub.add_parser("query", help="search the index")
    query.add_argument("query")
    query.add_argument("--limit", type=int, default=10)
    query.add_argument(
        "--mode", choices=[m.value for m in SearchMode], default=SearchMode.HYBRID.value
    )
    query.add_argument("--category")
    query.add_argument("--type", dest="type")
    query.add_argument("--platform")
    query.add_argument("--tag")
    query.add_argument("--show-text", action="store_true", help="show matched chunk")
    query.add_argument("--json", action="store_true")
    query.set_defaults(func=_cmd_query)

    compare = sub.add_parser("compare", help="keyword vs vector vs hybrid side by side")
    compare.add_argument("query")
    compare.add_argument("--limit", type=int, default=5)
    compare.set_defaults(func=_cmd_compare)

    stats = sub.add_parser("stats", help="index size and vector signatures")
    stats.set_defaults(func=_cmd_stats)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
