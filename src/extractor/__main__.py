"""CLI for exercising the cascade against real links.

    python -m extractor extract "https://www.reddit.com/r/python/comments/..."
    python -m extractor extract "<url>" --payload share.json
    python -m extractor stats
    python -m extractor canonical "<url>"
    python -m extractor degraded
    python -m extractor forget "<url>"

`stats` is the point of the whole telemetry layer: run thirty or forty real links
you actually saved, then read the per-tier success rates before deciding whether a
paid scraper subscription is worth it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .cache import EnvelopeCache
from .cascade import ExtractionCascade
from .config import ExtractorConfig
from .envelope import ContentEnvelope, ExtractionResult
from .urls import canonicalize, detect_platform, url_hash


def _print_result(result: ExtractionResult, show_document: bool) -> None:
    envelope = result.envelope
    print()
    print(f"  url        {envelope.canonical_url}")
    print(f"  hash       {envelope.url_hash[:16]}...")
    print(f"  platform   {envelope.platform.value}")
    print(f"  tier       {envelope.tier.value}" + ("  (cached)" if result.cache_hit else ""))
    print(f"  note       {envelope.extractor_note or '-'}")
    print(f"  media      {envelope.media_kind.value}", end="")
    if envelope.duration_s:
        print(f"  {envelope.duration_s:.0f}s", end="")
    print()
    print(f"  signal     {envelope.signal.value}  ({envelope.word_count} words)")
    print(f"  degraded   {envelope.degraded}")
    if envelope.degraded and not envelope.text_parts:
        next_step = "nothing extracted; queued for reprocessing"
    elif envelope.needs_media_understanding:
        next_step = "escalate to multimodal video understanding"
    else:
        next_step = "metadata-only enrichment is enough"
    print(f"  next step  {next_step}")

    if result.attempts:
        print("\n  attempts")
        for attempt in result.attempts:
            if attempt.skipped:
                status = "skip"
            elif attempt.ok:
                status = "ok"
            else:
                status = "fail"
            timing = f"{attempt.duration_ms:>6}ms" if not attempt.skipped else " " * 8
            reason = f"  {attempt.reason}" if attempt.reason else ""
            print(f"    {status:<4} {attempt.tier.value:<15}{timing}{reason}")

    print("\n  fields")
    for label, value in (
        ("title", envelope.title),
        ("author", envelope.author),
        ("caption", envelope.caption),
        ("tags", ", ".join(envelope.platform_tags) or None),
        ("thumbnail", envelope.thumbnail_url),
        ("transcript", envelope.transcript),
        ("article", envelope.article_text),
    ):
        if not value:
            continue
        flat = " ".join(str(value).split())
        if len(flat) > 160:
            flat = flat[:157] + "..."
        print(f"    {label:<11}{flat}")

    if show_document:
        print("\n  search document")
        for line in envelope.search_document().splitlines():
            print(f"    {line[:200]}")
    print()


def _load_payload(path: str | None) -> dict | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("payload file must contain a JSON object")
    return data


def _cmd_extract(args: argparse.Namespace) -> int:
    cascade = ExtractionCascade(ExtractorConfig.from_env())
    try:
        result = cascade.extract(
            args.url,
            client_payload=_load_payload(args.payload),
            use_cache=not args.no_cache,
            force=args.force,
            retry_degraded=args.retry_degraded,
        )
        if args.json:
            print(result.model_dump_json(indent=2, exclude_none=True))
        else:
            _print_result(result, show_document=args.document)
        return 0 if not result.envelope.degraded else 1
    finally:
        cascade.close()


def _cmd_stats(args: argparse.Namespace) -> int:
    cache = EnvelopeCache(ExtractorConfig.from_env().cache_path)
    try:
        summary = cache.summary()
        print(f"\n  items {summary['items']}   degraded {summary['degraded']}")
        if summary["by_platform"]:
            parts = ", ".join(f"{k} {v}" for k, v in summary["by_platform"].items())  # type: ignore[union-attr]
            print(f"  platforms  {parts}")
        if summary["by_signal"]:
            parts = ", ".join(f"{k} {v}" for k, v in summary["by_signal"].items())  # type: ignore[union-attr]
            print(f"  signal     {parts}")

        stats = cache.tier_stats()
        if not stats:
            print("\n  no attempts recorded yet\n")
            return 0
        print(f"\n  {'tier':<16}{'ran':>5}{'wins':>6}{'skipped':>9}{'success':>9}{'avg':>9}")
        for row in stats:
            rate = row["success_rate"]
            rate_text = f"{float(rate) * 100:.0f}%" if rate is not None else "-"
            avg = row["avg_ms"]
            avg_text = f"{avg}ms" if avg else "-"
            print(
                f"  {str(row['tier']):<16}{row['ran']:>5}{row['wins']:>6}"
                f"{row['skips']:>9}{rate_text:>9}{avg_text:>9}"
            )
        print()
        return 0
    finally:
        cache.close()


def _cmd_canonical(args: argparse.Namespace) -> int:
    canonical = canonicalize(args.url, follow_shorteners=not args.offline)
    print(f"\n  input      {args.url}")
    print(f"  canonical  {canonical}")
    print(f"  platform   {detect_platform(canonical).value}")
    print(f"  hash       {url_hash(canonical)}\n")
    return 0


def _cmd_degraded(args: argparse.Namespace) -> int:
    cache = EnvelopeCache(ExtractorConfig.from_env().cache_path)
    try:
        items: list[ContentEnvelope] = cache.degraded_items(limit=args.limit)
        if not items:
            print("\n  nothing degraded\n")
            return 0
        print(f"\n  {len(items)} item(s) awaiting reprocessing\n")
        for envelope in items:
            print(f"    {envelope.platform.value:<11}{envelope.canonical_url}")
            print(f"    {'':<11}{envelope.extractor_note or '-'}")
        print()
        return 0
    finally:
        cache.close()


def _cmd_forget(args: argparse.Namespace) -> int:
    cache = EnvelopeCache(ExtractorConfig.from_env().cache_path)
    try:
        canonical = canonicalize(args.url, follow_shorteners=False)
        removed = cache.delete(url_hash(canonical))
        print(("  removed " if removed else "  not cached ") + canonical)
        return 0
    finally:
        cache.close()


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, which raises UnicodeEncodeError on the first
    # arrow, emoji or smart quote in a caption -- and real captions are full of them.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass

    parser = argparse.ArgumentParser(prog="extractor", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="run the cascade on a URL")
    extract.add_argument("url")
    extract.add_argument("--payload", help="JSON file simulating a client share payload")
    extract.add_argument("--json", action="store_true", help="raw JSON output")
    extract.add_argument("--document", action="store_true", help="show the search document")
    extract.add_argument("--no-cache", action="store_true", help="neither read nor write cache")
    extract.add_argument("--force", action="store_true", help="ignore a cached envelope")
    extract.add_argument(
        "--retry-degraded", action="store_true", help="re-run if the cached item is degraded"
    )
    extract.set_defaults(func=_cmd_extract)

    stats = sub.add_parser("stats", help="per-tier success rates and latency")
    stats.set_defaults(func=_cmd_stats)

    canonical = sub.add_parser("canonical", help="show canonical URL, platform and hash")
    canonical.add_argument("url")
    canonical.add_argument("--offline", action="store_true", help="do not resolve shorteners")
    canonical.set_defaults(func=_cmd_canonical)

    degraded = sub.add_parser("degraded", help="list items awaiting reprocessing")
    degraded.add_argument("--limit", type=int, default=50)
    degraded.set_defaults(func=_cmd_degraded)

    forget = sub.add_parser("forget", help="drop a cached envelope")
    forget.add_argument("url")
    forget.set_defaults(func=_cmd_forget)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
