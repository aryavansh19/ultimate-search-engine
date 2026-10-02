"""CLI for the enrichment stage.

    python -m enrichment enrich "<url>"            extract + tag one link
    python -m enrichment enrich "<url>" --mode media   force the expensive path
    python -m enrichment backfill --limit 20       tag everything already extracted
    python -m enrichment cost                      what you have spent, by mode/model
    python -m enrichment tags                      tag frequency, to spot drift
    python -m enrichment budget 47                 price a video before calling

`cost` and `tags` are the two worth checking regularly. The first tells you whether the
escalation gate is doing its job; the second tells you whether your tag vocabulary is
staying coherent as the library grows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from extractor import ExtractionCascade

from .config import EnrichmentConfig
from .enricher import Enricher
from .pricing import PRICE_INCREASE_NOTE, describe_video_budget
from .schema import Enrichment, EnrichmentMode
from .store import EnrichmentStore


def _print_enrichment(enrichment: Enrichment, envelope_note: str | None = None) -> None:
    print()
    if envelope_note:
        print(f"  source     {envelope_note}")
    print(f"  mode       {enrichment.mode.value}   provider {enrichment.provider}")
    print(f"  model      {enrichment.model or '-'}")
    print(
        f"  tokens     in {enrichment.input_tokens}  out {enrichment.output_tokens}"
        f"   cost ${enrichment.cost_usd:.6f}   {enrichment.duration_ms}ms"
    )
    print(f"  degraded   {enrichment.degraded}")
    if enrichment.note:
        print(f"  note       {enrichment.note}")
    print()
    print(f"  summary    {enrichment.summary}")
    if enrichment.description:
        print(f"  detail     {_wrap(enrichment.description, 11)}")
    print(f"  category   {enrichment.category}   type {enrichment.content_type}")
    if enrichment.language:
        print(f"  language   {enrichment.language}")
    print(f"  tags       {', '.join(enrichment.tags) or '-'}")

    entities = enrichment.entities
    for label, values in (
        ("people", entities.people),
        ("orgs", entities.organizations),
        ("places", entities.places),
        ("products", entities.products),
    ):
        if values:
            print(f"  {label:<11}{', '.join(values)}")
    print()


def _wrap(text: str, indent: int, width: int = 88) -> str:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    pad = " " * (indent + 2)
    return ("\n" + pad).join(lines)


def _load_payload(path: str | None) -> dict | None:
    """Load a client share payload, the same shape the iOS app will POST."""
    if not path:
        return None
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("payload file must contain a JSON object")
    return data


def _cmd_enrich(args: argparse.Namespace) -> int:
    config = EnrichmentConfig.from_env()
    if getattr(args, "provider", None):
        config.provider_order = tuple(
            p.strip().lower() for p in args.provider.split(",") if p.strip()
        )
    cascade = ExtractionCascade()
    enricher = Enricher(config)
    try:
        result = cascade.extract(
            args.url, client_payload=_load_payload(args.payload)
        )
        envelope = result.envelope
        mode = EnrichmentMode(args.mode) if args.mode else None
        enrichment = enricher.enrich(envelope, force=args.force, mode=mode)

        if args.json:
            print(enrichment.model_dump_json(indent=2, exclude_none=True))
            return 0

        note = (
            f"{envelope.platform.value} via {envelope.tier.value}, "
            f"signal {envelope.signal.value} ({envelope.word_count} words)"
        )
        _print_enrichment(enrichment, note)
        return 0
    finally:
        enricher.close()
        cascade.close()


def _cmd_preview(args: argparse.Namespace) -> int:
    """Show the exact Gemini request for a URL without sending it."""
    from .prompts import trim_document
    from .providers import GeminiProvider

    config = EnrichmentConfig.from_env()
    cascade = ExtractionCascade()
    enricher = Enricher(config)
    try:
        envelope = cascade.extract(
            args.url, client_payload=_load_payload(args.payload)
        ).envelope
        mode = EnrichmentMode(args.mode) if args.mode else enricher.decide_mode(envelope)
        document = trim_document(envelope.search_document(), config.max_text_chars)
        model, payload = GeminiProvider(config).build_request(envelope, mode, document)
        print(
            json.dumps(
                {
                    "endpoint": f"{config.api_base}/models/{model}:generateContent",
                    "mode": mode.value,
                    "request": payload,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        enricher.close()
        cascade.close()


def _cmd_backfill(args: argparse.Namespace) -> int:
    enricher = Enricher()
    try:
        results = enricher.enrich_pending(
            limit=args.limit, upgrade_degraded=not args.keep_degraded
        )
        if not results:
            print("\n  nothing pending\n")
            return 0
        total = sum(e.cost_usd for e in results)
        print(f"\n  enriched {len(results)} item(s), ${total:.6f} total\n")
        for enrichment in results:
            flag = "!" if enrichment.degraded else " "
            print(
                f"  {flag} {enrichment.mode.value:<10}{enrichment.category:<22}"
                f"{enrichment.summary[:60]}"
            )
        print()
        return 0
    finally:
        enricher.close()


def _cmd_providers(args: argparse.Namespace) -> int:
    """Show the provider chain, what each supports, and whether it is usable."""
    from .enricher import build_providers
    from .schema import EnrichmentMode as Mode

    config = EnrichmentConfig.from_env()
    print()
    print(f"  ENRICH_PROVIDER = {','.join(config.provider_order)}")
    print()
    print(f"  {'provider':<16}{'ready':<8}{'text':<7}{'media':<7}model / reason")
    print(f"  {'-' * 74}")
    for provider in build_providers(config):
        ok, reason = provider.availability()
        text = "yes" if provider.supports(Mode.METADATA) else "-"
        media = "yes" if provider.supports(Mode.MEDIA) else "-"
        if provider.name == "gemini":
            detail = f"{config.text_model} / {config.media_model}"
        elif provider.name == "heuristic":
            detail = "offline keyword extraction"
        else:
            detail = config.compat_model
        print(
            f"  {provider.name:<16}{('yes' if ok else 'no'):<8}{text:<7}{media:<7}"
            f"{detail if ok else (reason or 'unavailable')}"
        )
    print()
    print("  Only gemini can analyse video: the gateway accepts a video part, silently")
    print("  ignores it, and answers from training knowledge instead.")
    print("  Embeddings for search always use gemini; the gateway has no /embeddings route.")
    print()
    return 0


def _cmd_cost(args: argparse.Namespace) -> int:
    store = EnrichmentStore(EnrichmentConfig.from_env().store_path)
    try:
        summary = store.cost_summary()
        items = int(summary["items"])  # type: ignore[arg-type]
        print(f"\n  items {items}")
        print(
            f"  spend ${float(summary['total_cost_usd']):.6f} total, "
            f"${float(summary['avg_cost_usd']):.6f} per item"
        )
        print(
            f"  tokens in {summary['input_tokens']}  out {summary['output_tokens']}"
        )
        by_mode = summary["by_mode"]
        if isinstance(by_mode, list) and by_mode:
            print(f"\n  {'mode':<12}{'items':>6}{'cost':>12}{'avg':>9}")
            for row in by_mode:
                print(
                    f"  {str(row['mode']):<12}{row['items']:>6}"
                    f"{float(row['cost_usd']):>12.6f}{row['avg_ms']:>8}ms"
                )
        by_model = summary["by_model"]
        if isinstance(by_model, list) and by_model:
            print(f"\n  {'model':<26}{'items':>6}{'cost':>12}")
            for row in by_model:
                print(
                    f"  {str(row['model']):<26}{row['items']:>6}"
                    f"{float(row['cost_usd']):>12.6f}"
                )
        print(f"\n  note: {PRICE_INCREASE_NOTE}\n")
        return 0
    finally:
        store.close()


def _cmd_tags(args: argparse.Namespace) -> int:
    store = EnrichmentStore(EnrichmentConfig.from_env().store_path)
    try:
        rows = store.top_tags(limit=args.limit)
        if not rows:
            print("\n  no tags yet\n")
            return 0
        print()
        for tag, uses in rows:
            print(f"  {uses:>4}  {tag}")
        print()
        return 0
    finally:
        store.close()


def _cmd_budget(args: argparse.Namespace) -> int:
    config = EnrichmentConfig.from_env()
    estimate = describe_video_budget(
        args.seconds, config.media_model, high_resolution=args.high
    )
    print()
    for key, value in estimate.items():
        label = key.replace("_", " ")
        if key == "est_cost_usd":
            print(f"  {label:<20}${float(value):.6f}")
        else:
            print(f"  {label:<20}{value}")
    print(f"\n  ceiling per item     ${config.max_cost_per_item_usd:.4f}")
    print(f"  text-mode model      {config.text_model}")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass

    parser = argparse.ArgumentParser(prog="enrichment", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    enrich = sub.add_parser("enrich", help="extract and tag one URL")
    enrich.add_argument("url")
    enrich.add_argument(
        "--mode",
        choices=[m.value for m in EnrichmentMode],
        help="override the automatic mode decision",
    )
    enrich.add_argument("--force", action="store_true", help="ignore stored enrichment")
    enrich.add_argument(
        "--payload",
        help="JSON file of client-extracted content, as the iOS app will send",
    )
    enrich.add_argument(
        "--provider",
        help="override ENRICH_PROVIDER for this run, e.g. 'ashna' or 'ashna,gemini'",
    )
    enrich.add_argument("--json", action="store_true")
    enrich.set_defaults(func=_cmd_enrich)

    preview = sub.add_parser(
        "preview", help="print the exact model request without sending it"
    )
    preview.add_argument("url")
    preview.add_argument("--mode", choices=[m.value for m in EnrichmentMode])
    preview.add_argument("--payload", help="JSON file of client-extracted content")
    preview.set_defaults(func=_cmd_preview)

    backfill = sub.add_parser("backfill", help="tag already-extracted items")
    backfill.add_argument("--limit", type=int, default=25)
    backfill.add_argument(
        "--keep-degraded",
        action="store_true",
        help="do not re-run items that were tagged by the offline fallback",
    )
    backfill.set_defaults(func=_cmd_backfill)

    cost = sub.add_parser("cost", help="spend by mode and model")
    cost.set_defaults(func=_cmd_cost)

    providers = sub.add_parser("providers", help="show which providers are usable")
    providers.set_defaults(func=_cmd_providers)

    tags = sub.add_parser("tags", help="tag frequency across the library")
    tags.add_argument("--limit", type=int, default=30)
    tags.set_defaults(func=_cmd_tags)

    budget = sub.add_parser("budget", help="estimate the cost of a video call")
    budget.add_argument("seconds", type=float)
    budget.add_argument("--high", action="store_true", help="high media resolution")
    budget.set_defaults(func=_cmd_budget)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
