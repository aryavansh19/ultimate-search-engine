"""Automated Benchmark Harness for LinQ Search Backend.

Runs empirical latency, quota, and reliability evaluations across real-world URLs.
Measures:
- Extraction wall-clock time and active extraction tier (ClientPayload / YtDlp / ManagedApi / OpenGraph)
- Enrichment multimodal tagging time and token count (Gemini Flash)
- Embedding generation time (Nemotron / Gemini)
- Upstream quota consumption and sliding-window RPM/TPM saturation

Usage:
    python3 -m benchmarks.run_benchmark --tag v1.0-baseline
    python3 -m benchmarks.run_benchmark --urls benchmarks/my_urls.json --tag v1.1-pooled
    python3 -m benchmarks.run_benchmark --extract-only --tag v1.0-extract-baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure src is on sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from enrichment import Enricher, EnrichmentConfig  # noqa: E402
from enrichment.schema import EnrichmentMode  # noqa: E402
from extractor import ExtractionCascade, ExtractorConfig  # noqa: E402
from quota import get_all_telemetry  # noqa: E402
from search import SearchConfig  # noqa: E402
from search.chunking import chunk_document  # noqa: E402
from search.embedder import build_embedder  # noqa: E402

from .results_store import BenchmarkLedger, BenchmarkSummary, ItemBenchmarkResult  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger("benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LinQ Search Engine Benchmark Harness")
    parser.add_argument(
        "--urls",
        type=str,
        default="",
        help="Path to JSON file containing test URLs (defaults to benchmarks/test_urls.json or test_urls.sample.json)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="v1.0-baseline",
        help="Identifier tag for this benchmark run (e.g. v1.0-baseline, v1.1-pooled)",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Benchmark extraction cascade only (skip LLM enrichment and embedding)",
    )
    parser.add_argument(
        "--force-media",
        action="store_true",
        help="Force multimodal video analysis where available",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit benchmark to first N URLs",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass local cache to test live network performance",
    )
    return parser.parse_args()


def load_urls(path_str: str) -> list[dict[str, Any]]:
    benchmarks_dir = Path(__file__).resolve().parent
    candidate_paths = [
        Path(path_str) if path_str else None,
        benchmarks_dir / "corpus" / "corpus-v3.0.json",
        benchmarks_dir / "corpus" / "corpus-v1.0.json",
        benchmarks_dir / "test_urls.json",
        benchmarks_dir / "test_urls.sample.json",
    ]

    chosen: Path | None = None
    for p in candidate_paths:
        if p and p.is_file():
            chosen = p
            break

    if not chosen:
        print(f"Error: No URLs file found. Create {benchmarks_dir / 'test_urls.json'}")
        sys.exit(1)

    print(f"[*] Loading test URLs from: {chosen}")
    with open(chosen, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: URLs file must contain a JSON array of objects.")
        sys.exit(1)

    return data


def format_hud(telemetry: dict[str, Any]) -> str:
    lines = ["\n========================= UPSTREAM QUOTA HUD ========================="]
    for name, stats in telemetry.items():
        rpm = stats.get("rpm", 0)
        max_rpm = stats.get("max_rpm", 1)
        rpm_pct = stats.get("rpm_utilization_pct", 0.0)
        status = stats.get("status", "unknown").upper()
        tokens = stats.get("lifetime_tokens", 0)
        cooldown = stats.get("cooldown_remaining_s", 0.0)

        bar_len = 20
        filled = int((rpm_pct / 100.0) * bar_len)
        bar = "#" * min(filled, bar_len) + "-" * max(0, bar_len - filled)

        line = (
            f"[{name.upper():<8}] [{bar}] {rpm:>2}/{int(max_rpm):<2} RPM ({rpm_pct:>5.1f}%) "
            f"| Status: {status:<10} | Tokens: {tokens:<7} | Cooldown: {cooldown:.1f}s"
        )
        lines.append(line)
    lines.append("======================================================================\n")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    urls_data = load_urls(args.urls)

    if args.limit > 0:
        urls_data = urls_data[: args.limit]

    total_items = len(urls_data)
    print(f"[*] Starting benchmark run `{args.tag}` with {total_items} items...")
    if args.extract_only:
        print("[*] Mode: EXTRACTION-ONLY (Enrichment and Embedding skipped)")
    else:
        print("[*] Mode: FULL PIPELINE (Extraction -> Gemini Enrichment -> Embedding)")

    cascade = ExtractionCascade(ExtractorConfig.from_env())
    enricher = Enricher(EnrichmentConfig.from_env()) if not args.extract_only else None
    embedder = build_embedder(SearchConfig.from_env()) if not args.extract_only else None

    results: list[ItemBenchmarkResult] = []

    print("-" * 80)
    for idx, item in enumerate(urls_data, start=1):
        item_id = str(item.get("id") or f"item_{idx}")
        url = str(item.get("final_url") or item.get("original_url") or item.get("normalized_url") or item.get("url") or "")
        platform = str(item.get("platform") or "web")
        genre = str(item.get("genre") or item.get("page_type") or "general")

        if not url:
            continue

        item_start = time.perf_counter()
        extr_ms = 0.0
        enr_ms = 0.0
        emb_ms = 0.0
        total_tokens = 0
        tag_count = 0
        detail_count = 0
        tier = "none"
        title: str | None = None
        author: str | None = None
        media_kind: str | None = None
        error_msg: str | None = None
        success = False

        try:
            # 1. Extraction Stage
            t0 = time.perf_counter()
            extr_result = cascade.extract(url, force=args.no_cache)
            extr_ms = (time.perf_counter() - t0) * 1000.0
            envelope = extr_result.envelope
            tier = envelope.tier.value if envelope.tier else "unknown"
            title = envelope.title
            author = envelope.author
            media_kind = envelope.media_kind.value if envelope.media_kind else None

            # 2. Enrichment Stage
            if enricher and not args.extract_only:
                t1 = time.perf_counter()
                mode = EnrichmentMode.MEDIA if args.force_media else EnrichmentMode.METADATA
                doc_text = f"{envelope.title or ''}\n{envelope.caption or ''}\n{envelope.article_text or ''}".strip()
                enrichment = enricher.enrich(envelope, mode=mode, document=doc_text)
                enr_ms = (time.perf_counter() - t1) * 1000.0
                tag_count = len(enrichment.tags)
                detail_count = len(enrichment.details)
                total_tokens = (enrichment.input_tokens or 0) + (enrichment.output_tokens or 0)

                # 3. Embedding Stage
                if embedder:
                    t2 = time.perf_counter()
                    chunks = chunk_document(
                        title=envelope.title or "",
                        caption=envelope.caption or "",
                        body_text=envelope.article_text or "",
                        tags=enrichment.tags,
                        details=enrichment.details,
                    )
                    texts = [c.text for c in chunks]
                    if texts:
                        embedder.embed_documents(texts)
                    emb_ms = (time.perf_counter() - t2) * 1000.0

            success = bool(envelope.title or envelope.caption or envelope.article_text)
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            success = False

        total_ms = (time.perf_counter() - item_start) * 1000.0

        res = ItemBenchmarkResult(
            item_id=item_id,
            url=url,
            platform=platform,
            genre=genre,
            success=success,
            tier=tier,
            extraction_ms=extr_ms,
            enrichment_ms=enr_ms,
            embedding_ms=emb_ms,
            total_ms=total_ms,
            title=title,
            author=author,
            media_kind=media_kind,
            tag_count=tag_count,
            detail_count=detail_count,
            token_count=total_tokens,
            error=error_msg,
        )
        results.append(res)

        status_str = "[PASS]" if success else "[FAIL]"
        clean_title = (title or error_msg or "No title")[:32]
        print(
            f"[{idx:03d}/{total_items:03d}] {status_str} [{platform.upper():<12}] "
            f"Total: {total_ms:>6.0f}ms (extr: {extr_ms:>4.0f}ms) "
            f"| Tier: {tier:<10} | \"{clean_title}\"",
            flush=True,
        )

    print("-" * 80)

    # Telemetry Snapshot
    telemetry = get_all_telemetry()
    print(format_hud(telemetry))

    # Aggregations
    successful_items = [r for r in results if r.success]
    succ_count = len(successful_items)
    succ_rate = (succ_count / total_items * 100.0) if total_items else 0.0

    avg_extr = sum(r.extraction_ms for r in results) / total_items if total_items else 0.0
    avg_enr = sum(r.enrichment_ms for r in results) / total_items if total_items else 0.0
    avg_emb = sum(r.embedding_ms for r in results) / total_items if total_items else 0.0
    avg_tot = sum(r.total_ms for r in results) / total_items if total_items else 0.0

    all_totals = sorted(r.total_ms for r in results)
    p95_idx = int(len(all_totals) * 0.95)
    p95_tot = all_totals[min(p95_idx, len(all_totals) - 1)] if all_totals else 0.0
    total_tokens_all = sum(r.token_count for r in results)

    summary = BenchmarkSummary(
        tag=args.tag,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_items=total_items,
        successful_items=succ_count,
        success_rate_pct=succ_rate,
        avg_extraction_ms=avg_extr,
        avg_enrichment_ms=avg_enr,
        avg_embedding_ms=avg_emb,
        avg_total_ms=avg_tot,
        p95_total_ms=p95_tot,
        total_tokens=total_tokens_all,
        quota_telemetry=telemetry,
        items=results,
    )

    ledger = BenchmarkLedger()
    saved_file = ledger.save_run(summary)
    print(f"\n[+] Benchmark complete! Saved JSON metrics to: {saved_file}")
    print(f"[+] Updated Markdown comparison ledger: {ledger.ledger_file}")


if __name__ == "__main__":
    main()
