#!/usr/bin/env python3
"""LinQ Live URL Analyzer CLI

Test any URL live to see exact latency, extraction tier, and response payload.
Usage:
    .venv/bin/python test_link.py "<URL>"
    .venv/bin/python test_link.py "https://www.instagram.com/reel/..."
"""

import json
import sys
import time
from pathlib import Path

# Setup paths
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from linq_server.models import AnalyzeRequest
from linq_server.service import AnalysisService


def main() -> None:
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  .venv/bin/python test_link.py \"<URL>\"\n")
        print("Example:")
        print("  .venv/bin/python test_link.py \"https://www.instagram.com/reel/DePWnYvJEVL/\"\n")
        sys.exit(1)

    url = sys.argv[1].strip()
    print("=" * 75)
    print(f"[*] LINQ LIVE PIPELINE TEST: {url}")
    print("=" * 75)

    svc = AnalysisService()

    print("[*] Running pipeline...")
    t0 = time.perf_counter()
    req = AnalyzeRequest(url=url, embed=True)
    resp = svc.analyze(req)
    total_ms = (time.perf_counter() - t0) * 1000.0

    d = resp.model_dump()
    extr = d.get("extraction") or {}

    extr_ok = not extr.get("degraded", True)
    print(f"\n[+] PIPELINE COMPLETED IN: {total_ms:.2f} ms")
    print("-" * 75)
    print(f"  • Extraction:     {'SUCCESS (PASS)' if extr_ok else 'DEGRADED (FAIL)'}")
    print(f"  • Winning Tier:   {extr.get('tier')} ({extr.get('note')})")
    print(f"  • Signal:         {extr.get('signal')}")
    print(f"  • Media Kind:     {extr.get('media_kind')}")
    print(f"  • Word Count:     {extr.get('word_count')} words")
    print(f"  • Title:          {d.get('title')}")
    print(f"  • Tags ({len(d.get('tags', []))}):        {d.get('tags', [])[:8]}")
    print(f"  • Category:       {d.get('category')} | Content-Type: {d.get('content_type')}")
    print(f"  • Entities:       {d.get('entities')}")
    print(f"  • Enrichment:     {d.get('mode')} ({d.get('provider')})")
    print("-" * 75)
    print("[*] COMPLETE RESPONSE PAYLOAD TO LINQ IOS APP:")
    print(json.dumps(d, indent=2))
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
