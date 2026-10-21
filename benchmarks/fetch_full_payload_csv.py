#!/usr/bin/env python3
"""Generates a single CSV containing the complete raw payload response from Render backend.

Columns:
  - Index
  - Platform
  - Target URL
  - Status Code
  - Latency (ms)
  - Outcome
  - Full Response Payload JSON (Complete unfragmented JSON response)
  - Details Count
  - Tags Count
  - Extracted Word Count
"""

import concurrent.futures
import csv
import json
import os
import threading
import time
from pathlib import Path

import httpx
from benchmarks.benchmark_render import get_token_pool

RENDER_URL = "https://linq-enrichment.onrender.com/v1/analyze"
URLS_FILE = Path("benchmarks/corpus/benchmark_100.json")
OUTPUT_CSV = Path("/Users/anks/Desktop/render_full_payload_responses.csv")
LOCAL_CSV = Path("benchmarks/reports/render_full_payload_responses.csv")
LINKS_DB_CSV = Path("/Users/anks/Desktop/links-db/render_full_payload_responses.csv")
ARTIFACT_CSV = Path("/Users/anks/.gemini/antigravity-ide/brain/f3ccbc80-47c7-42d9-b649-a9b454bc7c1e/render_full_payload_responses.csv")

print_lock = threading.Lock()


def fetch_one(idx: int, total: int, item: dict, token: str) -> dict:
    platform = item.get("platform", "web")
    url = item["url"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "LinQ-Benchmark-Client/1.0",
    }

    t0 = time.perf_counter()
    status_code = 0
    raw_payload_str = ""
    parsed_json = {}
    error_msg = None

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(RENDER_URL, json={"url": url, "embed": False}, headers=headers)
            status_code = resp.status_code
            raw_payload_str = resp.text
            if status_code == 200:
                parsed_json = resp.json()
            else:
                error_msg = f"HTTP {status_code}"
    except httpx.TimeoutException:
        status_code = 0
        error_msg = "Timeout (>30s)"
        raw_payload_str = json.dumps({"error": "Request timed out (>30s)"})
    except Exception as exc:
        status_code = 0
        error_msg = str(exc)
        raw_payload_str = json.dumps({"error": str(exc)})

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    details = parsed_json.get("details") or []
    tags = parsed_json.get("tags") or []
    extraction = parsed_json.get("extraction") or {}
    word_count = extraction.get("word_count", 0)

    success = (status_code == 200) and (word_count > 0 or len(details) > 0)
    outcome = "PASS" if success else "FAIL"

    with print_lock:
        print(f"[{idx:>3}/{total}] {platform:<14} | {status_code:<4} | {latency_ms:>7.1f}ms | {outcome:<4} | details: {len(details):<2} | tags: {len(tags):<2} | {url[:40]}")

    return {
        "Index": idx,
        "Platform": platform,
        "Target URL": url,
        "Status Code": status_code,
        "Latency (ms)": latency_ms,
        "Outcome": outcome,
        "Full Response Payload JSON": raw_payload_str,
        "Details Count": len(details),
        "Tags Count": len(tags),
        "Extracted Word Count": word_count,
        "Error / Notes": error_msg or ""
    }


def main():
    print("\n" + "=" * 90)
    print(" 🚀 FETCHING COMPLETE FULL PAYLOAD RESPONSES FROM RENDER BACKEND")
    print("=" * 90 + "\n")

    urls = json.loads(URLS_FILE.read_text())
    total = len(urls)
    tokens = get_token_pool(pool_size=5)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(fetch_one, idx, total, item, tokens[(idx - 1) % len(tokens)])
            for idx, item in enumerate(urls, 1)
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda x: x["Index"])

    headers = [
        "Index",
        "Platform",
        "Target URL",
        "Status Code",
        "Latency (ms)",
        "Outcome",
        "Full Response Payload JSON",
        "Details Count",
        "Tags Count",
        "Extracted Word Count",
        "Error / Notes"
    ]

    for p in [OUTPUT_CSV, LOCAL_CSV, LINKS_DB_CSV, ARTIFACT_CSV]:
        os.makedirs(p.parent, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[✓] Exported single complete payload CSV ({len(results)} rows) to:\n    {p}")

    print("\n" + "=" * 90)
    print(" ALL 100 FULL PAYLOAD RESPONSES SUCCESSFULLY SAVED TO SINGLE CSV!")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
