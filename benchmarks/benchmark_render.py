#!/usr/bin/env python3
"""LinQ Live Render Benchmark Suite

Runs a 100-link benchmark suite against the live Render backend
(https://linq-enrichment.onrender.com/v1/analyze), recording response time,
extraction tier, AI enrichment mode, and pass/fail metrics.

Rotates across a pool of benchmark test users to stay within Render's
per-user hourly budget (LINQ_ANALYSES_PER_HOUR=60).
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx

RENDER_URL = "https://linq-enrichment.onrender.com/v1/analyze"
SUPABASE_AUTH_URL = "https://btkfngukyxpqijlxrbhm.supabase.co/auth/v1"
SUPABASE_ANON_KEY = "sb_publishable_UjQoaKVibO4Owu3yfL7xkQ_nb9TaKFP"
TOKEN_CACHE_FILE = Path(__file__).resolve().parent.parent / "logs" / ".supabase_pool_cache.json"

print_lock = threading.Lock()


def get_token_pool(pool_size: int = 5) -> List[str]:
    """Get or create authenticated tokens for a pool of benchmark users."""
    now = time.time()
    TOKEN_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if TOKEN_CACHE_FILE.exists():
        try:
            cached = json.loads(TOKEN_CACHE_FILE.read_text())
            if cached.get("expires_at", 0) > now + 600 and len(cached.get("tokens", [])) >= pool_size:
                return cached["tokens"][:pool_size]
        except Exception:
            pass

    tokens = []
    headers = {"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"}
    with httpx.Client(timeout=15.0) as client:
        for i in range(1, pool_size + 1):
            email = f"linq_bench_user{i}@linq.app"
            payload = {"email": email, "password": "Password123!@#"}
            r = client.post(f"{SUPABASE_AUTH_URL}/token?grant_type=password", json=payload, headers=headers)
            if r.status_code != 200:
                r = client.post(f"{SUPABASE_AUTH_URL}/signup", json=payload, headers=headers)
            if r.status_code in (200, 201):
                tokens.append(r.json()["access_token"])
            else:
                raise RuntimeError(f"Failed to auth benchmark user {email}: HTTP {r.status_code}")

    TOKEN_CACHE_FILE.write_text(json.dumps({"tokens": tokens, "expires_at": now + 3600}))
    return tokens


def test_single_url(idx: int, total: int, item: Dict[str, str], token: str) -> Dict[str, Any]:
    platform = item.get("platform", "web")
    url = item["url"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "LinQ-Benchmark-Client/1.0"
    }

    t0 = time.perf_counter()
    status_code = 0
    error_msg = None
    data = {}

    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(RENDER_URL, json={"url": url, "embed": False}, headers=headers)
            status_code = resp.status_code
            if resp.status_code == 200:
                data = resp.json()
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:80]}"
    except httpx.TimeoutException:
        error_msg = "Timeout (>45s)"
    except Exception as exc:
        error_msg = str(exc)[:80]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    extraction = data.get("extraction") or {}
    tier = extraction.get("tier", "failed" if error_msg else "unknown")
    word_count = extraction.get("word_count", 0)
    enrichment = data.get("enrichment") or {}
    ai_mode = enrichment.get("mode", "none")
    cost = enrichment.get("cost_estimate", 0.0)
    title = data.get("title", "")
    details_count = len(data.get("details", []))

    success = (status_code == 200) and (word_count > 0 or details_count > 0)
    outcome = "PASS" if success else "FAIL"

    with print_lock:
        print(f"[{idx:>3}/{total}] | {platform:<14} | {status_code:<5} | {latency_ms:>7.1f}ms | {tier:<14} | {outcome:<4} | {url[:45]}")

    return {
        "index": idx,
        "platform": platform,
        "url": url,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "success": success,
        "tier": tier,
        "word_count": word_count,
        "ai_mode": ai_mode,
        "cost_estimate": cost,
        "title": title,
        "details_count": details_count,
        "error": error_msg
    }


def run_benchmark(urls: List[Dict[str, str]], output_path: Path, concurrency: int = 2) -> Dict[str, Any]:
    print("\n" + "=" * 95)
    print(" 🚀 RUNNING LIVE RENDER 100-LINK BENCHMARK (MULTI-USER ROTATION POOL)")
    print(f"[*] Target Endpoint:  {RENDER_URL}")
    print(f"[*] Total Test URLs:  {len(urls)}")
    print(f"[*] Concurrency:      {concurrency} worker(s)")
    print(f"[*] Started At:       {datetime.now(timezone.utc).isoformat()}")
    print("=" * 95 + "\n")

    token_pool = get_token_pool(pool_size=5)
    print(f"[✓] Initialized {len(token_pool)} benchmark user tokens (avoids 60 req/hr single-user rate cap)\n")

    total = len(urls)
    results = []

    print(f"{'Progress':<9} | {'Platform':<14} | {'Code':<5} | {'Latency':<9} | {'Tier':<14} | {'Stat':<4} | {'Target URL'}")
    print("-" * 95)

    if concurrency > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(test_single_url, idx, total, item, token_pool[(idx - 1) % len(token_pool)])
                for idx, item in enumerate(urls, 1)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
    else:
        for idx, item in enumerate(urls, 1):
            results.append(test_single_url(idx, total, item, token_pool[(idx - 1) % len(token_pool)]))

    results.sort(key=lambda r: r["index"])

    total_tested = len(results)
    passed = sum(1 for r in results if r["success"])
    pass_rate = round((passed / total_tested) * 100, 1) if total_tested else 0.0
    latencies = [r["latency_ms"] for r in results]
    mean_lat = round(statistics.mean(latencies), 1) if latencies else 0.0
    p50_lat = round(statistics.median(latencies), 1) if latencies else 0.0
    p95_lat = round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0, 1)

    plat_stats = defaultdict(lambda: {"total": 0, "passed": 0, "latencies": []})
    for r in results:
        p = r["platform"]
        plat_stats[p]["total"] += 1
        if r["success"]:
            plat_stats[p]["passed"] += 1
        plat_stats[p]["latencies"].append(r["latency_ms"])

    print("\n" + "=" * 95)
    print(" 📊 100-LINK BENCHMARK OVERALL SUMMARY")
    print(f"  • Total URLs Tested: {total_tested}")
    print(f"  • Passed:            {passed}/{total_tested} ({pass_rate}%)")
    print(f"  • Median Latency:    {p50_lat} ms")
    print(f"  • Mean Latency:      {mean_lat} ms")
    print(f"  • P95 Latency:       {p95_lat} ms")
    print("-" * 95)
    print(" Per-Platform Breakdown:")
    print(f"  {'Platform':<15} | {'Passed':<12} | {'Pass Rate':<10} | {'Median Latency':<15} | {'Mean Latency'}")
    print("  " + "-" * 75)

    breakdown_data = {}
    for p, s in sorted(plat_stats.items()):
        p_total = s["total"]
        p_pass = s["passed"]
        p_rate = round((p_pass / p_total) * 100, 1) if p_total else 0.0
        p_med = round(statistics.median(s["latencies"]), 1) if s["latencies"] else 0.0
        p_mean = round(statistics.mean(s["latencies"]), 1) if s["latencies"] else 0.0
        breakdown_data[p] = {
            "total": p_total,
            "passed": p_pass,
            "pass_rate_pct": p_rate,
            "median_latency_ms": p_med,
            "mean_latency_ms": p_mean
        }
        print(f"  {p:<15} | {p_pass}/{p_total:<10} | {p_rate:>5.1f}%    | {p_med:>7.1f} ms      | {p_mean:>7.1f} ms")
    print("=" * 95 + "\n")

    summary_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": RENDER_URL,
        "total": total_tested,
        "passed": passed,
        "pass_rate_pct": pass_rate,
        "p50_latency_ms": p50_lat,
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "platform_breakdown": breakdown_data,
        "results": results
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"[✓] Benchmark results saved to: {output_path}")
    return summary_payload


def compare_runs(before_file: Path, after_file: Path) -> None:
    before = json.loads(before_file.read_text())
    after = json.loads(after_file.read_text())

    print("\n" + "=" * 95)
    print(" ⚖️  100-LINK BEFORE VS AFTER BENCHMARK COMPARISON")
    print("=" * 95)
    print(f"{'Metric':<25} | {'Before (Baseline)':<22} | {'After (Optimized)':<22} | {'Delta'}")
    print("-" * 95)

    b_pass = f"{before['passed']}/{before['total']} ({before['pass_rate_pct']}%)"
    a_pass = f"{after['passed']}/{after['total']} ({after['pass_rate_pct']}%)"
    delta_pass = f"{after['pass_rate_pct'] - before['pass_rate_pct']:+.1f}%"
    print(f"{'Pass Rate':<25} | {b_pass:<22} | {a_pass:<22} | {delta_pass}")

    b_p50 = f"{before['p50_latency_ms']} ms"
    a_p50 = f"{after['p50_latency_ms']} ms"
    delta_p50 = f"{after['p50_latency_ms'] - before['p50_latency_ms']:+.1f} ms"
    print(f"{'P50 Median Latency':<25} | {b_p50:<22} | {a_p50:<22} | {delta_p50}")

    b_mean = f"{before['mean_latency_ms']} ms"
    a_mean = f"{after['mean_latency_ms']} ms"
    delta_mean = f"{after['mean_latency_ms'] - before['mean_latency_ms']:+.1f} ms"
    print(f"{'Mean Latency':<25} | {b_mean:<22} | {a_mean:<22} | {delta_mean}")

    print("-" * 95)
    print("\nPer-Platform Comparison:")
    print(f"{'Platform':<15} | {'Before Pass':<14} | {'After Pass':<14} | {'Before Latency':<16} | {'After Latency'}")
    print("-" * 95)

    b_plat = before.get("platform_breakdown", {})
    a_plat = after.get("platform_breakdown", {})
    all_keys = sorted(set(b_plat.keys()) | set(a_plat.keys()))
    for k in all_keys:
        bp = b_plat.get(k, {})
        ap = a_plat.get(k, {})
        b_p_str = f"{bp.get('passed', 0)}/{bp.get('total', 0)} ({bp.get('pass_rate_pct', 0)}%)"
        a_p_str = f"{ap.get('passed', 0)}/{ap.get('total', 0)} ({ap.get('pass_rate_pct', 0)}%)"
        b_l_str = f"{bp.get('median_latency_ms', 0)} ms"
        a_l_str = f"{ap.get('median_latency_ms', 0)} ms"
        print(f"{k:<15} | {b_p_str:<14} | {a_p_str:<14} | {b_l_str:<16} | {a_l_str}")
    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Live Render Benchmark Runner")
    parser.add_argument("--urls-file", type=Path, help="JSON file containing URLs to test")
    parser.add_argument("--concurrency", type=int, default=2, help="Worker concurrency")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/reports/render_benchmark.json"), help="Output JSON path")
    parser.add_argument("--compare", nargs=2, type=Path, help="Compare two benchmark runs (before after)")
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
    else:
        if args.urls_file and args.urls_file.exists():
            urls = json.loads(args.urls_file.read_text())
        else:
            urls = []
        run_benchmark(urls, args.output, concurrency=args.concurrency)


if __name__ == "__main__":
    main()
