#!/usr/bin/env python3
"""LinQ Render Live Caller & Response Logger

Test any link against the live Render deployment (https://linq-enrichment.onrender.com),
record precise millisecond response timings, and log the full response payload.

Usage:
    .venv/bin/python call_render.py "<URL>"
    .venv/bin/python call_render.py "https://www.instagram.com/reel/DePWnYvJEVL/"
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Configuration
RENDER_URL = "https://linq-enrichment.onrender.com/v1/analyze"
SUPABASE_AUTH_URL = "https://btkfngukyxpqijlxrbhm.supabase.co/auth/v1"
SUPABASE_ANON_KEY = "sb_publishable_UjQoaKVibO4Owu3yfL7xkQ_nb9TaKFP"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "render_call_history.jsonl"
TOKEN_CACHE_FILE = LOG_DIR / ".supabase_token_cache.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_auth_token() -> str:
    """Fetch or reuse a cached Supabase auth token."""
    now = time.time()
    if TOKEN_CACHE_FILE.exists():
        try:
            cached = json.loads(TOKEN_CACHE_FILE.read_text())
            if cached.get("expires_at", 0) > now + 300:
                return cached["access_token"]
        except Exception:
            pass

    # Signup / login a persistent benchmark test user
    payload = {"email": "linq_cli_tester@linq.app", "password": "Password123!@#"}
    headers = {"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"}

    with httpx.Client(timeout=15.0) as client:
        # Try sign in first
        r = client.post(f"{SUPABASE_AUTH_URL}/token?grant_type=password", json=payload, headers=headers)
        if r.status_code != 200:
            # If user does not exist, signup
            r = client.post(f"{SUPABASE_AUTH_URL}/signup", json=payload, headers=headers)

    if r.status_code not in (200, 201):
        raise RuntimeError(f"Failed to authenticate with Supabase: HTTP {r.status_code} {r.text}")

    data = r.json()
    token = data["access_token"]
    expires_at = data.get("expires_at", now + 3600)

    TOKEN_CACHE_FILE.write_text(json.dumps({"access_token": token, "expires_at": expires_at}))
    return token


def main() -> None:
    if len(sys.argv) < 2:
        print("\nUsage:")
        print('  .venv/bin/python call_render.py "<URL>"\n')
        print("Examples:")
        print('  .venv/bin/python call_render.py "https://www.instagram.com/reel/DePWnYvJEVL/"')
        print('  .venv/bin/python call_render.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"')
        print('  .venv/bin/python call_render.py "https://en.wikipedia.org/wiki/Artificial_intelligence"\n')
        sys.exit(1)

    url = sys.argv[1].strip()
    embed = "--embed" in sys.argv

    print("\n" + "=" * 78)
    print(" 🚀 LINQ RENDER LIVE CALLER")
    print("=" * 78)
    print(f"[*] Target URL:      {url}")
    print(f"[*] Render Endpoint: {RENDER_URL}")
    print(f"[*] Timestamp:       {datetime.now(timezone.utc).isoformat()}")

    # 1. Authenticate
    try:
        token = get_auth_token()
        print("[✓] Supabase Auth:   Authenticated (Bearer Token Active)")
    except Exception as exc:
        print(f"[!] Auth Warning:   Could not get Supabase token ({exc}). Trying anonymous...")
        token = ""

    # 2. Build Request
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "LinQ-CLI-Tester/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    body = {"url": url, "embed": embed}

    print("\n[*] Sending request to Render backend... (waiting for response)")
    start_time = time.perf_counter()

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(RENDER_URL, json=body, headers=headers)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        status_code = resp.status_code
    except Exception as exc:
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        print(f"\n[!] Network Error: {exc} (after {latency_ms:.2f} ms)")
        sys.exit(1)

    # 3. Parse Response
    try:
        data = resp.json()
    except Exception:
        data = {"raw_text": resp.text}

    # 4. Display Metrics
    print("\n" + "-" * 78)
    print(f" ⏱️  RESPONSE METRICS & LATENCY")
    print("-" * 78)
    status_label = "200 OK" if status_code == 200 else f"HTTP {status_code}"
    print(f"  • Total Latency:    {latency_ms:.2f} ms  ({latency_ms / 1000.0:.2f} seconds)")
    print(f"  • HTTP Status:      {status_label}")

    if status_code == 200:
        extr = data.get("extraction") or {}
        print(f"  • Extraction Tier:  {extr.get('tier')} ({extr.get('note')})")
        print(f"  • AI Mode / Model:  {data.get('mode')} ({data.get('model')})")
        print(f"  • Category / Type:  {data.get('category')} | {data.get('content_type')}")
        print(f"  • Cost Estimate:    ${data.get('cost_usd', 0.0):.6f}")
        print(f"  • Title:            {data.get('title')}")
        print(f"  • Summary:          {data.get('summary')}")
        print(f"  • Details Count:    {len(data.get('details', []))} key insights generated")
        print(f"  • Tags ({len(data.get('tags', []))}):        {', '.join(data.get('tags', [])[:8])}")
        print(f"  • Entities:         {data.get('entities')}")
        if data.get("details"):
            print("\n  🔍 Generated Key Details (Bullet Points):")
            for idx, bullet in enumerate(data.get("details", []), start=1):
                print(f"     {idx}. {bullet}")

    # 5. Save to Log File
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "latency_ms": round(latency_ms, 2),
        "status_code": status_code,
        "response": data,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    print("\n" + "-" * 78)
    print(f"[+] Full response logged to: {LOG_FILE}")
    print("=" * 78)

    # 6. Pretty Print Full JSON Payload
    print("\n[*] RAW RESPONSE PAYLOAD (JSON):")
    print(json.dumps(data, indent=2))
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
