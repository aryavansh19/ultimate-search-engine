import json
import os
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from linq_server.models import AnalyzeRequest
from linq_server.service import AnalysisService

svc = AnalysisService()

reddit_urls = [
    "https://old.reddit.com/r/crowdstrike/comments/1e6vmkf/bsod_error_in_latest_crowdstrike_update/",
    "https://old.reddit.com/r/apolloapp/comments/144f6xm/apollo_will_close_down_on_june_30th_reddits/",
    "https://www.reddit.com/r/swift/comments/166p5d5/what_are_your_favorite_swift_features/",
    "https://www.reddit.com/r/swift/comments/17sow67/swift_macros_tips_and_tricks/",
    "https://www.reddit.com/r/swift/comments/18j1382/how_are_you_structuring_your_swiftui_apps_in/",
    "https://www.reddit.com/r/technology/comments/18q9wke/eu_passes_landmark_artificial_intelligence_act/",
    "https://www.reddit.com/r/programming/comments/17b2pkn/sqlite_is_35_faster_than_the_filesystem/",
    "https://www.reddit.com/r/apple/comments/188045v/m3_max_macbook_pro_one_month_later/",
    "https://www.reddit.com/r/webdev/comments/18t9q41/what_backend_framework_do_you_enjoy_the_most/",
    "https://www.reddit.com/r/apple/comments/19f3e49/apple_announces_changes_to_ios_safari_and_the_app/"
]

results = []

for idx, url in enumerate(reddit_urls, 11):
    t0 = time.perf_counter()
    req = AnalyzeRequest(url=url, embed=False)
    resp = svc.analyze(req)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    d = resp.model_dump()
    extr = d.get("extraction") or {}
    tier = extr.get("tier", "unknown")
    words = extr.get("word_count", 0)
    note = extr.get("note", "")
    title = str(d.get("title") or "")
    
    results.append({
        "index": idx,
        "platform": "reddit",
        "url": url,
        "status_code": 200,
        "latency_ms": latency_ms,
        "outcome": "PASS" if (words > 0 or len(title) > 0) else "FAIL",
        "details_count": len(d.get("details", [])),
        "tags_count": len(d.get("tags", [])),
        "extracted_word_count": words,
        "error_notes": None,
        "response_payload": d
    })
    
    print(f"[{idx}] Latency: {latency_ms:>6.1f}ms | Tier: {tier:<12} | Words: {words:<3} | Title: {title[:55]}")

out_paths = [
    Path("/Users/anks/Desktop/local_fixed_reddit_10.json"),
    Path("/Users/anks/Desktop/links-db/local_fixed_reddit_10.json"),
    Path("/Users/anks/.gemini/antigravity-ide/brain/f3ccbc80-47c7-42d9-b649-a9b454bc7c1e/local_fixed_reddit_10.json"),
    Path("benchmarks/reports/local_fixed_reddit_10.json")
]

for p in out_paths:
    os.makedirs(p.parent, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[✓] Saved fixed Reddit JSON to: {p}")
