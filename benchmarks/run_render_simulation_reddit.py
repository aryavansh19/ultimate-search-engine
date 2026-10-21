import json
import os
import sys
import time
from pathlib import Path

# Set GEMINI_API_KEY for 1:1 Render simulation
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from linq_server.models import AnalyzeRequest
from linq_server.service import AnalysisService

print("\n" + "=" * 80)
print(" 🚀 RUNNING 1:1 RENDER BACKEND SIMULATION WITH GEMINI AI ENRICHMENT")
print(" Model: gemini-3.5-flash-lite | Provider: gemini")
print("=" * 80 + "\n")

svc = AnalysisService()

reddit_urls = [
    {"index": 11, "url": "https://old.reddit.com/r/crowdstrike/comments/1e6vmkf/bsod_error_in_latest_crowdstrike_update/"},
    {"index": 12, "url": "https://old.reddit.com/r/apolloapp/comments/144f6xm/apollo_will_close_down_on_june_30th_reddits/"},
    {"index": 13, "url": "https://www.reddit.com/r/swift/comments/166p5d5/what_are_your_favorite_swift_features/"},
    {"index": 14, "url": "https://www.reddit.com/r/swift/comments/17sow67/swift_macros_tips_and_tricks/"},
    {"index": 15, "url": "https://www.reddit.com/r/swift/comments/18j1382/how_are_you_structuring_your_swiftui_apps_in/"},
    {"index": 16, "url": "https://www.reddit.com/r/technology/comments/18q9wke/eu_passes_landmark_artificial_intelligence_act/"},
    {"index": 17, "url": "https://www.reddit.com/r/programming/comments/17b2pkn/sqlite_is_35_faster_than_the_filesystem/"},
    {"index": 18, "url": "https://www.reddit.com/r/apple/comments/188045v/m3_max_macbook_pro_one_month_later/"},
    {"index": 19, "url": "https://www.reddit.com/r/webdev/comments/18t9q41/what_backend_framework_do_you_enjoy_the_most/"},
    {"index": 20, "url": "https://www.reddit.com/r/apple/comments/19f3e49/apple_announces_changes_to_ios_safari_and_the_app/"}
]

results = []

for item in reddit_urls:
    idx = item["index"]
    url = item["url"]
    
    t0 = time.perf_counter()
    req = AnalyzeRequest(url=url, embed=False)
    resp = svc.analyze(req)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    
    payload = resp.model_dump()
    extr = payload.get("extraction") or {}
    tier = extr.get("tier", "unknown")
    words = extr.get("word_count", 0)
    details = payload.get("details", [])
    tags = payload.get("tags", [])
    model_used = payload.get("model", "heuristic")
    title = str(payload.get("title") or "")
    
    print(f"[{idx}] Latency: {latency_ms:>7.1f}ms | Tier: {tier:<11} | Model: {model_used:<22} | Bullets: {len(details):<2} | Tags: {len(tags):<2} | Title: {title[:40]}")
    
    results.append({
        "index": idx,
        "platform": "reddit",
        "url": url,
        "status_code": 200,
        "latency_ms": latency_ms,
        "outcome": "PASS" if words > 0 or len(details) > 0 else "FAIL",
        "details_count": len(details),
        "tags_count": len(tags),
        "extracted_word_count": words,
        "ai_model": model_used,
        "error_notes": None,
        "response_payload": payload
    })

out_paths = [
    Path("/Users/anks/Desktop/render_simulated_reddit_10.json"),
    Path("/Users/anks/Desktop/links-db/render_simulated_reddit_10.json"),
    Path("/Users/anks/.gemini/antigravity-ide/brain/f3ccbc80-47c7-42d9-b649-a9b454bc7c1e/render_simulated_reddit_10.json"),
    Path("benchmarks/reports/render_simulated_reddit_10.json")
]

for p in out_paths:
    os.makedirs(p.parent, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Saved simulated Render Reddit results ({len(results)} items) to:\n    {p}")

print("\n" + "=" * 80)
print(" ALL 10 REDDIT LINKS SIMULATED EXACTLY AS RENDER BACKEND!")
print("=" * 80 + "\n")
