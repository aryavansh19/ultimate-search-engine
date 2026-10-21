import csv
import json
import os
from pathlib import Path

json_file = Path("benchmarks/reports/render_baseline_100_before.json")
with open(json_file) as f:
    data = json.load(f)

results = data["results"]

headers = [
    "Index",
    "Platform",
    "Status Code",
    "Latency (ms)",
    "Outcome",
    "Extraction Tier",
    "Word Count",
    "Details Count",
    "AI Mode",
    "Cost Estimate ($)",
    "Title",
    "Target URL",
    "Error / Notes"
]

rows = []
for r in results:
    rows.append([
        r.get("index"),
        r.get("platform"),
        r.get("status_code"),
        r.get("latency_ms"),
        "PASS" if r.get("success") else "FAIL",
        r.get("tier"),
        r.get("word_count"),
        r.get("details_count"),
        r.get("ai_mode"),
        r.get("cost_estimate"),
        (r.get("title") or "").replace("\n", " ").strip(),
        r.get("url"),
        r.get("error") or ""
    ])

out_paths = [
    Path("benchmarks/reports/render_baseline_100_payload.csv"),
    Path("/Users/anks/Desktop/render_baseline_100_payload.csv"),
    Path("/Users/anks/Desktop/links-db/render_baseline_100_payload.csv"),
    Path("/Users/anks/.gemini/antigravity-ide/brain/f3ccbc80-47c7-42d9-b649-a9b454bc7c1e/render_baseline_100_payload.csv")
]

for p in out_paths:
    os.makedirs(p.parent, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to: {p}")

print("SUCCESS: CSV exported to all target destinations!")
