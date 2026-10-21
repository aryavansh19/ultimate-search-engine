import csv
import json
import os
from pathlib import Path

csv_path = Path("/Users/anks/Desktop/render_full_payload_responses.csv")

records = []
with open(csv_path, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw_json_str = row["Full Response Payload JSON"]
        try:
            parsed_payload = json.loads(raw_json_str)
        except Exception:
            parsed_payload = raw_json_str

        record = {
            "index": int(row["Index"]),
            "platform": row["Platform"],
            "url": row["Target URL"],
            "status_code": int(row["Status Code"]),
            "latency_ms": float(row["Latency (ms)"]),
            "outcome": row["Outcome"],
            "details_count": int(row["Details Count"]),
            "tags_count": int(row["Tags Count"]),
            "extracted_word_count": int(row["Extracted Word Count"]),
            "error_notes": row["Error / Notes"] if row["Error / Notes"] else None,
            "response_payload": parsed_payload
        }
        records.append(record)

out_paths = [
    Path("/Users/anks/Desktop/render_full_payload_responses.json"),
    Path("/Users/anks/Desktop/links-db/render_full_payload_responses.json"),
    Path("/Users/anks/.gemini/antigravity-ide/brain/f3ccbc80-47c7-42d9-b649-a9b454bc7c1e/render_full_payload_responses.json"),
    Path("benchmarks/reports/render_full_payload_responses.json")
]

for p in out_paths:
    os.makedirs(p.parent, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(records)} JSON items to: {p}")

print("SUCCESS: JSON files generated!")
