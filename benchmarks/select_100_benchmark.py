import json
from collections import defaultdict
from pathlib import Path

corpus_file = Path("benchmarks/corpus/corpus-v3.0.json")
with open(corpus_file) as f:
    data = json.load(f)

by_platform = defaultdict(list)
for item in data:
    p = item.get("platform", "web")
    by_platform[p].append(item["normalized_url"])

# Target 10 platforms with 10 links each = exactly 100 links
target_platforms = [
    ("wikipedia", 10),
    ("reddit", 10),
    ("medium", 10),
    ("youtube", 10),
    ("instagram", 10),
    ("github", 10),
    ("developer_docs", 10),
    ("arxiv", 10),
    ("substack", 10),
    ("saas_product", 10)
]

selected = []
for plat, count in target_platforms:
    available = by_platform.get(plat, [])
    picked = available[:count]
    for url in picked:
        selected.append({"platform": plat, "url": url})

out_file = Path("benchmarks/corpus/benchmark_100.json")
with open(out_file, "w") as f:
    json.dump(selected, f, indent=2)

print(f"Generated {len(selected)} benchmark links in {out_file}:")
for plat, count in target_platforms:
    print(f"  {plat:<15}: {count}")
