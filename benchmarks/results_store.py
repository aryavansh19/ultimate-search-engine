"""Benchmark Results Store & Markdown Ledger Generator.

Manages persistent benchmark run logs in JSON and automatically updates
BENCHMARK_RESULTS.md with before-and-after comparison tables, latency deltas,
and platform success rates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ItemBenchmarkResult:
    item_id: str
    url: str
    platform: str
    genre: str
    success: bool
    tier: str
    extraction_ms: float
    enrichment_ms: float
    embedding_ms: float
    total_ms: float
    title: str | None = None
    author: str | None = None
    media_kind: str | None = None
    tag_count: int = 0
    detail_count: int = 0
    token_count: int = 0
    error: str | None = None


@dataclass(slots=True)
class BenchmarkSummary:
    tag: str
    timestamp: str
    total_items: int
    successful_items: int
    success_rate_pct: float
    avg_extraction_ms: float
    avg_enrichment_ms: float
    avg_embedding_ms: float
    avg_total_ms: float
    p95_total_ms: float
    total_tokens: int
    quota_telemetry: dict[str, Any]
    items: list[ItemBenchmarkResult]


class BenchmarkLedger:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_file = self.base_dir / "BENCHMARK_RESULTS.md"

    def save_run(self, summary: BenchmarkSummary) -> Path:
        filename = f"benchmark_{summary.tag}_{int(datetime.now(timezone.utc).timestamp())}.json"
        target = self.results_dir / filename
        with open(target, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)
        self.update_markdown_ledger()
        return target

    def load_all_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for p in sorted(self.results_dir.glob("benchmark_*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    runs.append(json.load(f))
            except Exception:
                continue
        return runs

    def update_markdown_ledger(self) -> None:
        runs = self.load_all_runs()
        if not runs:
            return

        lines: list[str] = [
            "# LinQ Search Backend: Empirical Benchmark Ledger",
            "",
            "This document tracks all empirical benchmark runs, latency metrics, and extraction success rates across engine versions.",
            "",
            "## 1. Version Comparison & Latency Deltas",
            "",
            "| Version Tag | Timestamp (UTC) | Items | Success % | Avg Extr (ms) | Avg Enrich (ms) | Avg Embed (ms) | Avg Total (ms) | P95 Total (ms) | Delta vs Baseline |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        baseline_total_ms: float | None = None

        for idx, run in enumerate(runs):
            tag = run.get("tag", "unknown")
            ts = run.get("timestamp", "")[:19]
            total_items = run.get("total_items", 0)
            success_pct = run.get("success_rate_pct", 0.0)
            avg_extr = run.get("avg_extraction_ms", 0.0)
            avg_enr = run.get("avg_enrichment_ms", 0.0)
            avg_emb = run.get("avg_embedding_ms", 0.0)
            avg_tot = run.get("avg_total_ms", 0.0)
            p95_tot = run.get("p95_total_ms", 0.0)

            if idx == 0:
                baseline_total_ms = avg_tot
                delta_str = "**BASELINE**"
            elif baseline_total_ms and baseline_total_ms > 0:
                diff = avg_tot - baseline_total_ms
                pct = (diff / baseline_total_ms) * 100
                sign = "+" if diff > 0 else ""
                delta_str = f"{sign}{diff:.1f}ms ({sign}{pct:.1f}%)"
            else:
                delta_str = "-"

            lines.append(
                f"| `{tag}` | {ts} | {total_items} | {success_pct:.1f}% | "
                f"{avg_extr:.1f}ms | {avg_enr:.1f}ms | {avg_emb:.1f}ms | **{avg_tot:.1f}ms** | {p95_tot:.1f}ms | {delta_str} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 2. Latest Run Details (`" + str(runs[-1].get("tag", "")) + "`)",
            "",
            "### Per-Platform Performance Breakdown",
            "",
            "| Platform | Count | Success % | Avg Extraction (ms) | Primary Extraction Tier |",
            "| :--- | :---: | :---: | :---: | :--- |",
        ])

        latest = runs[-1]
        items = latest.get("items", [])
        platform_stats: dict[str, dict[str, Any]] = {}

        for item in items:
            plat = item.get("platform", "other")
            if plat not in platform_stats:
                platform_stats[plat] = {"count": 0, "success": 0, "extr_ms": [], "tiers": {}}
            platform_stats[plat]["count"] += 1
            if item.get("success"):
                platform_stats[plat]["success"] += 1
            platform_stats[plat]["extr_ms"].append(item.get("extraction_ms", 0.0))
            tier = item.get("tier", "none")
            platform_stats[plat]["tiers"][tier] = platform_stats[plat]["tiers"].get(tier, 0) + 1

        for plat, stats in sorted(platform_stats.items()):
            cnt = stats["count"]
            succ_pct = (stats["success"] / cnt * 100) if cnt else 0.0
            avg_ms = sum(stats["extr_ms"]) / len(stats["extr_ms"]) if stats["extr_ms"] else 0.0
            primary_tier = max(stats["tiers"].items(), key=lambda x: x[1])[0] if stats["tiers"] else "n/a"
            lines.append(f"| **{plat.title()}** | {cnt} | {succ_pct:.1f}% | {avg_ms:.1f}ms | `{primary_tier}` |")

        lines.extend([
            "",
            "### Upstream Quota Status After Run",
            "",
            "```json",
            json.dumps(latest.get("quota_telemetry", {}), indent=2),
            "```",
            "",
            "---",
            "*Generated automatically by `benchmarks/run_benchmark.py`.*",
        ])

        with open(self.ledger_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
