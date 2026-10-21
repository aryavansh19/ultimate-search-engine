# LinQ Search Backend: Empirical Benchmark Ledger

This document tracks all empirical benchmark runs, latency metrics, and extraction success rates across engine versions.

## 1. Version Comparison & Latency Deltas

| Version Tag | Timestamp (UTC) | Items | Success % | Avg Extr (ms) | Avg Enrich (ms) | Avg Embed (ms) | Avg Total (ms) | P95 Total (ms) | Delta vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `v1.0-baseline-475` | 2026-10-20T19:04:09 | 475 | 58.1% | 7696.3ms | 0.0ms | 0.0ms | **7696.4ms** | 46995.7ms | **BASELINE** |
| `v1.1-wikipedia-fix` | 2026-10-21T03:50:49 | 25 | 100.0% | 447.6ms | 0.0ms | 0.0ms | **447.6ms** | 1037.5ms | -7248.8ms (-94.2%) |
| `v1.2-verify` | 2026-10-21T04:08:07 | 25 | 100.0% | 2.8ms | 0.0ms | 0.0ms | **2.8ms** | 12.9ms | -7693.7ms (-100.0%) |
| `v1.3-reddit-fix` | 2026-10-21T04:08:29 | 25 | 0.0% | 0.7ms | 0.0ms | 0.0ms | **0.7ms** | 2.4ms | -7695.8ms (-100.0%) |
| `v1.3-reddit-fix` | 2026-10-21T04:10:25 | 25 | 68.0% | 4374.1ms | 0.0ms | 0.0ms | **4374.6ms** | 5902.0ms | -3321.8ms (-43.2%) |

---

## 2. Latest Run Details (`v1.3-reddit-fix`)

### Per-Platform Performance Breakdown

| Platform | Count | Success % | Avg Extraction (ms) | Primary Extraction Tier |
| :--- | :---: | :---: | :---: | :--- |
| **Reddit** | 25 | 68.0% | 4374.1ms | `open_graph` |

### Upstream Quota Status After Run

```json
{
  "gemini": {
    "service": "gemini",
    "rpm": 0,
    "max_rpm": 14.0,
    "rpm_utilization_pct": 0.0,
    "tpm": 0,
    "max_tpm": 1000000,
    "tpm_utilization_pct": 0.0,
    "rpd": 0,
    "max_rpd": 1500,
    "rpd_utilization_pct": 0.0,
    "status": "healthy",
    "cooldown_remaining_s": 0.0,
    "lifetime_requests": 0,
    "lifetime_tokens": 0,
    "error_counts": {}
  },
  "embedder": {
    "service": "embedder",
    "rpm": 0,
    "max_rpm": 30.0,
    "rpm_utilization_pct": 0.0,
    "tpm": 0,
    "max_tpm": 500000,
    "tpm_utilization_pct": 0.0,
    "rpd": 0,
    "max_rpd": 10000,
    "rpd_utilization_pct": 0.0,
    "status": "healthy",
    "cooldown_remaining_s": 0.0,
    "lifetime_requests": 0,
    "lifetime_tokens": 0,
    "error_counts": {}
  },
  "scraper": {
    "service": "scraper",
    "rpm": 50,
    "max_rpm": 60.0,
    "rpm_utilization_pct": 83.3,
    "tpm": 0,
    "max_tpm": 2000000,
    "tpm_utilization_pct": 0.0,
    "rpd": 100,
    "max_rpd": 20000,
    "rpd_utilization_pct": 0.5,
    "status": "warning",
    "cooldown_remaining_s": 0.0,
    "lifetime_requests": 100,
    "lifetime_tokens": 0,
    "error_counts": {}
  }
}
```

---
*Generated automatically by `benchmarks/run_benchmark.py`.*
