"""Enrichment persistence and cost reporting.

Shares the SQLite file with the extractor cache but owns its own tables and its own
connection. WAL mode makes that safe, and keeping the schemas separate means the two
stages can be split across processes -- or across a client and a server -- without
untangling a shared table first.

`enrichment_tags` is denormalized on purpose. It is what a tag facet and a
tag-filtered search read from, and writing it here is what makes the normalization in
`taxonomy` actually enforced rather than advisory.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .schema import Enrichment

_SCHEMA = """
CREATE TABLE IF NOT EXISTS enrichments (
    url_hash        TEXT PRIMARY KEY,
    source_hash     TEXT NOT NULL,
    mode            TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model           TEXT,
    prompt_version  INTEGER NOT NULL DEFAULT 1,
    category        TEXT NOT NULL DEFAULT 'other',
    content_type    TEXT NOT NULL DEFAULT 'other',
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL NOT NULL DEFAULT 0,
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    degraded        INTEGER NOT NULL DEFAULT 0,
    payload_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_enrich_category ON enrichments(category);
CREATE INDEX IF NOT EXISTS idx_enrich_type     ON enrichments(content_type);
CREATE INDEX IF NOT EXISTS idx_enrich_degraded ON enrichments(degraded);

CREATE TABLE IF NOT EXISTS enrichment_tags (
    url_hash TEXT NOT NULL,
    tag      TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (url_hash, tag)
);

CREATE INDEX IF NOT EXISTS idx_enrich_tags_tag ON enrichment_tags(tag);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EnrichmentStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if self.path.parent and str(self.path.parent) not in {"", "."}:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ read/write
    def get(self, url_hash: str) -> Enrichment | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM enrichments WHERE url_hash = ?", (url_hash,)
            ).fetchone()
        if row is None:
            return None
        return Enrichment.model_validate_json(row["payload_json"])

    def put(self, enrichment: Enrichment) -> None:
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO enrichments (
                    url_hash, source_hash, mode, provider, model, prompt_version,
                    category, content_type, input_tokens, output_tokens, cost_usd,
                    duration_ms, degraded, payload_json, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(url_hash) DO UPDATE SET
                    source_hash    = excluded.source_hash,
                    mode           = excluded.mode,
                    provider       = excluded.provider,
                    model          = excluded.model,
                    prompt_version = excluded.prompt_version,
                    category       = excluded.category,
                    content_type   = excluded.content_type,
                    input_tokens   = excluded.input_tokens,
                    output_tokens  = excluded.output_tokens,
                    cost_usd       = excluded.cost_usd,
                    duration_ms    = excluded.duration_ms,
                    degraded       = excluded.degraded,
                    payload_json   = excluded.payload_json,
                    updated_at     = excluded.updated_at
                """,
                (
                    enrichment.url_hash,
                    enrichment.source_hash,
                    enrichment.mode.value,
                    enrichment.provider,
                    enrichment.model,
                    enrichment.prompt_version,
                    enrichment.category,
                    enrichment.content_type,
                    enrichment.input_tokens,
                    enrichment.output_tokens,
                    enrichment.cost_usd,
                    enrichment.duration_ms,
                    int(enrichment.degraded),
                    enrichment.model_dump_json(),
                    now,
                    now,
                ),
            )
            self._conn.execute(
                "DELETE FROM enrichment_tags WHERE url_hash = ?", (enrichment.url_hash,)
            )
            self._conn.executemany(
                "INSERT OR IGNORE INTO enrichment_tags (url_hash, tag, position) "
                "VALUES (?, ?, ?)",
                [
                    (enrichment.url_hash, tag, index)
                    for index, tag in enumerate(enrichment.tags)
                ],
            )
            self._conn.commit()

    def delete(self, url_hash: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM enrichments WHERE url_hash = ?", (url_hash,)
            )
            self._conn.execute(
                "DELETE FROM enrichment_tags WHERE url_hash = ?", (url_hash,)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def is_current(
        self, url_hash: str, source_hash: str, prompt_version: int
    ) -> bool:
        """Whether a stored enrichment still matches its source content and prompt.

        Re-extraction that changes the envelope, or a prompt revision, both invalidate
        the enrichment. Cheap to check and it prevents silently serving tags derived
        from content that has since been replaced.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT source_hash, prompt_version FROM enrichments WHERE url_hash = ?",
                (url_hash,),
            ).fetchone()
        if row is None:
            return False
        return (
            row["source_hash"] == source_hash
            and int(row["prompt_version"]) == prompt_version
        )

    # ------------------------------------------------------------------- reporting
    def cost_summary(self) -> dict[str, object]:
        with self._lock:
            totals = self._conn.execute(
                """
                SELECT COUNT(*) AS items,
                       COALESCE(SUM(cost_usd), 0)      AS total_cost,
                       COALESCE(SUM(input_tokens), 0)  AS input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens
                FROM enrichments
                """
            ).fetchone()
            by_mode = self._conn.execute(
                """
                SELECT mode,
                       COUNT(*) AS items,
                       COALESCE(SUM(cost_usd), 0) AS cost,
                       COALESCE(AVG(duration_ms), 0) AS avg_ms
                FROM enrichments GROUP BY mode ORDER BY cost DESC
                """
            ).fetchall()
            by_model = self._conn.execute(
                """
                SELECT COALESCE(model, 'none') AS model,
                       COUNT(*) AS items,
                       COALESCE(SUM(cost_usd), 0) AS cost
                FROM enrichments GROUP BY model ORDER BY cost DESC
                """
            ).fetchall()

        items = int(totals["items"] or 0)
        total_cost = float(totals["total_cost"] or 0.0)
        return {
            "items": items,
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_usd": round(total_cost / items, 6) if items else 0.0,
            "input_tokens": int(totals["input_tokens"] or 0),
            "output_tokens": int(totals["output_tokens"] or 0),
            "by_mode": [
                {
                    "mode": row["mode"],
                    "items": row["items"],
                    "cost_usd": round(float(row["cost"]), 6),
                    "avg_ms": int(row["avg_ms"]),
                }
                for row in by_mode
            ],
            "by_model": [
                {
                    "model": row["model"],
                    "items": row["items"],
                    "cost_usd": round(float(row["cost"]), 6),
                }
                for row in by_model
            ],
        }

    def degraded_hashes(self, limit: int = 500) -> set[str]:
        """Items tagged by a fallback path, eligible for an upgrade.

        A degraded enrichment is *current* in the sense that it matches its source
        content, so a plain freshness check considers it done. It still deserves
        re-running once a real provider is reachable -- otherwise everything tagged
        while the API key was missing stays permanently worse than it needs to be.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT url_hash FROM enrichments WHERE degraded = 1 LIMIT ?",
                (limit,),
            ).fetchall()
        return {row["url_hash"] for row in rows}

    def top_tags(self, limit: int = 25) -> list[tuple[str, int]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tag, COUNT(*) AS uses FROM enrichment_tags
                GROUP BY tag ORDER BY uses DESC, tag ASC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [(row["tag"], row["uses"]) for row in rows]

    def find_by_tag(self, tag: str, limit: int = 50) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT url_hash FROM enrichment_tags WHERE tag = ? LIMIT ?",
                (tag, limit),
            ).fetchall()
        return [row["url_hash"] for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
