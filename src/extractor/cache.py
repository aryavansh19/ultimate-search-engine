"""Envelope cache and attempt telemetry, keyed by canonical URL hash.

Two jobs, both cheap and both load-bearing:

1. Fetch each piece of content exactly once, ever. Extraction is the slowest and
   most expensive stage and the most likely to get you rate-limited; re-running it
   for a link you already have is pure waste.
2. Record every tier attempt, including the ones that failed. After a hundred real
   links, `tier_stats()` tells you which tiers are actually carrying the system --
   which is how you decide where to spend money instead of guessing.

SQLite here is a deliberate prototype choice. The schema maps cleanly onto the
Postgres tables this becomes later.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from .envelope import ContentEnvelope, ExtractionAttempt, ExtractionTier

_SCHEMA = """
CREATE TABLE IF NOT EXISTS envelopes (
    url_hash      TEXT PRIMARY KEY,
    canonical_url TEXT NOT NULL,
    platform      TEXT NOT NULL,
    winning_tier  TEXT NOT NULL,
    degraded      INTEGER NOT NULL DEFAULT 0,
    signal        TEXT NOT NULL DEFAULT 'none',
    envelope_json TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_envelopes_platform ON envelopes(platform);
CREATE INDEX IF NOT EXISTS idx_envelopes_degraded ON envelopes(degraded);

CREATE TABLE IF NOT EXISTS attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url_hash    TEXT NOT NULL,
    platform    TEXT NOT NULL,
    tier        TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    skipped     INTEGER NOT NULL DEFAULT 0,
    reason      TEXT,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attempts_hash ON attempts(url_hash);
CREATE INDEX IF NOT EXISTS idx_attempts_tier ON attempts(tier);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EnvelopeCache:
    """SQLite-backed envelope store. Safe to share across threads."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if self.path.parent and str(self.path.parent) not in {"", "."}:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.commit()

    # ------------------------------------------------------------- envelopes
    def get(self, url_hash: str) -> ContentEnvelope | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT envelope_json FROM envelopes WHERE url_hash = ?",
                (url_hash,),
            ).fetchone()
        if row is None:
            return None
        return ContentEnvelope.model_validate_json(row["envelope_json"])

    def put(self, envelope: ContentEnvelope) -> None:
        payload = envelope.model_dump_json()
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO envelopes (
                    url_hash, canonical_url, platform, winning_tier, degraded,
                    signal, envelope_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url_hash) DO UPDATE SET
                    canonical_url = excluded.canonical_url,
                    platform      = excluded.platform,
                    winning_tier  = excluded.winning_tier,
                    degraded      = excluded.degraded,
                    signal        = excluded.signal,
                    envelope_json = excluded.envelope_json,
                    updated_at    = excluded.updated_at
                """,
                (
                    envelope.url_hash,
                    envelope.canonical_url,
                    envelope.platform.value,
                    envelope.tier.value,
                    int(envelope.degraded),
                    envelope.signal.value,
                    payload,
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def delete(self, url_hash: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM envelopes WHERE url_hash = ?", (url_hash,)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def iter_envelopes(
        self, limit: int = 1000, offset: int = 0, newest_first: bool = True
    ) -> list[ContentEnvelope]:
        """List stored envelopes. The repository call downstream stages should use.

        Exists so later stages do not have to reach into this class's connection to walk
        the library, which is the sort of shortcut that makes moving to Postgres painful.
        """
        order = "DESC" if newest_first else "ASC"
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT envelope_json FROM envelopes
                ORDER BY updated_at {order}
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [ContentEnvelope.model_validate_json(r["envelope_json"]) for r in rows]

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM envelopes").fetchone()
        return int(row["c"] or 0)

    def degraded_items(self, limit: int = 100) -> list[ContentEnvelope]:
        """Items worth re-processing later, once a better tier is available."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT envelope_json FROM envelopes
                WHERE degraded = 1
                ORDER BY updated_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [ContentEnvelope.model_validate_json(r["envelope_json"]) for r in rows]

    # ------------------------------------------------------------- telemetry
    def record_attempts(
        self, url_hash: str, platform: str, attempts: Iterable[ExtractionAttempt]
    ) -> None:
        rows = [
            (
                url_hash,
                platform,
                attempt.tier.value,
                int(attempt.ok),
                int(attempt.skipped),
                attempt.reason,
                attempt.duration_ms,
                attempt.at.isoformat(),
            )
            for attempt in attempts
        ]
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO attempts (
                    url_hash, platform, tier, ok, skipped, reason,
                    duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()

    def tier_stats(self) -> list[dict[str, object]]:
        """Per-tier attempt counts, success rate and median-ish latency.

        Read this before buying a scraper subscription.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tier,
                       COUNT(*)                                   AS attempts,
                       SUM(CASE WHEN ok = 1 THEN 1 ELSE 0 END)    AS wins,
                       SUM(CASE WHEN skipped = 1 THEN 1 ELSE 0 END) AS skips,
                       AVG(CASE WHEN skipped = 0 THEN duration_ms END) AS avg_ms
                FROM attempts
                GROUP BY tier
                """
            ).fetchall()

        order = {tier.value: tier.rank for tier in ExtractionTier}
        stats = []
        for row in rows:
            ran = (row["attempts"] or 0) - (row["skips"] or 0)
            stats.append(
                {
                    "tier": row["tier"],
                    "attempts": row["attempts"] or 0,
                    "ran": ran,
                    "wins": row["wins"] or 0,
                    "skips": row["skips"] or 0,
                    "success_rate": (row["wins"] or 0) / ran if ran else None,
                    "avg_ms": int(row["avg_ms"]) if row["avg_ms"] else None,
                }
            )
        stats.sort(key=lambda s: order.get(str(s["tier"]), 99))
        return stats

    def summary(self) -> dict[str, object]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) AS c FROM envelopes").fetchone()
            degraded = self._conn.execute(
                "SELECT COUNT(*) AS c FROM envelopes WHERE degraded = 1"
            ).fetchone()
            by_platform = self._conn.execute(
                """
                SELECT platform, COUNT(*) AS c FROM envelopes
                GROUP BY platform ORDER BY c DESC
                """
            ).fetchall()
            by_signal = self._conn.execute(
                """
                SELECT signal, COUNT(*) AS c FROM envelopes
                GROUP BY signal ORDER BY c DESC
                """
            ).fetchall()
        return {
            "items": total["c"],
            "degraded": degraded["c"],
            "by_platform": {r["platform"]: r["c"] for r in by_platform},
            "by_signal": {r["signal"]: r["c"] for r in by_signal},
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
