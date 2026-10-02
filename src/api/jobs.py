"""Background job tracking for link ingestion.

Ingestion is slow and multi-stage: extraction can take seconds, the tagging call takes a
few more, and video understanding took 37 seconds on a real reel. None of that can happen
inside a request without the browser hanging, so `POST /api/items` records a job, returns
immediately, and a worker thread advances it.

Jobs live in SQLite rather than memory so a restart mid-ingest leaves a visible failed job
instead of a link that silently vanished. That mattered from the first design sketch: a
user who shares something and sees nothing appear has lost data, as far as they are
concerned.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT NOT NULL,
    url_hash    TEXT,
    status      TEXT NOT NULL,
    stage       TEXT,
    error       TEXT,
    payload     TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON api_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_hash   ON api_jobs(url_hash);
"""


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"


class JobStage(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    TAGGING = "tagging"
    INDEXING = "indexing"
    DONE = "done"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Job:
    id: int
    url: str
    url_hash: str | None
    status: JobStatus
    stage: JobStage
    error: str | None
    created_at: str
    updated_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "url_hash": self.url_hash,
            "status": self.status.value,
            "stage": self.stage.value,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobStore:
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
            # Anything left RUNNING is from a process that died mid-job. Surface it as
            # failed rather than leaving a spinner that never resolves.
            self._conn.execute(
                "UPDATE api_jobs SET status = ?, error = ?, updated_at = ? "
                "WHERE status IN (?, ?)",
                (
                    JobStatus.FAILED.value,
                    "interrupted by server restart",
                    _now(),
                    JobStatus.RUNNING.value,
                    JobStatus.QUEUED.value,
                ),
            )
            self._conn.commit()

    def create(self, url: str) -> Job:
        now = _now()
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO api_jobs (url, status, stage, created_at, updated_at)
                VALUES (?,?,?,?,?)
                """,
                (url, JobStatus.QUEUED.value, JobStage.QUEUED.value, now, now),
            )
            self._conn.commit()
            job_id = int(cursor.lastrowid or 0)
        return Job(
            id=job_id,
            url=url,
            url_hash=None,
            status=JobStatus.QUEUED,
            stage=JobStage.QUEUED,
            error=None,
            created_at=now,
            updated_at=now,
        )

    def update(
        self,
        job_id: int,
        *,
        status: JobStatus | None = None,
        stage: JobStage | None = None,
        url_hash: str | None = None,
        error: str | None = None,
    ) -> None:
        sets: list[str] = ["updated_at = ?"]
        params: list[Any] = [_now()]
        if status is not None:
            sets.append("status = ?")
            params.append(status.value)
        if stage is not None:
            sets.append("stage = ?")
            params.append(stage.value)
        if url_hash is not None:
            sets.append("url_hash = ?")
            params.append(url_hash)
        if error is not None:
            sets.append("error = ?")
            params.append(error)
        params.append(job_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE api_jobs SET {', '.join(sets)} WHERE id = ?", params
            )
            self._conn.commit()

    def get(self, job_id: int) -> Job | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM api_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return _to_job(row) if row else None

    def active(self) -> list[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM api_jobs WHERE status IN (?, ?) ORDER BY id ASC",
                (JobStatus.QUEUED.value, JobStatus.RUNNING.value),
            ).fetchall()
        return [_to_job(row) for row in rows]

    def recent(self, limit: int = 20) -> list[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM api_jobs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_to_job(row) for row in rows]

    def recent_failures(self, limit: int = 10) -> list[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM api_jobs WHERE status = ? ORDER BY id DESC LIMIT ?",
                (JobStatus.FAILED.value, limit),
            ).fetchall()
        return [_to_job(row) for row in rows]

    def clear_finished(self) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM api_jobs WHERE status IN (?, ?)",
                (JobStatus.READY.value, JobStatus.FAILED.value),
            )
            self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=int(row["id"]),
        url=row["url"],
        url_hash=row["url_hash"],
        status=JobStatus(row["status"]),
        stage=JobStage(row["stage"] or JobStage.QUEUED.value),
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
