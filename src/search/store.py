"""Search index storage: FTS5 for keyword, float32 blobs for vectors.

Shares the SQLite file with the other stages, own tables and own connection.

Two decisions worth stating.

**FTS5 with per-column weights.** Keyword search exists to nail exact tokens -- a
handle, a version number, a library name -- which is precisely where embeddings smear.
Splitting the searchable text into columns lets a title or tag match outrank the same
word appearing incidentally in a transcript. `bm25()` returns *negative* scores where
more negative is better, which is a reliable source of inverted rankings if you forget.

**Brute-force vector scan.** Every vector is loaded into one contiguous numpy matrix and
scored with a single matrix-vector product. At personal-library scale this is the right
call: a hundred thousand 768-dim vectors is ~300 MB and scores in milliseconds, with no
index to build, no approximation, and no extra dependency. pgvector with HNSW is the
answer when this moves to Postgres, not before.

Every vector row records its model and dimension, and mixed signatures are filtered out
rather than compared -- vectors from different models are not in the same space, and
comparing them yields rankings that look fine and mean nothing.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# FTS5 column order. `bm25_weights` in the config must line up with this.
FTS_COLUMNS: tuple[str, ...] = (
    "url_hash",     # UNINDEXED, weight 0
    "title",
    "summary",
    "description",
    "tags",
    "entities",
    "body",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_documents (
    url_hash      TEXT PRIMARY KEY,
    canonical_url TEXT NOT NULL,
    platform      TEXT NOT NULL DEFAULT 'web',
    title         TEXT,
    author        TEXT,
    summary       TEXT,
    description   TEXT,
    category      TEXT,
    content_type  TEXT,
    tags_text     TEXT,
    thumbnail_url TEXT,
    published_at  TEXT,
    source_hash   TEXT NOT NULL DEFAULT '',
    enriched      INTEGER NOT NULL DEFAULT 0,
    -- Analysis coverage, recorded so the UI can state plainly how deeply an item was
    -- examined. Without this the distinction is invisible: an item tagged from a caption
    -- and an item where every frame was described look identical in a list, and the only
    -- way to tell them apart is querying the database by hand.
    analysis_mode  TEXT,
    audio_covered  INTEGER NOT NULL DEFAULT 0,
    visual_covered INTEGER NOT NULL DEFAULT 0,
    detail_count   INTEGER NOT NULL DEFAULT 0,
    is_video       INTEGER NOT NULL DEFAULT 0,
    indexed_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sdoc_category ON search_documents(category);
CREATE INDEX IF NOT EXISTS idx_sdoc_type     ON search_documents(content_type);
CREATE INDEX IF NOT EXISTS idx_sdoc_platform ON search_documents(platform);

CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
    url_hash UNINDEXED,
    title,
    summary,
    description,
    tags,
    entities,
    body,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS search_chunks (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    url_hash TEXT NOT NULL,
    ordinal  INTEGER NOT NULL,
    kind     TEXT NOT NULL DEFAULT 'body',
    text     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunk_hash ON search_chunks(url_hash);

CREATE TABLE IF NOT EXISTS search_vectors (
    chunk_id INTEGER PRIMARY KEY,
    url_hash TEXT NOT NULL,
    model    TEXT NOT NULL,
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vec_model ON search_vectors(model, dim);
CREATE INDEX IF NOT EXISTS idx_vec_hash  ON search_vectors(url_hash);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SearchStore:
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
            self._migrate()
            self._conn.commit()
        self._vector_cache: tuple[str, int, list[int], list[str], np.ndarray] | None = None

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created.

        `CREATE TABLE IF NOT EXISTS` silently does nothing on an existing table, so new
        columns never appear and every insert fails with a column-count error. Cheaper to
        handle here than to ask anyone to delete their library.
        """
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(search_documents)")
        }
        additions = (
            ("analysis_mode", "TEXT"),
            ("audio_covered", "INTEGER NOT NULL DEFAULT 0"),
            ("visual_covered", "INTEGER NOT NULL DEFAULT 0"),
            ("detail_count", "INTEGER NOT NULL DEFAULT 0"),
            ("is_video", "INTEGER NOT NULL DEFAULT 0"),
        )
        for name, ddl in additions:
            if name not in existing:
                self._conn.execute(
                    f"ALTER TABLE search_documents ADD COLUMN {name} {ddl}"
                )

        # Index the new column here, not in the schema script. The script runs before this
        # migration, so on a pre-existing database a CREATE INDEX referencing a
        # not-yet-added column fails with "no such column" and takes startup down with it.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sdoc_analysis "
            "ON search_documents(analysis_mode)"
        )

    # ------------------------------------------------------------------- documents
    def upsert_document(self, record: dict[str, Any], fields: dict[str, str]) -> None:
        """Write the document row and refresh its FTS entry."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO search_documents (
                    url_hash, canonical_url, platform, title, author, summary,
                    description, category, content_type, tags_text, thumbnail_url,
                    published_at, source_hash, enriched, analysis_mode, audio_covered,
                    visual_covered, detail_count, is_video, indexed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(url_hash) DO UPDATE SET
                    canonical_url  = excluded.canonical_url,
                    platform       = excluded.platform,
                    title          = excluded.title,
                    author         = excluded.author,
                    summary        = excluded.summary,
                    description    = excluded.description,
                    category       = excluded.category,
                    content_type   = excluded.content_type,
                    tags_text      = excluded.tags_text,
                    thumbnail_url  = excluded.thumbnail_url,
                    published_at   = excluded.published_at,
                    source_hash    = excluded.source_hash,
                    enriched       = excluded.enriched,
                    analysis_mode  = excluded.analysis_mode,
                    audio_covered  = excluded.audio_covered,
                    visual_covered = excluded.visual_covered,
                    detail_count   = excluded.detail_count,
                    is_video       = excluded.is_video,
                    indexed_at     = excluded.indexed_at
                """,
                (
                    record["url_hash"],
                    record["canonical_url"],
                    record.get("platform") or "web",
                    record.get("title"),
                    record.get("author"),
                    record.get("summary"),
                    record.get("description"),
                    record.get("category"),
                    record.get("content_type"),
                    record.get("tags_text"),
                    record.get("thumbnail_url"),
                    record.get("published_at"),
                    record.get("source_hash") or "",
                    int(bool(record.get("enriched"))),
                    record.get("analysis_mode"),
                    int(bool(record.get("audio_covered"))),
                    int(bool(record.get("visual_covered"))),
                    int(record.get("detail_count") or 0),
                    int(bool(record.get("is_video"))),
                    _now(),
                ),
            )
            # FTS5 has no upsert; delete then insert.
            self._conn.execute(
                "DELETE FROM search_fts WHERE url_hash = ?", (record["url_hash"],)
            )
            self._conn.execute(
                """
                INSERT INTO search_fts
                    (url_hash, title, summary, description, tags, entities, body)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    record["url_hash"],
                    fields.get("title", ""),
                    fields.get("summary", ""),
                    fields.get("description", ""),
                    fields.get("tags", ""),
                    fields.get("entities", ""),
                    fields.get("body", ""),
                ),
            )
            self._conn.commit()

    def replace_chunks(
        self, url_hash: str, chunks: Iterable[tuple[int, str, str]]
    ) -> list[int]:
        """Replace an item's chunks and their vectors. Returns new chunk ids."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM search_vectors WHERE url_hash = ?", (url_hash,)
            )
            self._conn.execute(
                "DELETE FROM search_chunks WHERE url_hash = ?", (url_hash,)
            )
            ids: list[int] = []
            for ordinal, kind, text in chunks:
                cursor = self._conn.execute(
                    "INSERT INTO search_chunks (url_hash, ordinal, kind, text) "
                    "VALUES (?,?,?,?)",
                    (url_hash, ordinal, kind, text),
                )
                ids.append(int(cursor.lastrowid or 0))
            self._conn.commit()
        self._vector_cache = None
        return ids

    def put_vectors(
        self,
        url_hash: str,
        chunk_ids: list[int],
        vectors: np.ndarray,
        model: str,
        dim: int,
    ) -> None:
        rows = [
            (
                chunk_id,
                url_hash,
                model,
                dim,
                np.asarray(vector, dtype=np.float32).tobytes(),
            )
            for chunk_id, vector in zip(chunk_ids, vectors)
        ]
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO search_vectors (chunk_id, url_hash, model, dim, vec)
                VALUES (?,?,?,?,?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    url_hash = excluded.url_hash,
                    model    = excluded.model,
                    dim      = excluded.dim,
                    vec      = excluded.vec
                """,
                rows,
            )
            self._conn.commit()
        self._vector_cache = None

    def has_vectors(self, url_hash: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM search_vectors WHERE url_hash = ? LIMIT 1", (url_hash,)
            ).fetchone()
        return row is not None

    def indexed_source_hash(self, url_hash: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT source_hash FROM search_documents WHERE url_hash = ?",
                (url_hash,),
            ).fetchone()
        return row["source_hash"] if row else None

    def delete(self, url_hash: str) -> None:
        with self._lock:
            for statement in (
                "DELETE FROM search_vectors WHERE url_hash = ?",
                "DELETE FROM search_chunks WHERE url_hash = ?",
                "DELETE FROM search_fts WHERE url_hash = ?",
                "DELETE FROM search_documents WHERE url_hash = ?",
            ):
                self._conn.execute(statement, (url_hash,))
            self._conn.commit()
        self._vector_cache = None

    # --------------------------------------------------------------------- keyword
    def keyword_search(
        self,
        query: str,
        limit: int,
        weights: tuple[float, ...],
        filters: dict[str, str] | None = None,
        *,
        prefix_last: bool = True,
    ) -> list[tuple[str, float]]:
        """BM25 ranking over the FTS index. Returns (url_hash, score) best first.

        `bm25()` is negative-better, so it is negated here. Every caller downstream can
        then assume higher is better, which removes a whole category of inverted-ranking
        bugs.
        """
        match = to_fts_query(query, prefix_last=prefix_last)
        if not match:
            return []

        weight_args = ", ".join("?" for _ in weights)
        clauses = ["search_fts MATCH ?"]
        params: list[Any] = [*weights, match]

        joins = ""
        if filters:
            joins = "JOIN search_documents d ON d.url_hash = search_fts.url_hash"
            for column, value in filters.items():
                clauses.append(f"d.{column} = ?")
                params.append(value)

        sql = f"""
            SELECT search_fts.url_hash AS url_hash,
                   bm25(search_fts, {weight_args}) AS score
            FROM search_fts
            {joins}
            WHERE {' AND '.join(clauses)}
            ORDER BY score ASC
            LIMIT ?
        """
        params.append(limit)

        with self._lock:
            try:
                rows = self._conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                # A malformed MATCH expression should return nothing, not explode.
                return []
        return [(row["url_hash"], -float(row["score"])) for row in rows]

    # ---------------------------------------------------------------------- vector
    def load_vectors(self, model: str, dim: int) -> tuple[list[int], list[str], np.ndarray]:
        """All vectors for one model/dimension as a single matrix, cached in memory."""
        cached = self._vector_cache
        if cached and cached[0] == model and cached[1] == dim:
            return cached[2], cached[3], cached[4]

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT chunk_id, url_hash, vec FROM search_vectors
                WHERE model = ? AND dim = ?
                ORDER BY chunk_id
                """,
                (model, dim),
            ).fetchall()

        if not rows:
            empty = np.zeros((0, dim), dtype=np.float32)
            self._vector_cache = (model, dim, [], [], empty)
            return [], [], empty

        chunk_ids = [int(row["chunk_id"]) for row in rows]
        hashes = [str(row["url_hash"]) for row in rows]
        matrix = np.frombuffer(
            b"".join(row["vec"] for row in rows), dtype=np.float32
        ).reshape(len(rows), dim)
        self._vector_cache = (model, dim, chunk_ids, hashes, matrix)
        return chunk_ids, hashes, matrix

    def vector_signatures(self) -> list[tuple[str, int, int]]:
        """(model, dim, count) present in the index, for mismatch detection."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT model, dim, COUNT(*) AS c FROM search_vectors "
                "GROUP BY model, dim ORDER BY c DESC"
            ).fetchall()
        return [(row["model"], int(row["dim"]), int(row["c"])) for row in rows]

    # ------------------------------------------------------------------- retrieval
    def get_documents(self, url_hashes: list[str]) -> dict[str, dict[str, Any]]:
        if not url_hashes:
            return {}
        placeholders = ", ".join("?" for _ in url_hashes)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM search_documents WHERE url_hash IN ({placeholders})",
                url_hashes,
            ).fetchall()
        return {row["url_hash"]: dict(row) for row in rows}

    def list_documents(self, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        """The library view: everything indexed, newest first."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM search_documents
                ORDER BY indexed_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chunk_texts(self, chunk_ids: list[int]) -> dict[int, str]:
        if not chunk_ids:
            return {}
        placeholders = ", ".join("?" for _ in chunk_ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, text FROM search_chunks WHERE id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        return {int(row["id"]): row["text"] for row in rows}

    def matching_hashes(self, filters: dict[str, str]) -> set[str] | None:
        """Hashes passing metadata filters, or None when no filters are set."""
        if not filters:
            return None
        clauses = " AND ".join(f"{column} = ?" for column in filters)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT url_hash FROM search_documents WHERE {clauses}",
                list(filters.values()),
            ).fetchall()
        return {row["url_hash"] for row in rows}

    def hashes_with_tag(self, tag: str) -> set[str]:
        """Tag filter, read from the enrichment stage's denormalized tag table."""
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT url_hash FROM enrichment_tags WHERE tag = ?", (tag,)
                ).fetchall()
            except sqlite3.OperationalError:
                return set()
        return {row["url_hash"] for row in rows}

    # ----------------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        with self._lock:
            docs = self._conn.execute(
                "SELECT COUNT(*) AS c, SUM(enriched) AS e FROM search_documents"
            ).fetchone()
            chunks = self._conn.execute(
                "SELECT COUNT(*) AS c FROM search_chunks"
            ).fetchone()
            vectors = self._conn.execute(
                "SELECT COUNT(*) AS c FROM search_vectors"
            ).fetchone()
            platforms = self._conn.execute(
                "SELECT platform, COUNT(*) AS c FROM search_documents "
                "GROUP BY platform ORDER BY c DESC"
            ).fetchall()
        return {
            "documents": int(docs["c"] or 0),
            "enriched": int(docs["e"] or 0),
            "chunks": int(chunks["c"] or 0),
            "vectors": int(vectors["c"] or 0),
            "signatures": self.vector_signatures(),
            "platforms": {row["platform"]: int(row["c"]) for row in platforms},
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_FTS_TOKEN = re.compile(r"[^\w]+", re.UNICODE)

# Removed from keyword queries only. Natural questions are mostly function words -- "how
# do I remember things better" carries one real term -- and OR-ing the rest matches
# everything in the library. BM25's IDF weighting is supposed to discount common terms,
# but IDF is weak on a small corpus, so an unrelated document that happens to contain
# "things" still ranks. These words stay fully indexed; they are only dropped from the
# query side, where they contribute noise rather than meaning.
_QUERY_STOPWORDS: frozenset[str] = frozenset(
    """
a an and are as at be been but by can could did do does doing for from had has have
how i if in into is it its me my of on or our so some that the their them then there
these they this those to us was we were what when where which who whom why will with
would you your about again all also any because before being below between both during
each few further here more most no nor not now only other over own same should such
than too under until up very
thing things stuff lot lots way ways better best good great nice really just get got
want need make made like show tell find look see know
""".split()
)


def to_fts_query(query: str, *, prefix_last: bool = True) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    User input cannot go straight into FTS5's query language: bare `"`, `*`, `-`, `^`,
    `NEAR`, `AND` and `OR` are operators there, so `what's the "best" pizza` is a syntax
    error rather than a search. Tokens are extracted and quoted individually.

    They are OR-ed rather than AND-ed. FTS5 defaults to AND, which makes any query longer
    than a few words return nothing -- and long, vague queries are exactly what this
    system is for.

    `prefix_last` makes search-as-you-type actually work. FTS5 matches whole tokens, so a
    half-typed word matches nothing at all: measured on this index, "p" and "pa" both
    returned zero results while "pas" returned one. Treating the final token as a prefix
    when the query does not end on a separator means results appear from the first
    keystroke and narrow as you type, which is the entire point of a live search box.

    The final token is only treated as a prefix when the query ends mid-word. Once the
    user types a space, that word is complete and matching it exactly is more precise.
    """
    raw = [token for token in _FTS_TOKEN.split(query.lower()) if token]
    if not raw:
        return ""

    still_typing = prefix_last and bool(query) and (query[-1].isalnum() or query[-1] == "_")
    partial = raw[-1] if still_typing else None
    complete = raw[:-1] if still_typing else raw

    tokens = [token for token in complete if len(token) > 1]
    terms = [token for token in tokens if token not in _QUERY_STOPWORDS]
    parts = [f'"{token}"' for token in terms]

    # Fall back to function words only when they are genuinely all there is. If a partial
    # term is still being typed, that partial carries the meaning -- "how do i rem" should
    # search on "rem", not on "how" and "do", which match most of the library.
    if not parts and not partial:
        parts = [f'"{token}"' for token in tokens]

    if partial:
        # A one-character prefix is only worth running when there is nothing else to go
        # on -- otherwise it drags in a large slice of the library on every keystroke.
        if len(partial) > 1 or not parts:
            parts.append(f'"{partial}"*')

    return " OR ".join(parts)
