"""Cold Storage (Layer 4)

Portable, dependency-light implementation using SQLite.

Design goals:
  - Works everywhere Python does (no external services)
  - Supports archiving + retrieval immediately (no stub)
  - Provides a baseline text search via SQLite FTS5 when available

Notes:
  - FTS5 is included in most modern Python SQLite builds, but not all.
    If unavailable, we fall back to LIKE-based search.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ColdStorageStats:
    enabled: bool
    status: str
    total_rows: int
    fts_enabled: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "total_rows": self.total_rows,
            "fts_enabled": self.fts_enabled,
        }


class ColdStorage:
    """Compressed-ish archive for rarely-accessed historical data.

    This implementation stores plaintext + metadata JSON. Compression can be
    added later (e.g., zstd) without changing the external API.
    """

    def __init__(self, db_path: Optional[Path] = None):
        # If db_path isn't provided, default to a per-user location.
        if db_path is None:
            db_path = Path.home() / ".engram" / "cold.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._enabled = True
        self._fts_enabled = False

        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection.

        Streamlit (and many web servers) can execute user code in different threads.
        SQLite connections are, by default, bound to the thread that created them.
        We therefore keep one connection per thread to avoid cross-thread usage errors.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            # WAL mode: allows the forgetting daemon (reader) and archive/retrieve
            # calls (writer) to run concurrently without serializing at file level.
            # synchronous=NORMAL is safe with WAL and significantly faster than FULL.
            # cache_size=-8000 reserves 8MB page cache; helps repeated FTS5 scans.
            try:
                conn.execute("PRAGMA foreign_keys=ON;")
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA cache_size=-8000;")
            except Exception:
                pass
            self._local.conn = conn
        return conn

    @staticmethod
    def _to_fts5_query(user_query: str) -> str:
        """Convert arbitrary user text into a safe SQLite FTS5 query.

        SQLite FTS5 has its own query syntax; punctuation like '?' can raise
        errors (e.g., 'fts5: syntax error near "?"'). To keep UX robust, we
        extract "word" tokens and join them with OR semantics.

        OR semantics give better recall: a document matches if it contains
        *any* of the query terms (rather than *all* of them), which is the
        right behavior for fuzzy memory retrieval.  The Python-side re-ranker
        that runs after FTS handles relevance ordering.

        Returns an empty string if no usable tokens were found.
        """
        # Keep letters/digits/underscore as tokens; drop punctuation.
        terms = re.findall(r"[A-Za-z0-9_]+", user_query or "")
        if not terms:
            return ""
        # OR semantics: document matches if any term is present.
        return " OR ".join(terms)


    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        """Initialize schema with lightweight migrations.

        Schema versions:
          - v1: cold_memories(id, timestamp, project_id, session_id, text, metadata_json) + optional FTS
          - v2: add content_hash (dedup) + schema_version table
        """
        cur = self._get_conn().cursor()

        # Ensure schema_version table exists.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );
            """
        )
        cur.execute("SELECT version FROM schema_version LIMIT 1;")
        row = cur.fetchone()
        current_version = int(row[0]) if row else 0

        # Base table (create if missing) in its latest form.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cold_memories (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                project_id TEXT,
                session_id TEXT,
                text TEXT NOT NULL,
                metadata_json TEXT,
                content_hash TEXT
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cold_ts ON cold_memories(timestamp);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cold_project ON cold_memories(project_id);")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_cold_hash ON cold_memories(project_id, content_hash);")  # project-scoped dedup

        # Migrate to v2 if needed: add content_hash and backfill.
        if current_version < 2:
            # Add column if it's missing (SQLite doesn't support IF NOT EXISTS for ADD COLUMN).
            try:
                cur.execute("ALTER TABLE cold_memories ADD COLUMN content_hash TEXT;")
            except sqlite3.OperationalError:
                pass  # column already exists

            # Backfill hashes for existing rows.
            cur.execute("SELECT id, project_id, text FROM cold_memories WHERE content_hash IS NULL OR content_hash = ''; ")
            rows = cur.fetchall()
            for r in rows:
                mid = r[0]
                proj = r[1] or ""
                text_val = (r[2] or "").strip()
                h = hashlib.sha256((proj + "\n" + " ".join(text_val.split())).encode("utf-8", errors="ignore")).hexdigest()
                cur.execute("UPDATE cold_memories SET content_hash = ? WHERE id = ?;", (h, mid))

            # Set schema version to 2 (insert or replace the single row).
            cur.execute("DELETE FROM schema_version;")
            cur.execute("INSERT INTO schema_version(version) VALUES (2);")
            current_version = 2

        # Try to enable FTS5.
        try:
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS cold_fts
                USING fts5(text, memory_id UNINDEXED, project_id UNINDEXED, session_id UNINDEXED, timestamp UNINDEXED);
                """
            )
            self._fts_enabled = True
        except sqlite3.OperationalError as e:
            logger.info(
                "SQLite FTS5 unavailable for cold storage (%s); falling back to LIKE search.",
                e,
            )
            self._fts_enabled = False

        self._get_conn().commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def archive(self, memories: List[Dict[str, Any]]) -> int:
        """Move memories to cold storage. Returns count archived."""
        if not self._enabled or not memories:
            return 0

        rows = []
        fts_rows = []
        now = time.time()

        for m in memories:
            mid = m.get("id") or str(uuid.uuid4())
            ts = float(m.get("timestamp") or now)
            project_id = m.get("project_id")
            session_id = m.get("session_id")
            text = m.get("text") or ""
            metadata = m.get("metadata") or {}
            norm_text = " ".join((text or "").split())
            content_hash = hashlib.sha256(((project_id or "") + "\n" + norm_text).encode("utf-8", errors="ignore")).hexdigest()

            rows.append(
                (
                    mid,
                    ts,
                    project_id,
                    session_id,
                    text,
                    json.dumps(metadata, ensure_ascii=False),
                    content_hash,
                )
            )
            if self._fts_enabled:
                fts_rows.append((text, mid, project_id, session_id, ts))

        cur = self._get_conn().cursor()
        cur.executemany(
            """
            INSERT OR IGNORE INTO cold_memories
            (id, timestamp, project_id, session_id, text, metadata_json, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        if self._fts_enabled and fts_rows:
            # Keep FTS entries in sync. Replace by deleting and reinserting memory_id.
            for (_, mid, *_rest) in fts_rows:
                cur.execute("DELETE FROM cold_fts WHERE memory_id = ?;", (mid,))
            cur.executemany(
                "INSERT INTO cold_fts (text, memory_id, project_id, session_id, timestamp) VALUES (?, ?, ?, ?, ?);",
                fts_rows,
            )

        self._get_conn().commit()
        return len(rows)

    def get_all(
        self,
        project_id: str,
        *,
        after_ts: float = 0.0,
        batch_size: int = 500,
    ):
        """Yield all cold-storage records for a project as dicts.

        Uses the ``idx_cold_project`` index; never does a full table scan.
        Results are ordered by timestamp ascending.  Yields in batches of
        ``batch_size`` to keep memory usage bounded for large archives.

        Args:
            project_id: Project to scope the scan.
            after_ts: Only return records with timestamp > this value.
            batch_size: Rows fetched per SQLite round-trip.
        """
        if not self._enabled:
            return
        cur = self._get_conn().cursor()
        offset = 0
        while True:
            cur.execute(
                """
                SELECT id, timestamp, project_id, session_id, text, metadata_json
                FROM cold_memories
                WHERE project_id = ? AND timestamp > ?
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
                """,
                (project_id, after_ts, batch_size, offset),
            )
            rows = cur.fetchall()
            if not rows:
                break
            for row in rows:
                yield {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "project_id": row["project_id"],
                    "session_id": row["session_id"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                }
            if len(rows) < batch_size:
                break
            offset += batch_size

    def retrieve(
        self,
        query: str,
        n: int = 5,
        *,
        project_id: Optional[str] = None,
        recency_half_life_days: float = 30.0,
        bm25_weight: float = 1.0,
        recency_weight: float = 0.35,
        importance_weight: float = 0.20,
        fetch_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve from archive.

        If `query` is empty, returns the most recent items.
        If `project_id` is given, results are filtered to that project only.

        Reranking (FTS path) combines three signals:
          - BM25 lexical relevance (normalised to [0, 1])
          - Recency decay (exponential half-life)
          - Episode importance (stored in metadata_json at archive time)

        The importance signal ensures that high-importance episodes (importance
        close to 1.0) rank above low-importance episodes with similar lexical
        match — preventing the archive from surfacing trivial exchanges ahead of
        significant ones just because they match more query words.
        """
        if not self._enabled:
            return []

        query = (query or "").strip()
        cur = self._get_conn().cursor()

        if not query:
            if project_id:
                cur.execute(
                    """
                    SELECT id, timestamp, project_id, session_id, text, metadata_json
                    FROM cold_memories
                    WHERE project_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?;
                    """,
                    (project_id, n),
                )
            else:
                cur.execute(
                    """
                    SELECT id, timestamp, project_id, session_id, text, metadata_json
                    FROM cold_memories
                    ORDER BY timestamp DESC
                    LIMIT ?;
                    """,
                    (n,),
                )
        elif self._fts_enabled:
            # Retrieve candidates by bm25 (lower is better), then re-rank in Python
            # using a combined score that also rewards recency.
            fts_query = self._to_fts5_query(query)
            if not fts_query:
                # If nothing usable after sanitization, fall back to LIKE.
                like = f"%{query}%"
                params_like = (like, project_id, n) if project_id else (like, n)
                proj_clause = " AND project_id = ?" if project_id else ""
                cur.execute(
                    f"""
                    SELECT id, timestamp, project_id, session_id, text, metadata_json
                    FROM cold_memories
                    WHERE text LIKE ?{proj_clause}
                    ORDER BY timestamp DESC
                    LIMIT ?;
                    """,
                    params_like,
                )
            else:
                if fetch_k is None:
                    fetch_k = max(50, n * 10)
                if project_id:
                    cur.execute(
                        """
                        SELECT m.id, m.timestamp, m.project_id, m.session_id, m.text, m.metadata_json,
                               bm25(cold_fts) AS bm25
                        FROM cold_fts
                        JOIN cold_memories m ON m.id = cold_fts.memory_id
                        WHERE cold_fts MATCH ? AND m.project_id = ?
                        ORDER BY bm25(cold_fts)
                        LIMIT ?;
                        """,
                        (fts_query, project_id, fetch_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT m.id, m.timestamp, m.project_id, m.session_id, m.text, m.metadata_json,
                               bm25(cold_fts) AS bm25
                        FROM cold_fts
                        JOIN cold_memories m ON m.id = cold_fts.memory_id
                        WHERE cold_fts MATCH ?
                        ORDER BY bm25(cold_fts)
                        LIMIT ?;
                        """,
                        (fts_query, fetch_k),
                    )
        else:
            # Fallback: LIKE search.
            like = f"%{query}%"
            params_like = (like, project_id, n) if project_id else (like, n)
            proj_clause = " AND project_id = ?" if project_id else ""
            cur.execute(
                f"""
                SELECT id, timestamp, project_id, session_id, text, metadata_json
                FROM cold_memories
                WHERE text LIKE ?{proj_clause}
                ORDER BY timestamp DESC
                LIMIT ?;
                """,
                params_like,
            )

        rows = cur.fetchall()

        # If FTS: apply combined re-ranking.
        if query and self._fts_enabled and rows:
            now = time.time()
            half_life_s = max(1.0, float(recency_half_life_days) * 86400.0)

            scored = []
            for r in rows:
                ts = float(r["timestamp"])
                bm = float(r["bm25"]) if r["bm25"] is not None else 0.0
                sim = 1.0 / (1.0 + max(0.0, bm))  # higher is better
                age_s = max(0.0, now - ts)
                rec = 0.5 ** (age_s / half_life_s)  # [0, 1]
                # Extract importance from metadata_json (stored at archive time).
                # Importance is in the outer metadata dict as "importance" key.
                try:
                    meta = json.loads(r["metadata_json"] or "{}")
                    imp = float(meta.get("importance", 0.5) or 0.5)
                    imp = max(0.0, min(1.0, imp))
                except Exception:
                    imp = 0.5
                score = (bm25_weight * sim) + (recency_weight * rec) + (importance_weight * imp)
                scored.append((score, r))

            scored.sort(key=lambda t: t[0], reverse=True)
            rows = [r for _, r in scored[:n]]

        out: List[Dict[str, Any]] = []
        for row in rows[:n]:
            out.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "project_id": row["project_id"],
                    "session_id": row["session_id"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                }
            )
        return out

    def get_stats(self) -> Dict[str, Any]:
        if not self._enabled:
            return {"enabled": False, "status": "disabled"}
        cur = self._get_conn().cursor()
        cur.execute("SELECT COUNT(*) FROM cold_memories;")
        total = int(cur.fetchone()[0])
        return ColdStorageStats(
            enabled=True,
            status="ok",
            total_rows=total,
            fts_enabled=self._fts_enabled,
        ).to_dict()

    def close(self) -> None:
        """Close the current thread's connection (if any)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            pass
        finally:
            self._local.conn = None

    def __enter__(self) -> "ColdStorage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
