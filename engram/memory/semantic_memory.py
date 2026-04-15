"""
Semantic Memory Implementation

SQLite-backed structured knowledge store replacing the previous Kuzu
graph database implementation. Stores facts, preferences, and events
with FTS5 full-text search and WAL-mode concurrent reads.

Preserves the full public interface of the Kuzu implementation so no
changes are required in retrieval.py, ingestion.py, or project_memory.py.

Graph traversal (add_node, add_relationship, query) is stubbed — those
methods return safely but do nothing. When graph traversal becomes a
concrete requirement (e.g. language tutor learner model queries), a
migration path from this flat schema to a proper graph store is
straightforward since all data is in well-structured SQLite tables.

Author: Jeffrey Dean
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import Node, ProjectType, Relationship
# Re-export ProjectType so existing import sites don't need updating:
# `from .memory.semantic_memory import ProjectType` continues to work.
__all__ = ["SemanticMemory", "ProjectType", "Node", "Relationship"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA cache_size=-8000;

CREATE TABLE IF NOT EXISTS facts (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.7,
    source      TEXT NOT NULL DEFAULT 'conversation',
    metadata    TEXT NOT NULL DEFAULT '{}',
    timestamp   REAL NOT NULL,
    user_id     TEXT NOT NULL DEFAULT 'primary_user'
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    content,
    content='facts',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 1'
);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO facts_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS preferences (
    id          TEXT PRIMARY KEY,
    category    TEXT NOT NULL,
    value       TEXT NOT NULL,
    strength    REAL NOT NULL DEFAULT 0.7,
    source      TEXT NOT NULL DEFAULT 'conversation',
    timestamp   REAL NOT NULL,
    user_id     TEXT NOT NULL DEFAULT 'primary_user'
);

CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '',
    importance  REAL NOT NULL DEFAULT 0.6,
    source      TEXT NOT NULL DEFAULT 'lifecycle',
    metadata    TEXT NOT NULL DEFAULT '{}',
    timestamp   REAL NOT NULL,
    user_id     TEXT NOT NULL DEFAULT 'default_user'
);

CREATE INDEX IF NOT EXISTS facts_timestamp     ON facts(timestamp DESC);
CREATE INDEX IF NOT EXISTS facts_source        ON facts(source);
CREATE INDEX IF NOT EXISTS prefs_category      ON preferences(category);
CREATE INDEX IF NOT EXISTS prefs_timestamp     ON preferences(timestamp DESC);
CREATE INDEX IF NOT EXISTS events_timestamp    ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS events_importance   ON events(importance DESC);
"""


def _stable_id(prefix: str, *parts: str) -> str:
    key = ":".join(parts)
    return f"{prefix}_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def _tokenize(text: str) -> set:
    raw = re.findall(r"[A-Za-z0-9_./-]+", (text or "").lower())
    out: set = set()
    for tok in raw:
        if len(tok) > 2:
            out.add(tok)
    return out


def _lexical_overlap(query_terms: set, text: str) -> float:
    if not query_terms:
        return 0.0
    text_terms = _tokenize(text)
    if not text_terms:
        return 0.0
    return len(query_terms & text_terms) / max(1, len(query_terms))


class SemanticMemory:
    """SQLite-backed semantic memory store.

    Replaces the Kuzu graph database with a simpler, more robust
    implementation using SQLite WAL mode and FTS5 full-text search.

    Thread safety: all writes are serialized via a threading.RLock.
    Reads use a separate connection per thread via threading.local
    to allow concurrent reads under WAL mode without blocking writers.

    Args:
        db_path:      Path to SQLite database file, or directory (creates
                      semantic.db inside). None = in-memory (testing).
        project_type: Accepted for interface compatibility; no effect.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        project_type: Optional[ProjectType] = None,
        **kwargs,  # absorb buffer_pool_size etc. from old call sites
    ):
        self.db_path = db_path
        self.project_type = project_type

        # Resolve database file path
        if db_path is None:
            self._db_file = ":memory:"
        else:
            db_path = Path(db_path).expanduser().resolve(strict=False)
            if db_path.suffix in (".db", ".sqlite", ".sqlite3"):
                self._db_file = str(db_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Treat as directory (legacy: old code passed a dir for Kuzu)
                db_path.mkdir(parents=True, exist_ok=True)
                self._db_file = str(db_path / "semantic.db")

        self._lock = threading.RLock()
        self._local = threading.local()

        # Writer connection (single, serialized)
        self._write_conn = self._open_connection()
        self._apply_schema(self._write_conn)

        # Compatibility attributes expected by tests / legacy callers
        self.node_tables: set = {"Fact", "Preference", "Event", "User"}
        self.rel_tables: set = set()

        logger.info("SemanticMemory (SQLite) initialized at %s", self._db_file)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_file,
            check_same_thread=False,
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA cache_size=-8000;")
        return conn

    def _apply_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(_DDL)
        conn.commit()

    def _reader(self) -> sqlite3.Connection:
        """Return a per-thread read connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._open_connection()
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        confidence: float = 0.7,
        source: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
        user_id: str = "primary_user",
        fact_id: Optional[str] = None,
    ) -> Node:
        """Add or update a fact. Returns a Node for interface compatibility."""
        fid = fact_id or _stable_id("fact", content)
        meta = json.dumps(metadata or {})
        ts = time.time()

        with self._lock:
            self._write_conn.execute(
                """
                INSERT INTO facts (id, content, confidence, source, metadata, timestamp, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content    = excluded.content,
                    confidence = excluded.confidence,
                    source     = excluded.source,
                    metadata   = excluded.metadata,
                    timestamp  = excluded.timestamp
                """,
                (fid, content, confidence, source, meta, ts, user_id),
            )
            self._write_conn.commit()

        return Node(
            table="Fact",
            id=fid,
            properties={"content": content, "confidence": confidence,
                        "source": source, "metadata": metadata or {},
                        "timestamp": ts},
        )

    def list_facts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return facts ordered by timestamp descending."""
        try:
            cur = self._reader().execute(
                "SELECT id, content, confidence, source, metadata, timestamp "
                "FROM facts ORDER BY timestamp DESC LIMIT ?",
                (int(limit),),
            )
            rows = []
            for r in cur.fetchall():
                rows.append({
                    "id": r["id"],
                    "content": r["content"],
                    "confidence": r["confidence"],
                    "source": r["source"],
                    "metadata": json.loads(r["metadata"] or "{}"),
                    "timestamp": r["timestamp"],
                })
            return rows
        except Exception as exc:
            logger.warning("list_facts failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    def add_preference(
        self,
        category: str,
        value: str,
        strength: float = 0.7,
        source: str = "conversation",
        user_id: str = "primary_user",
        preference_id: Optional[str] = None,
    ) -> Node:
        """Add or update a preference. Returns a Node for interface compatibility."""
        pid = preference_id or _stable_id("pref", category, value)
        ts = time.time()

        with self._lock:
            self._write_conn.execute(
                """
                INSERT INTO preferences (id, category, value, strength, source, timestamp, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    strength  = excluded.strength,
                    source    = excluded.source,
                    timestamp = excluded.timestamp
                """,
                (pid, category, value, strength, source, ts, user_id),
            )
            self._write_conn.commit()

        return Node(
            table="Preference",
            id=pid,
            properties={"category": category, "value": value,
                        "strength": strength, "source": source,
                        "timestamp": ts},
        )

    def list_preferences(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return preferences ordered by timestamp descending."""
        try:
            cur = self._reader().execute(
                "SELECT id, category, value, strength, source, timestamp "
                "FROM preferences ORDER BY timestamp DESC LIMIT ?",
                (int(limit),),
            )
            rows = []
            for r in cur.fetchall():
                rows.append({
                    "id": r["id"],
                    "category": r["category"],
                    "value": r["value"],
                    "strength": r["strength"],
                    "source": r["source"],
                    "timestamp": r["timestamp"],
                })
            return rows
        except Exception as exc:
            logger.warning("list_preferences failed: %s", exc)
            return []

    def search_preferences(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search preferences by lexical overlap on category and value."""
        query_terms = _tokenize(query)
        rows = self.list_preferences(limit=100)
        scored = []
        for row in rows:
            text = f"{row['category']} {row['value']}"
            score = _lexical_overlap(query_terms, text)
            row["match_score"] = round(score, 4)
            scored.append(row)
        scored.sort(key=lambda r: r["match_score"], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def add_event(
        self,
        summary: str,
        *,
        detail: str = "",
        importance: float = 0.6,
        source: str = "lifecycle",
        metadata: Optional[Dict[str, Any]] = None,
        user_id: str = "default_user",
        event_id: Optional[str] = None,
    ) -> Node:
        """Add an event. Returns a Node for interface compatibility."""
        eid = event_id or f"event_{int(time.time() * 1_000_000)}"
        meta = json.dumps(metadata or {})
        ts = time.time()

        with self._lock:
            self._write_conn.execute(
                """
                INSERT INTO events (id, summary, detail, importance, source, metadata, timestamp, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    summary    = excluded.summary,
                    detail     = excluded.detail,
                    importance = excluded.importance,
                    source     = excluded.source,
                    metadata   = excluded.metadata,
                    timestamp  = excluded.timestamp
                """,
                (eid, summary, detail, importance, source, meta, ts, user_id),
            )
            self._write_conn.commit()

        return Node(
            table="Event",
            id=eid,
            properties={"summary": summary, "detail": detail,
                        "importance": importance, "source": source,
                        "metadata": metadata or {}, "timestamp": ts},
        )

    def list_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return events ordered by timestamp descending."""
        try:
            cur = self._reader().execute(
                "SELECT id, summary, detail, importance, source, metadata, timestamp "
                "FROM events ORDER BY timestamp DESC LIMIT ?",
                (int(limit),),
            )
            rows = []
            for r in cur.fetchall():
                rows.append({
                    "id": r["id"],
                    "summary": r["summary"],
                    "detail": r["detail"],
                    "importance": r["importance"],
                    "source": r["source"],
                    "metadata": json.loads(r["metadata"] or "{}"),
                    "timestamp": r["timestamp"],
                })
            return rows
        except Exception as exc:
            logger.warning("list_events failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Unified search (SemanticLayerProtocol)
    # ------------------------------------------------------------------

    def search_generic_memories(
        self,
        query: str,
        *,
        limit: int = 20,
        per_type_limit: int = 60,
        include_graph: bool = True,
        graph_sentence_limit: int = 6,
    ) -> List[Dict[str, Any]]:
        """Return scored semantic rows matching query.

        Satisfies SemanticLayerProtocol so UnifiedRetriever can use the
        preferred code path without falling back to generic_memory_rows.

        Scoring:
        - Facts: FTS5 BM25 rank (if match) + lexical overlap
        - Preferences: lexical overlap on category + value
        - Events: lexical overlap on summary
        All rows include a normalised match_score in [0, 1].
        """
        results: List[Dict[str, Any]] = []
        query_terms = _tokenize(query)
        now = time.time()

        # --- Facts (FTS5) ---
        try:
            # FTS5 rank: negative, more negative = better match
            fts_rows = self._reader().execute(
                """
                SELECT f.id, f.content, f.confidence, f.source,
                       f.metadata, f.timestamp,
                       bm25(facts_fts) AS bm25_rank
                FROM facts_fts
                JOIN facts f ON facts_fts.rowid = f.rowid
                WHERE facts_fts MATCH ?
                ORDER BY bm25_rank
                LIMIT ?
                """,
                (self._fts_query(query), per_type_limit),
            ).fetchall()

            # Normalise BM25: convert negative rank to [0,1]
            bm25_vals = [abs(r["bm25_rank"]) for r in fts_rows] or [1.0]
            bm25_max = max(bm25_vals) or 1.0

            for r in fts_rows:
                lexical = _lexical_overlap(query_terms, r["content"])
                bm25_norm = abs(r["bm25_rank"]) / bm25_max
                match_score = min(1.0, 0.6 * bm25_norm + 0.4 * lexical)
                results.append({
                    "type": "fact",
                    "id": r["id"],
                    "content": r["content"],
                    "confidence": r["confidence"],
                    "source": r["source"],
                    "metadata": json.loads(r["metadata"] or "{}"),
                    "timestamp": r["timestamp"],
                    "match_score": round(match_score, 4),
                })
        except Exception as exc:
            logger.debug("FTS search failed, falling back: %s", exc)
            # Fallback: plain lexical scan
            for row in self.list_facts(limit=per_type_limit):
                score = _lexical_overlap(query_terms, row["content"])
                if score > 0:
                    row["type"] = "fact"
                    row["match_score"] = round(score, 4)
                    results.append(row)

        # --- Preferences (lexical) ---
        try:
            for row in self.list_preferences(limit=per_type_limit):
                text = f"{row['category']} {row['value']}"
                score = _lexical_overlap(query_terms, text)
                row["type"] = "preference"
                row["match_score"] = round(score, 4)
                results.append(row)
        except Exception as exc:
            logger.debug("Preference search failed: %s", exc)

        # --- Events (lexical) ---
        try:
            for row in self.list_events(limit=per_type_limit):
                text = f"{row['summary']} {row['detail']}"
                score = _lexical_overlap(query_terms, text)
                row["type"] = "event"
                row["match_score"] = round(score, 4)
                results.append(row)
        except Exception as exc:
            logger.debug("Event search failed: %s", exc)

        # Sort by match_score descending, cap at limit
        results.sort(key=lambda r: r["match_score"], reverse=True)
        return results[:limit]

    def _fts_query(self, query: str) -> str:
        """Convert a natural language query to an FTS5 match expression.

        Strips punctuation that would cause FTS5 parse errors, then
        wraps each token in double-quotes to force exact-token matching
        rather than prefix matching.  Falls back to a simple phrase if
        quoting fails.
        """
        tokens = re.findall(r"[A-Za-z0-9_]+", query)
        if not tokens:
            return '""'
        return " OR ".join(f'"{tok}"' for tok in tokens if len(tok) > 1)

    # ------------------------------------------------------------------
    # Generic rows (legacy fallback path in retrieval.py)
    # ------------------------------------------------------------------

    def generic_memory_rows(self, limit_per_type: int = 100) -> List[Dict[str, Any]]:
        """Return all rows from all tables — legacy fallback."""
        rows: List[Dict[str, Any]] = []
        rows.extend({"type": "fact", **r} for r in self.list_facts(limit=limit_per_type))
        rows.extend({"type": "preference", **r} for r in self.list_preferences(limit=limit_per_type))
        rows.extend({"type": "event", **r} for r in self.list_events(limit=limit_per_type))
        return rows

    # ------------------------------------------------------------------
    # Graph stubs (interface compatibility)
    # ------------------------------------------------------------------

    def add_node(
        self,
        table: str,
        node_id: str,
        properties: Dict[str, Any],
    ) -> Node:
        """Graph stub — routes Fact/Preference/Event to their SQL methods.

        Other node types (Entity, Concept, etc.) are silently ignored until
        graph traversal becomes a concrete requirement.
        """
        t = table.lower()
        if t == "fact":
            content = properties.get("content", "")
            return self.add_fact(
                content=content,
                confidence=float(properties.get("confidence", 0.7)),
                source=str(properties.get("source", "conversation")),
                metadata=properties.get("metadata") if isinstance(properties.get("metadata"), dict) else None,
                fact_id=node_id,
            )
        if t == "preference":
            return self.add_preference(
                category=str(properties.get("category", "general")),
                value=str(properties.get("value", "")),
                strength=float(properties.get("strength", 0.7)),
                preference_id=node_id,
            )
        if t == "event":
            return self.add_event(
                summary=str(properties.get("summary", "")),
                detail=str(properties.get("detail", "")),
                importance=float(properties.get("importance", 0.6)),
                source=str(properties.get("source", "lifecycle")),
                event_id=node_id,
            )
        # Unknown table type (Entity, Concept, etc.) — no-op
        logger.debug("add_node: ignoring unsupported table type %r", table)
        return Node(table=table, id=node_id, properties=properties)

    def add_relationship(
        self,
        from_table: str,
        from_id: str,
        to_table: str,
        to_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Relationship:
        """Graph stub — no-op. Relationships are not stored in the SQLite backend."""
        logger.debug(
            "add_relationship: %s(%s)-[%s]->%s(%s) not stored (SQLite backend)",
            from_table, from_id, rel_type, to_table, to_id,
        )
        return Relationship(
            rel_type=rel_type,
            from_id=from_id,
            to_id=to_id,
            properties=properties or {},
        )

    def query(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Graph stub — Cypher queries are not supported in the SQLite backend.

        Returns an empty list. Code that calls this for graph traversal
        will silently get no results. This is intentional — graph traversal
        is not yet a concrete requirement and will be implemented when needed.
        """
        logger.debug("query(): Cypher not supported in SQLite backend — returning []")
        return []

    def get_node(self, table: str, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID from the appropriate table."""
        t = table.lower()
        try:
            if t == "fact":
                cur = self._reader().execute(
                    "SELECT * FROM facts WHERE id = ?", (node_id,)
                )
                r = cur.fetchone()
                if r:
                    return Node(table=table, id=node_id,
                                properties=dict(r))
            elif t == "preference":
                cur = self._reader().execute(
                    "SELECT * FROM preferences WHERE id = ?", (node_id,)
                )
                r = cur.fetchone()
                if r:
                    return Node(table=table, id=node_id,
                                properties=dict(r))
            elif t == "event":
                cur = self._reader().execute(
                    "SELECT * FROM events WHERE id = ?", (node_id,)
                )
                r = cur.fetchone()
                if r:
                    return Node(table=table, id=node_id,
                                properties=dict(r))
        except Exception as exc:
            logger.debug("get_node(%s, %s) failed: %s", table, node_id, exc)
        return None

    def delete_node(self, table: str, node_id: str) -> None:
        """Delete a node from the appropriate table."""
        t = table.lower()
        table_map = {"fact": "facts", "preference": "preferences", "event": "events"}
        sql_table = table_map.get(t)
        if sql_table is None:
            logger.debug("delete_node: unsupported table %r", table)
            return
        with self._lock:
            self._write_conn.execute(
                f"DELETE FROM {sql_table} WHERE id = ?", (node_id,)
            )
            self._write_conn.commit()

    # ------------------------------------------------------------------
    # Compat stubs expected by tests / project_memory.py
    # ------------------------------------------------------------------

    def _create_node_table_safe(self, *args, **kwargs) -> None:
        """No-op — schema is fixed at init time."""

    def _create_rel_table_safe(self, *args, **kwargs) -> None:
        """No-op — relationships are not stored."""

    def _ensure_default_user(self, user_id: str = "primary_user") -> None:
        """No-op — User table not present in SQLite backend."""

    # ------------------------------------------------------------------
    # Stats & lifecycle
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return row counts and storage info."""
        stats: Dict[str, Any] = {
            "backend": "sqlite",
            "db_file": self._db_file,
            "node_tables": list(self.node_tables),
            "node_counts": {},
        }
        try:
            conn = self._reader()
            for table, sql_table in (
                ("Fact", "facts"),
                ("Preference", "preferences"),
                ("Event", "events"),
            ):
                cur = conn.execute(f"SELECT COUNT(*) FROM {sql_table}")
                stats["node_counts"][table] = cur.fetchone()[0]
        except Exception as exc:
            logger.debug("get_stats failed: %s", exc)
        return stats

    def close(self) -> None:
        """Close database connections."""
        try:
            with self._lock:
                if self._write_conn:
                    self._write_conn.close()
                    self._write_conn = None
        except Exception:
            pass
        try:
            local_conn = getattr(self._local, "conn", None)
            if local_conn:
                local_conn.close()
                self._local.conn = None
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()