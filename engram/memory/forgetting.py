"""Forgetting policy for Engram.

Manages the episodic → cold storage lifecycle:

  1. Track access patterns (which episodes get retrieved by search)
  2. Score episodes by retention value
  3. Archive low-retention episodes to cold storage
  4. Delete archived episodes from episodic memory

Retention score = weighted combination of:
  - Recency:    exponential decay (recent = higher)
  - Importance: the 0-1 score assigned at storage time
  - Access:     log-scaled frequency of retrieval hits
  - Surprise:   higher surprise at storage time = more worth keeping

Episodes below the retention threshold are archived to cold storage
where they remain searchable via FTS5 but no longer consume vector
index resources.

Usage:
    policy = ForgettingPolicy(
        access_db_path=project_dir / "access_tracker.db",
        config=ForgettingConfig(),
    )

    # After each search, record access
    episodes = memory.episodic.search("asyncio")
    policy.record_access([ep.id for ep in episodes])

    # Periodically run maintenance
    archived = policy.run(
        episodic=memory.episodic,
        cold=memory.cold,
        project_id="my_project",
    )

Author: Jeffrey Dean
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ForgettingConfig:
    """Configuration for the forgetting policy.

    Tune per project type:
    - Programming assistant: longer retention, access matters most
    - Language tutor: importance matters most (corrections are gold)
    - Voice interface: shorter retention, recency matters most
    """
    # Retention threshold: episodes scoring below this get archived
    retention_threshold: float = 0.3

    # Weight for each factor (must sum to 1.0)
    weight_recency: float = 0.35
    weight_importance: float = 0.30
    weight_access: float = 0.25
    weight_surprise: float = 0.10

    # Recency decay: half-life in days
    recency_half_life_days: float = 30.0

    # Minimum age before an episode can be archived (days)
    # Prevents archiving fresh episodes that haven't had time to be accessed
    min_age_days: float = 7.0

    # Maximum episodes to archive per run (prevents huge batch operations)
    max_archive_per_run: int = 100

    # If episodic memory has fewer than this many episodes, skip archival
    # (no point archiving when the index is small)
    min_episodes_before_archival: int = 50

    # Auto-run: trigger archival after this many new episodes since last run
    auto_trigger_interval: int = 50

    # Enable/disable
    enabled: bool = True


@dataclass
class RetentionScore:
    """Breakdown of an episode's retention score."""
    episode_id: str
    total: float
    recency: float
    importance: float
    access: float
    surprise: float
    age_days: float
    access_count: int


class AccessTracker:
    """SQLite-based access tracking for episodic memory.

    Records when episodes are retrieved via search, enabling
    access-frequency-based retention scoring.

    Thread-safe via per-thread connections.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def _init_schema(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                episode_id TEXT PRIMARY KEY,
                access_count INTEGER NOT NULL DEFAULT 0,
                first_accessed REAL NOT NULL,
                last_accessed REAL NOT NULL
            )
        """)
        conn.commit()

    def record_access(self, episode_ids: List[str]):
        """Record that episodes were retrieved by a search."""
        if not episode_ids:
            return
        now = time.time()
        conn = self._get_conn()
        for eid in episode_ids:
            conn.execute("""
                INSERT INTO access_log (episode_id, access_count, first_accessed, last_accessed)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                    access_count = access_count + 1,
                    last_accessed = ?
            """, (eid, now, now, now))
        conn.commit()

    def get_access_info(self, episode_ids: List[str]) -> Dict[str, Tuple[int, float]]:
        """Get (access_count, last_accessed) for a batch of episodes.

        Returns dict mapping episode_id → (count, last_accessed).
        Missing episodes get (0, 0.0).
        """
        if not episode_ids:
            return {}
        conn = self._get_conn()
        result = {}
        # Batch in chunks to avoid SQLite variable limit
        for chunk_start in range(0, len(episode_ids), 500):
            chunk = episode_ids[chunk_start:chunk_start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT episode_id, access_count, last_accessed FROM access_log "
                f"WHERE episode_id IN ({placeholders})",
                chunk,
            ).fetchall()
            for eid, count, last in rows:
                result[eid] = (count, last)
        return result

    def delete_episodes(self, episode_ids: List[str]):
        """Remove access records for archived episodes."""
        if not episode_ids:
            return
        conn = self._get_conn()
        for chunk_start in range(0, len(episode_ids), 500):
            chunk = episode_ids[chunk_start:chunk_start + 500]
            placeholders = ",".join("?" * len(chunk))
            conn.execute(
                f"DELETE FROM access_log WHERE episode_id IN ({placeholders})",
                chunk,
            )
        conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(access_count), 0), "
            "COALESCE(MAX(access_count), 0) FROM access_log"
        ).fetchone()
        return {
            "tracked_episodes": row[0],
            "total_accesses": row[1],
            "max_access_count": row[2],
        }

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None


class ForgettingPolicy:
    """Manages episodic → cold storage lifecycle.

    Scores episodes by retention value and archives those below threshold.
    Call `run()` periodically or after a batch of new episodes.
    """

    def __init__(
        self,
        access_db_path: Path,
        config: Optional[ForgettingConfig] = None,
    ):
        self.config = config or ForgettingConfig()
        self.tracker = AccessTracker(access_db_path)
        self._episodes_since_last_run = 0

    def record_access(self, episode_ids: List[str]):
        """Record search hits. Call after episodic search."""
        self.tracker.record_access(episode_ids)

    def record_new_episode(self):
        """Track that a new episode was stored. May trigger auto-run."""
        self._episodes_since_last_run += 1

    def should_auto_run(self) -> bool:
        """Check if auto-run threshold is met."""
        return (
            self.config.enabled
            and self._episodes_since_last_run >= self.config.auto_trigger_interval
        )

    def score_episode(
        self,
        episode_id: str,
        timestamp: float,
        importance: float,
        access_count: int,
        max_access_count: int,
        surprise: float = 0.0,
    ) -> RetentionScore:
        """Compute retention score for a single episode.

        Returns RetentionScore with breakdown.
        """
        now = time.time()
        age_s = max(0.0, now - timestamp)
        age_days = age_s / 86400.0
        cfg = self.config

        # Recency: exponential decay with configurable half-life
        decay_rate = math.log(2) / max(cfg.recency_half_life_days, 0.1)
        recency = math.exp(-decay_rate * age_days)

        # Importance: direct pass-through (already 0-1)
        imp = max(0.0, min(1.0, importance))

        # Access frequency: log-scaled relative to max
        if max_access_count > 0 and access_count > 0:
            access = math.log1p(access_count) / math.log1p(max_access_count)
        else:
            # No accesses: give a baseline score so fresh episodes aren't penalized
            access = 0.3 if age_days < cfg.min_age_days else 0.0

        # Surprise: direct (already 0-1 after normalization)
        surp = max(0.0, min(1.0, surprise))

        total = (
            cfg.weight_recency * recency
            + cfg.weight_importance * imp
            + cfg.weight_access * access
            + cfg.weight_surprise * surp
        )

        return RetentionScore(
            episode_id=episode_id,
            total=total,
            recency=recency,
            importance=imp,
            access=access,
            surprise=surp,
            age_days=age_days,
            access_count=access_count,
        )

    def score_all(self, episodic_memory, project_id: str) -> List[RetentionScore]:
        """Score all episodes in episodic memory.

        Uses ``include=["metadatas"]`` so ChromaDB returns only what scoring
        needs — no document text or embedding vectors are loaded.  Episodes
        are fetched in batches to bound peak memory usage.

        Args:
            episodic_memory: EpisodicMemory instance
            project_id: Project to scope scoring

        Returns list of RetentionScores, sorted by total (ascending = lowest first).
        """
        import json as _json

        BATCH = 500
        all_rows: List[Dict[str, Any]] = []
        offset = 0

        while True:
            batch = episodic_memory.get_metadata_batch(
                project_id, limit=BATCH, offset=offset
            )
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < BATCH:
                break
            offset += BATCH

        if not all_rows:
            return []

        all_ids = [r["id"] for r in all_rows]

        # Get access info in one batched SQLite call
        access_info = self.tracker.get_access_info(all_ids)
        max_access = max((v[0] for v in access_info.values()), default=0)

        scores = []
        for row in all_rows:
            eid = row["id"]
            timestamp = row["timestamp"]
            importance = row["importance"]
            inner_meta = row.get("metadata", "{}")
            if isinstance(inner_meta, str):
                try:
                    inner_meta = _json.loads(inner_meta)
                except Exception:
                    inner_meta = {}
            surprise = inner_meta.get("neural_surprise", 0.0)
            ac, _ = access_info.get(eid, (0, 0.0))

            scores.append(self.score_episode(
                episode_id=eid,
                timestamp=timestamp,
                importance=importance,
                access_count=ac,
                max_access_count=max_access,
                surprise=surprise,
            ))

        scores.sort(key=lambda s: s.total)
        return scores

    def run(
        self,
        episodic_memory,
        cold_storage,
        project_id: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute the forgetting policy.

        Scores all episodes, archives those below threshold to cold storage,
        and deletes them from episodic memory.

        Args:
            episodic_memory: EpisodicMemory instance
            cold_storage: ColdStorage instance
            project_id: Project to process
            dry_run: If True, score and report but don't archive/delete.

        Returns dict with stats about the run.
        """
        if not self.config.enabled:
            return {"status": "disabled"}

        # Check minimum episode count
        stats = episodic_memory.get_stats()
        total_episodes = stats.get("total_episodes", 0)
        if total_episodes < self.config.min_episodes_before_archival:
            return {
                "status": "skipped",
                "reason": f"Only {total_episodes} episodes (min: {self.config.min_episodes_before_archival})",
                "total_episodes": total_episodes,
            }

        # Score all episodes
        scores = self.score_all(episodic_memory, project_id)

        # Filter: below threshold AND old enough
        candidates = [
            s for s in scores
            if s.total < self.config.retention_threshold
            and s.age_days >= self.config.min_age_days
        ]

        # Cap per run
        to_archive = candidates[:self.config.max_archive_per_run]

        if dry_run:
            return {
                "status": "dry_run",
                "total_scored": len(scores),
                "below_threshold": len(candidates),
                "would_archive": len(to_archive),
                "lowest_scores": [
                    {"id": s.episode_id, "score": round(s.total, 4),
                     "age_days": round(s.age_days, 1), "accesses": s.access_count}
                    for s in to_archive[:10]
                ],
                "score_distribution": _score_distribution(scores),
            }

        if not to_archive:
            self._episodes_since_last_run = 0
            return {
                "status": "ok",
                "archived": 0,
                "total_scored": len(scores),
                "below_threshold": len(candidates),
            }

        # Fetch full episode data for archival
        archive_ids = [s.episode_id for s in to_archive]
        episodes = episodic_memory.get_by_ids(archive_ids)

        # Build cold storage records
        cold_records = []
        for ep in episodes:
            cold_records.append({
                "id": ep.id,
                "timestamp": ep.timestamp,
                "project_id": ep.project_id,
                "session_id": ep.session_id,
                "text": ep.text,
                "metadata": {
                    "importance": ep.importance,
                    "original_metadata": ep.metadata,
                    "archived_at": time.time(),
                    "archive_reason": "forgetting_policy",
                },
            })

        # Archive to cold storage
        archived_count = cold_storage.archive(cold_records)

        # Delete from episodic
        episodic_memory.delete_episodes(archive_ids)

        # Clean up access tracker
        self.tracker.delete_episodes(archive_ids)

        self._episodes_since_last_run = 0

        logger.info(
            "Forgetting policy: archived %d/%d episodes (threshold=%.2f) for project %s",
            archived_count, len(scores), self.config.retention_threshold, project_id,
        )

        return {
            "status": "ok",
            "archived": archived_count,
            "deleted_from_episodic": len(archive_ids),
            "total_scored": len(scores),
            "below_threshold": len(candidates),
            "score_distribution": _score_distribution(scores),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "retention_threshold": self.config.retention_threshold,
            "episodes_since_last_run": self._episodes_since_last_run,
            "auto_trigger_at": self.config.auto_trigger_interval,
            "access_tracker": self.tracker.get_stats(),
        }

    def close(self):
        self.tracker.close()


def _score_distribution(scores: List[RetentionScore]) -> Dict[str, Any]:
    """Summary statistics for a list of retention scores."""
    if not scores:
        return {}
    totals = [s.total for s in scores]
    return {
        "count": len(totals),
        "min": round(min(totals), 4),
        "max": round(max(totals), 4),
        "mean": round(sum(totals) / len(totals), 4),
        "median": round(sorted(totals)[len(totals) // 2], 4),
        "below_0.3": sum(1 for t in totals if t < 0.3),
        "below_0.5": sum(1 for t in totals if t < 0.5),
    }
