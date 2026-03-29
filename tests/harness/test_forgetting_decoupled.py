"""Validation tests for P1 Bug #2 — ForgettingPolicy decoupled from ChromaDB.

Tests verify:
1. The three new EpisodicMemory abstraction methods work correctly.
2. ForgettingPolicy no longer touches .collection directly — runs against
   a pure-Python fake episodic backend with no ChromaDB installed.
3. The full archive cycle (score → archive → delete) works end-to-end
   against a real EpisodicMemory when chromadb IS available.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .runner import test_group, require
from .mocks import TempDir, unique_session


# ── Fake episodic backend (no ChromaDB) ───────────────────────────────────────

class _FakeEpisodic:
    """Pure-Python episodic backend satisfying the new abstraction surface.

    Deliberately does NOT have a .collection attribute — proves the policy
    no longer requires ChromaDB internals.
    """

    def __init__(self):
        self._episodes: Dict[str, Dict] = {}  # id → episode dict

    def add_episode(self, text: str, *, project_id: str, importance: float = 0.5,
                    timestamp: Optional[float] = None,
                    metadata: Optional[Dict] = None,
                    session_id: Optional[str] = None,
                    episode_id: Optional[str] = None) -> str:
        """Match real EpisodicMemory.add_episode() signature — returns str id."""
        import uuid
        eid = episode_id or str(uuid.uuid4())
        self._episodes[eid] = {
            "id": eid,
            "text": text,
            "project_id": project_id,
            "session_id": session_id or "test",
            "importance": importance,
            "timestamp": timestamp or time.time(),
            "metadata": metadata or {},
        }
        return eid

    def add(self, text: str, *, project_id: str, importance: float = 0.5,
            timestamp: Optional[float] = None) -> str:
        """Legacy alias — delegates to add_episode()."""
        return self.add_episode(text, project_id=project_id,
                                importance=importance, timestamp=timestamp)

    def get_stats(self) -> Dict[str, Any]:
        return {"total_episodes": len(self._episodes)}

    def get_metadata_batch(self, project_id: str, limit: int = 500,
                           offset: int = 0) -> List[Dict]:
        rows = [
            {
                "id": ep["id"],
                "timestamp": ep["timestamp"],
                "importance": ep["importance"],
                "metadata": ep.get("metadata", {}),
            }
            for ep in self._episodes.values()
            if ep["project_id"] == project_id
        ]
        return rows[offset: offset + limit]

    def get_by_ids(self, ids: List[str]) -> List[Any]:
        from dataclasses import dataclass

        @dataclass
        class Ep:
            id: str
            text: str
            timestamp: float
            project_id: str
            session_id: str
            importance: float
            metadata: dict

        return [
            Ep(**{k: v for k, v in self._episodes[eid].items()})
            for eid in ids if eid in self._episodes
        ]

    def delete_episodes(self, ids: List[str]) -> int:
        for eid in ids:
            self._episodes.pop(eid, None)
        return len(ids)

    def count(self, project_id: Optional[str] = None) -> int:
        if project_id is None:
            return len(self._episodes)
        return sum(1 for ep in self._episodes.values()
                   if ep["project_id"] == project_id)


# ── EpisodicMemory new abstraction methods ─────────────────────────────────────

@test_group("P1-2: ForgettingPolicy Decoupling")
def test_episodic_get_metadata_batch_returns_correct_fields():
    require("chromadb", "sentence_transformers")
    from engram.memory.episodic_memory import EpisodicMemory
    with TempDir() as d:
        em = EpisodicMemory(persist_dir=d / "ep", embedding_device="cpu")
        em.add_episode("Python asyncio tutorial", project_id="p", importance=0.8)
        em.add_episode("FastAPI REST patterns", project_id="p", importance=0.6)

        rows = em.get_metadata_batch("p", limit=10, offset=0)
        assert len(rows) == 2
        for row in rows:
            for field in ("id", "timestamp", "importance", "metadata"):
                assert field in row, f"Missing field {field!r}"
            assert isinstance(row["id"], str)
            assert isinstance(row["importance"], float)


@test_group("P1-2: ForgettingPolicy Decoupling")
def test_episodic_get_metadata_batch_project_isolation():
    require("chromadb", "sentence_transformers")
    from engram.memory.episodic_memory import EpisodicMemory
    with TempDir() as d:
        em = EpisodicMemory(persist_dir=d / "ep", embedding_device="cpu")
        em.add_episode("Episode A", project_id="proj_a")
        em.add_episode("Episode B", project_id="proj_b")

        rows_a = em.get_metadata_batch("proj_a", limit=10)
        rows_b = em.get_metadata_batch("proj_b", limit=10)

        ids_a = {r["id"] for r in rows_a}
        ids_b = {r["id"] for r in rows_b}
        assert not ids_a & ids_b, "Project metadata leaked across projects"


@test_group("P1-2: ForgettingPolicy Decoupling")
def test_episodic_get_metadata_batch_pagination():
    require("chromadb", "sentence_transformers")
    from engram.memory.episodic_memory import EpisodicMemory
    with TempDir() as d:
        em = EpisodicMemory(persist_dir=d / "ep", embedding_device="cpu")
        for i in range(5):
            em.add_episode(f"Episode {i}", project_id="p")

        page1 = em.get_metadata_batch("p", limit=3, offset=0)
        page2 = em.get_metadata_batch("p", limit=3, offset=3)
        all_ids = {r["id"] for r in page1} | {r["id"] for r in page2}
        assert len(page1) == 3
        assert len(page2) == 2
        assert len(all_ids) == 5  # no overlaps


@test_group("P1-2: ForgettingPolicy Decoupling")
def test_episodic_get_by_ids_returns_episodes():
    require("chromadb", "sentence_transformers")
    from engram.memory.episodic_memory import EpisodicMemory, Episode
    with TempDir() as d:
        em = EpisodicMemory(persist_dir=d / "ep", embedding_device="cpu")
        ep1_id = em.add_episode("Hello asyncio", project_id="p", importance=0.9)
        ep2_id = em.add_episode("Hello FastAPI", project_id="p", importance=0.7)

        episodes = em.get_by_ids([ep1_id, ep2_id])
        assert len(episodes) == 2
        texts = {ep.text for ep in episodes}
        assert "Hello asyncio" in texts
        assert "Hello FastAPI" in texts
        for ep in episodes:
            assert isinstance(ep, Episode)


@test_group("P1-2: ForgettingPolicy Decoupling")
def test_episodic_get_by_ids_empty_list():
    require("chromadb", "sentence_transformers")
    from engram.memory.episodic_memory import EpisodicMemory
    with TempDir() as d:
        em = EpisodicMemory(persist_dir=d / "ep", embedding_device="cpu")
        result = em.get_by_ids([])
        assert result == []


@test_group("P1-2: ForgettingPolicy Decoupling")
def test_episodic_delete_episodes_batch():
    require("chromadb", "sentence_transformers")
    from engram.memory.episodic_memory import EpisodicMemory
    with TempDir() as d:
        em = EpisodicMemory(persist_dir=d / "ep", embedding_device="cpu")
        ep_ids = [em.add_episode(f"Episode {i}", project_id="p") for i in range(4)]
        ids_to_delete = [ep_ids[0], ep_ids[1]]

        n = em.delete_episodes(ids_to_delete)
        assert n == 2

        stats = em.get_stats()
        assert stats["total_episodes"] == 2


@test_group("P1-2: ForgettingPolicy Decoupling")
def test_episodic_delete_episodes_empty():
    require("chromadb", "sentence_transformers")
    from engram.memory.episodic_memory import EpisodicMemory
    with TempDir() as d:
        em = EpisodicMemory(persist_dir=d / "ep", embedding_device="cpu")
        n = em.delete_episodes([])
        assert n == 0


# ── ForgettingPolicy against fake backend (no ChromaDB required) ───────────────

@test_group("P1-2: ForgettingPolicy Decoupling")
def test_forgetting_score_all_no_chromadb():
    """score_all runs against a pure-Python backend — proves no .collection use."""
    from engram.memory.forgetting import ForgettingPolicy, ForgettingConfig
    with TempDir() as d:
        config = ForgettingConfig(min_episodes_before_archival=0)
        policy = ForgettingPolicy(access_db_path=d / "acc.db", config=config)
        episodic = _FakeEpisodic()

        old_ts = time.time() - 60 * 86400  # 60 days ago
        episodic.add("Old low-importance note", project_id="p",
                     importance=0.1, timestamp=old_ts)
        episodic.add("Recent important fact", project_id="p",
                     importance=0.9)

        scores = policy.score_all(episodic, "p")
        assert len(scores) == 2
        # Lowest retention score comes first
        assert scores[0].total <= scores[1].total


@test_group("P1-2: ForgettingPolicy Decoupling")
def test_forgetting_run_dry_run_no_chromadb():
    """dry_run works against fake backend — reports candidates without deleting."""
    from engram.memory.forgetting import ForgettingPolicy, ForgettingConfig
    from engram.memory.cold_storage import ColdStorage
    with TempDir() as d:
        config = ForgettingConfig(
            min_episodes_before_archival=0,
            retention_threshold=1.0,  # archive everything
            min_age_days=0.0,
        )
        policy = ForgettingPolicy(access_db_path=d / "acc.db", config=config)
        cold = ColdStorage(db_path=d / "cold.db")
        episodic = _FakeEpisodic()
        for i in range(3):
            episodic.add_episode(f"Episode {i}", project_id="p", importance=0.1)

        result = policy.run(episodic, cold, project_id="p", dry_run=True)
        assert result["status"] == "dry_run"
        assert result["total_scored"] == 3
        assert result["would_archive"] == 3
        # Episodes NOT deleted
        assert episodic.count("p") == 3


@test_group("P1-2: ForgettingPolicy Decoupling")
def test_forgetting_run_archives_and_deletes_no_chromadb():
    """Full archive cycle runs against fake backend — episodes moved to cold."""
    from engram.memory.forgetting import ForgettingPolicy, ForgettingConfig
    from engram.memory.cold_storage import ColdStorage
    with TempDir() as d:
        config = ForgettingConfig(
            min_episodes_before_archival=0,
            retention_threshold=1.0,
            min_age_days=0.0,
        )
        policy = ForgettingPolicy(access_db_path=d / "acc.db", config=config)
        cold = ColdStorage(db_path=d / "cold.db")
        episodic = _FakeEpisodic()

        old_ts = time.time() - 10  # old enough
        for i in range(3):
            episodic.add(f"Archivable episode {i}", project_id="p",
                         importance=0.1, timestamp=old_ts)

        result = policy.run(episodic, cold, project_id="p")
        assert result["status"] == "ok"
        assert result["archived"] == 3
        assert result["deleted_from_episodic"] == 3
        assert episodic.count("p") == 0

        # Verify content landed in cold storage
        cold_results = cold.retrieve("Archivable episode", n=10, project_id="p")
        assert len(cold_results) == 3


@test_group("P1-2: ForgettingPolicy Decoupling")
def test_forgetting_skips_below_minimum():
    """run() returns 'skipped' when episode count < min_episodes_before_archival."""
    from engram.memory.forgetting import ForgettingPolicy, ForgettingConfig
    from engram.memory.cold_storage import ColdStorage
    with TempDir() as d:
        config = ForgettingConfig(min_episodes_before_archival=50)
        policy = ForgettingPolicy(access_db_path=d / "acc.db", config=config)
        cold = ColdStorage(db_path=d / "cold.db")
        episodic = _FakeEpisodic()
        episodic.add("Just one episode", project_id="p")

        result = policy.run(episodic, cold, project_id="p")
        assert result["status"] == "skipped"
        assert episodic.count("p") == 1  # untouched


@test_group("P1-2: ForgettingPolicy Decoupling")
def test_forgetting_respects_min_age():
    """Very recent episodes are not archived even when below retention threshold."""
    from engram.memory.forgetting import ForgettingPolicy, ForgettingConfig
    from engram.memory.cold_storage import ColdStorage
    with TempDir() as d:
        config = ForgettingConfig(
            min_episodes_before_archival=0,
            retention_threshold=1.0,
            min_age_days=7.0,   # must be 7+ days old
        )
        policy = ForgettingPolicy(access_db_path=d / "acc.db", config=config)
        cold = ColdStorage(db_path=d / "cold.db")
        episodic = _FakeEpisodic()

        # Fresh episode — added just now
        episodic.add("New episode", project_id="p", importance=0.01)

        result = policy.run(episodic, cold, project_id="p")
        assert result.get("archived", 0) == 0
        assert episodic.count("p") == 1  # untouched


# ── End-to-end with real EpisodicMemory ───────────────────────────────────────

@test_group("P1-2: ForgettingPolicy Decoupling")
def test_forgetting_end_to_end_real_episodic():
    """Full cycle against real EpisodicMemory (chromadb required)."""
    require("chromadb", "sentence_transformers")
    from engram.memory.forgetting import ForgettingPolicy, ForgettingConfig
    from engram.memory.episodic_memory import EpisodicMemory
    from engram.memory.cold_storage import ColdStorage

    with TempDir() as d:
        config = ForgettingConfig(
            min_episodes_before_archival=0,
            retention_threshold=1.0,
            min_age_days=0.0,
        )
        policy = ForgettingPolicy(access_db_path=d / "acc.db", config=config)
        em = EpisodicMemory(persist_dir=d / "ep", embedding_device="cpu")
        cold = ColdStorage(db_path=d / "cold.db")

        for i in range(3):
            em.add_episode(f"Episode to archive {i}", project_id="p", importance=0.1)

        result = policy.run(em, cold, project_id="p")
        assert result["status"] == "ok"
        assert result["archived"] == 3
        assert em.get_stats()["total_episodes"] == 0

        cold_results = cold.retrieve("archive", n=10, project_id="p")
        assert len(cold_results) == 3
