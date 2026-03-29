"""Tests for EpisodicMemory (Layer 2) — ChromaDB + SentenceTransformers."""

from __future__ import annotations

import time

from .runner import test_group, require
from .mocks import TempDir, FakeEmbeddingModel


def _make_episodic(persist_dir=None):
    require("chromadb", "sentence_transformers")
    from engram.memory.episodic_memory import EpisodicMemory
    return EpisodicMemory(
        persist_dir=persist_dir,
        collection_name="test_episodes",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_device="cpu",
    )


# ── tests ─────────────────────────────────────────────────────────────────────

@test_group("Episodic Memory")
def test_episodic_add_and_search():
    require("chromadb", "sentence_transformers")
    with TempDir() as d:
        em = _make_episodic(persist_dir=d / "episodic")
        em.add_episode("User prefers pytest over unittest", project_id="test")
        em.add_episode("User uses type hints throughout their code", project_id="test")

        results = em.search("testing framework", n=5, project_id="test")
        assert len(results) >= 1
        assert any("pytest" in r.text for r in results)


@test_group("Episodic Memory")
def test_episodic_semantic_search():
    """Semantic search (not just keyword) surfaces related content."""
    require("chromadb", "sentence_transformers")
    with TempDir() as d:
        em = _make_episodic(d / "ep")
        em.add_episode("User finds async patterns confusing", project_id="p")
        em.add_episode("The weather is nice today", project_id="p")

        # Query is semantically related to first episode, not second
        results = em.search("concurrency and coroutines", n=3, project_id="p")
        assert len(results) >= 1
        assert any("async" in r.text.lower() for r in results)


@test_group("Episodic Memory")
def test_episodic_project_isolation():
    """Episodes from different projects are not mixed."""
    require("chromadb", "sentence_transformers")
    with TempDir() as d:
        em = _make_episodic(d / "ep")
        em.add_episode("Project A prefers Django", project_id="project_a")
        em.add_episode("Project B prefers FastAPI", project_id="project_b")

        results_a = em.search("web framework", n=10, project_id="project_a")
        results_b = em.search("web framework", n=10, project_id="project_b")

        texts_a = [r.text for r in results_a]
        texts_b = [r.text for r in results_b]

        assert any("Django" in t for t in texts_a), "Expected Django in project_a"
        assert not any("FastAPI" in t for t in texts_a), "FastAPI leaked into project_a"
        assert any("FastAPI" in t for t in texts_b), "Expected FastAPI in project_b"
        assert not any("Django" in t for t in texts_b), "Django leaked into project_b"


@test_group("Episodic Memory")
def test_episodic_importance_stored():
    """Importance score is preserved through storage/retrieval."""
    require("chromadb", "sentence_transformers")
    with TempDir() as d:
        em = _make_episodic(d / "ep")
        em.add_episode("Critical preference", project_id="p", importance=0.95)

        results = em.search("Critical preference", n=5, project_id="p")
        assert len(results) >= 1
        assert results[0].importance == 0.95  # importance stored in chromadb metadata


@test_group("Episodic Memory")
def test_episodic_get_recent():
    """get_recent returns most recently added episodes."""
    require("chromadb", "sentence_transformers")
    with TempDir() as d:
        em = _make_episodic(d / "ep")
        for i in range(5):
            em.add_episode(f"Episode {i}", project_id="p")
            time.sleep(0.01)

        recent = em.get_recent_episodes(n=3, project_id="p")
        assert len(recent) == 3


@test_group("Episodic Memory")
def test_episodic_persistence():
    """Episodes survive close and reopen of EpisodicMemory."""
    require("chromadb", "sentence_transformers")
    with TempDir() as d:
        ep_dir = d / "ep"
        from engram.memory.episodic_memory import EpisodicMemory

        em1 = EpisodicMemory(persist_dir=ep_dir, collection_name="persist_test",
                             embedding_device="cpu")
        em1.add_episode("Persistent episode", project_id="p")

        em2 = EpisodicMemory(persist_dir=ep_dir, collection_name="persist_test",
                             embedding_device="cpu")
        results = em2.search("Persistent episode", n=5, project_id="p")
        assert len(results) >= 1
        assert results[0].text == "Persistent episode"


@test_group("Episodic Memory")
def test_episodic_embedding_cache_shared():
    """EpisodicMemory accepts and uses a shared EmbeddingCache."""
    require("chromadb", "sentence_transformers")
    from engram.memory.embedding_cache import EmbeddingCache
    from engram.memory.episodic_memory import EpisodicMemory

    with TempDir() as d:
        cache = EmbeddingCache(cache_dir=None, enabled=True)
        em = EpisodicMemory(
            persist_dir=d / "ep",
            collection_name="cache_test",
            embedding_device="cpu",
            embedding_cache=cache,
        )
        em.add_episode("Cache test episode", project_id="p")
        results = em.search("Cache test", n=5, project_id="p")
        assert len(results) >= 1


@test_group("Episodic Memory")
def test_episodic_empty_search():
    """Search on empty collection returns empty list, not exception."""
    require("chromadb", "sentence_transformers")
    with TempDir() as d:
        em = _make_episodic(d / "ep")
        results = em.search("anything", n=5, project_id="p")
        assert results == []


@test_group("Episodic Memory")
def test_episodic_delete():
    """Deleted episodes no longer appear in search results."""
    require("chromadb", "sentence_transformers")
    with TempDir() as d:
        em = _make_episodic(d / "ep")
        ep_id = em.add_episode("Delete me", project_id="p")

        results_before = em.search("Delete me", n=5, project_id="p")
        assert len(results_before) >= 1

        em.delete_episode(ep_id)
        results_after = em.search("Delete me", n=5, project_id="p")
        assert not any(r.id == ep.id for r in results_after)
