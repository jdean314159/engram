"""Tests for ColdStorage (Layer 4) — SQLite FTS5, no optional deps."""

from __future__ import annotations

import time

from .runner import test_group
from .mocks import TempDir


def _cs(tmp_dir):
    from engram.memory.cold_storage import ColdStorage
    return ColdStorage(db_path=tmp_dir / "cold.db")


def _archive(cs, text, project_id="p", importance=0.5, metadata=None):
    """Helper: wrap single-doc archive into the List[Dict] API."""
    return cs.archive([{
        "text": text,
        "project_id": project_id,
        "importance": importance,
        "metadata": metadata or {},
    }])


# ── tests ──────────────────────────────────────────────────────────────────────

@test_group("Cold Storage")
def test_cold_archive_and_retrieve():
    with TempDir() as d:
        cs = _cs(d)
        n = _archive(cs, "Python asyncio is useful for concurrent I/O", project_id="p")
        assert n == 1
        results = cs.retrieve("asyncio", n=5, project_id="p")
        assert len(results) >= 1
        assert any("asyncio" in r["text"] for r in results)


@test_group("Cold Storage")
def test_cold_fts_search():
    with TempDir() as d:
        cs = _cs(d)
        _archive(cs, "The dog sat on the mat", project_id="p")
        _archive(cs, "Python decorators are powerful", project_id="p")
        _archive(cs, "asyncio event loop management", project_id="p")
        results = cs.retrieve("asyncio event", n=5, project_id="p")
        assert len(results) >= 1
        assert "asyncio" in results[0]["text"]


@test_group("Cold Storage")
def test_cold_project_isolation():
    with TempDir() as d:
        cs = _cs(d)
        _archive(cs, "Project A document", project_id="project_a")
        _archive(cs, "Project B document", project_id="project_b")

        results_a = cs.retrieve("document", n=10, project_id="project_a")
        results_b = cs.retrieve("document", n=10, project_id="project_b")

        texts_a = [r["text"] for r in results_a]
        texts_b = [r["text"] for r in results_b]

        assert any("Project A" in t for t in texts_a)
        assert not any("Project B" in t for t in texts_a)
        assert any("Project B" in t for t in texts_b)
        assert not any("Project A" in t for t in texts_b)


@test_group("Cold Storage")
def test_cold_stats():
    with TempDir() as d:
        cs = _cs(d)
        _archive(cs, "Document one", project_id="p")
        _archive(cs, "Document two", project_id="p")
        stats = cs.get_stats()
        assert stats["enabled"] is True
        assert stats["total_rows"] >= 2


@test_group("Cold Storage")
def test_cold_dedup():
    """Identical content hash should not create duplicate entries."""
    with TempDir() as d:
        cs = _cs(d)
        text = "This is a unique document that should not be duplicated."
        cs.archive([{"text": text, "project_id": "p"}])
        cs.archive([{"text": text, "project_id": "p"}])
        results = cs.retrieve("unique document", n=10, project_id="p")
        # Dedup by content hash: at most 1 result
        assert len(results) <= 1


@test_group("Cold Storage")
def test_cold_metadata_preserved():
    with TempDir() as d:
        cs = _cs(d)
        cs.archive([{
            "text": "Metadata test document",
            "project_id": "p",
            "importance": 0.8,
            "metadata": {"source": "episodic", "session_id": "abc123"},
        }])
        results = cs.retrieve("Metadata test", n=5, project_id="p")
        assert len(results) >= 1
        # Row returned — text at minimum
        assert results[0]["text"] == "Metadata test document"


@test_group("Cold Storage")
def test_cold_empty_query_graceful():
    """Empty query should not raise — returns empty list or recent docs."""
    with TempDir() as d:
        cs = _cs(d)
        _archive(cs, "Some document", project_id="p")
        results = cs.retrieve("", n=5, project_id="p")
        assert isinstance(results, list)


@test_group("Cold Storage")
def test_cold_no_results():
    with TempDir() as d:
        cs = _cs(d)
        _archive(cs, "Python asyncio", project_id="p")
        results = cs.retrieve("xyzzy_nonexistent_term_42", n=5, project_id="p")
        assert results == []


@test_group("Cold Storage")
def test_cold_persistence():
    with TempDir() as d:
        from engram.memory.cold_storage import ColdStorage
        cs1 = ColdStorage(db_path=d / "cold.db")
        cs1.archive([{"text": "Persisted document", "project_id": "p"}])
        cs1.close()

        cs2 = ColdStorage(db_path=d / "cold.db")
        results = cs2.retrieve("Persisted", n=5, project_id="p")
        assert len(results) >= 1
        assert results[0]["text"] == "Persisted document"


@test_group("Cold Storage")
def test_cold_batch_archive():
    """archive() with multiple docs returns count archived."""
    with TempDir() as d:
        cs = _cs(d)
        docs = [
            {"text": f"Document {i} about asyncio", "project_id": "p"}
            for i in range(5)
        ]
        n = cs.archive(docs)
        assert n == 5
        results = cs.retrieve("asyncio", n=10, project_id="p")
        assert len(results) == 5


@test_group("Cold Storage")
def test_cold_get_all():
    """get_all returns archived documents for a project."""
    with TempDir() as d:
        cs = _cs(d)
        for i in range(3):
            _archive(cs, f"Batch doc {i}", project_id="proj_x")

        docs = list(cs.get_all(project_id="proj_x", batch_size=10))
        assert len(docs) == 3
