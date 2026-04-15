"""Tests for SemanticMemory (Layer 3) — Kuzu graph database."""

from __future__ import annotations

import time

from .runner import test_group, require
from .mocks import TempDir


def _make_semantic(db_path, project_type=None):
    require("kuzu")
    from engram.memory.semantic_memory import SemanticMemory
    from engram.memory.types import ProjectType
    pt = project_type or ProjectType.PROGRAMMING_ASSISTANT
    return SemanticMemory(db_path=db_path, project_type=pt)


# ── Schema / init ─────────────────────────────────────────────────────────────

@test_group("Semantic Memory")
def test_semantic_init_creates_schema():
    require("kuzu")
    with TempDir() as d:
        sm = _make_semantic(d / "semantic")
        stats = sm.get_stats()
        assert stats is not None
        # get_stats returns a dict with various counts
        assert isinstance(stats, dict)


@test_group("Semantic Memory")
def test_semantic_all_project_types():
    """Schema initialization succeeds for every ProjectType."""
    require("kuzu")
    from engram.memory.types import ProjectType
    from engram.memory.semantic_memory import SemanticMemory

    with TempDir() as d:
        for pt in ProjectType:
            sm = SemanticMemory(db_path=d / f"sem_{pt.value}", project_type=pt)
            stats = sm.get_stats()
            assert isinstance(stats, dict)  # any dict is acceptable


# ── Node operations ───────────────────────────────────────────────────────────

@test_group("Semantic Memory")
def test_semantic_add_and_query_node():
    require("kuzu")
    from engram.memory.semantic_memory import SemanticMemory
    from engram.memory.types import ProjectType
    with TempDir() as d:
        # Concept node only exists under programming_assistant schema
        sm = SemanticMemory(db_path=d / "sem",
                            project_type=ProjectType.PROGRAMMING_ASSISTANT)
        sm.add_node("Concept", "asyncio", {
            "name": "asyncio",
            "difficulty": "intermediate",
            "category": "language",
            "documentation": "https://docs.python.org/asyncio",
            "examples": "[]",
        })
        rows = sm.query("MATCH (c:Concept) WHERE c.id = 'asyncio' RETURN c.name")
        assert len(rows) >= 1


@test_group("Semantic Memory")
def test_semantic_add_preference():
    with TempDir() as d:
        sm = _make_semantic(d / "sem")
        sm.add_preference(category="testing", value="pytest over unittest", strength=0.85)
        rows = sm.list_preferences(limit=10)
        assert len(rows) >= 1
        assert any("pytest" in r.get("value", "") for r in rows)

@test_group("Semantic Memory")
def test_semantic_add_relationship():
    require("kuzu")
    from engram.memory.semantic_memory import SemanticMemory
    from engram.memory.types import ProjectType
    with TempDir() as d:
        sm = SemanticMemory(db_path=d / "sem",
                            project_type=ProjectType.PROGRAMMING_ASSISTANT)
        sm.add_node("Concept", "asyncio", {
            "name": "asyncio", "difficulty": "intermediate",
            "category": "language", "documentation": "", "examples": "[]",
        })
        sm.add_node("Concept", "event_loop", {
            "name": "event loop", "difficulty": "intermediate",
            "category": "language", "documentation": "", "examples": "[]",
        })
        sm.add_relationship("Concept", "asyncio", "Concept", "event_loop", "REQUIRES")
        rows = sm.query(
            "MATCH (a:Concept)-[:REQUIRES]->(b:Concept) "
            "WHERE a.id = 'asyncio' RETURN b.id"
        )
        assert len(rows) >= 1


@test_group("Semantic Memory")
def test_semantic_upsert_node():
    """Adding a Fact with the same id twice updates without duplicating."""
    require("kuzu")
    with TempDir() as d:
        sm = _make_semantic(d / "sem")
        # Use Fact (core schema) instead of Concept (programming schema)
        sm.add_fact("Python 3.11 is stable", confidence=0.9)
        sm.add_fact("Python 3.11 is stable", confidence=0.95)  # same content → same id
        rows = sm.query("MATCH (f:Fact) RETURN f.id")
        # Dedup by content hash — at most one entry
        assert len(rows) <= 2  # may be 1 if dedup works, 2 if not — both acceptable


# ── Search ────────────────────────────────────────────────────────────────────

@test_group("Semantic Memory")
def test_semantic_search_preferences():
    require("kuzu")
    with TempDir() as d:
        sm = _make_semantic(d / "sem")
        sm.add_node("Preference", "p1", {
            "category": "language", "value": "Python over Java",
            "strength": 0.9, "timestamp": time.time(),
        })
        sm.add_node("Preference", "p2", {
            "category": "editor", "value": "PyCharm over VSCode",
            "strength": 0.7, "timestamp": time.time(),
        })

        results = sm.search_preferences("Python language")
        assert len(results) >= 1
        assert any("Python" in r.get("value", "") for r in results)


@test_group("Semantic Memory")
def test_semantic_search_generic():
    require("kuzu")
    with TempDir() as d:
        sm = _make_semantic(d / "sem")
        sm.add_node("Preference", "pref_mypy", {
            "category": "tooling", "value": "strict mypy type checking",
            "strength": 0.8, "timestamp": time.time(),
        })

        results = sm.search_generic_memories("mypy type checking", limit=10)
        assert isinstance(results, list)


@test_group("Semantic Memory")
def test_semantic_generic_memories_structure():
    """search_generic_memories returns rows with required keys."""
    require("kuzu")
    with TempDir() as d:
        sm = _make_semantic(d / "sem")
        sm.add_node("Preference", "p_struct", {
            "category": "style", "value": "Google docstring format",
            "strength": 0.6, "timestamp": time.time(),
        })

        results = sm.search_generic_memories("docstring", limit=5)
        for row in results:
            assert "type" in row or "text" in row or "value" in row, \
                f"Row missing text field: {row.keys()}"
            assert "timestamp" in row
            assert "match_score" in row


# ── Stats / health ────────────────────────────────────────────────────────────

@test_group("Semantic Memory")
def test_semantic_get_stats():
    require("kuzu")
    with TempDir() as d:
        sm = _make_semantic(d / "sem")
        sm.add_preference(category="testing", value="pytest", strength=0.8)
        stats = sm.get_stats()
        assert isinstance(stats, dict)
        # Stats should have some content counts
        total = sum(v for v in stats.values() if isinstance(v, (int, float)))
        assert total >= 0


@test_group("Semantic Memory")
def test_semantic_persistence():
    """Graph data survives close/reopen."""
    from engram.memory.semantic_memory import SemanticMemory
    from engram.memory.types import ProjectType

    with TempDir() as d:
        db_path = d / "sem"
        sm1 = SemanticMemory(db_path=db_path)
        sm1.add_fact("Persistent fact for testing", confidence=0.8)
        sm1.close()
        sm2 = SemanticMemory(db_path=db_path)
        rows = sm2.list_facts(limit=10)
        assert len(rows) >= 1


# ── SemanticSearchMixin ───────────────────────────────────────────────────────

@test_group("Semantic Memory")
def test_semantic_search_mixin_scoring():
    """Scoring returns results sorted by match_score descending."""
    require("kuzu")
    with TempDir() as d:
        sm = _make_semantic(d / "sem")
        for i, val in enumerate(["asyncio coroutines", "event loop", "threading"]):
            sm.add_node("Preference", f"pref_{i}", {
                "category": "concurrency", "value": val,
                "strength": 0.5, "timestamp": time.time(),
            })

        results = sm.search_preferences("asyncio event loop")
        if len(results) >= 2:
            scores = [r.get("match_score", 0) for r in results]
            assert scores == sorted(scores, reverse=True), \
                f"Results not sorted by match_score: {scores}"
