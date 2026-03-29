"""Validation tests for P1 Bug #4 — index_text/index_documents graceful no-op.

Tests verify:
1. Both methods return ExtractionStats (not raise) when semantic is disabled.
2. The returned stats have sensible zero values and method="disabled".
3. index_documents preserves the document count in the no-op stats.
4. When semantic IS available (kuzu installed), indexing actually extracts.
5. Callers can unconditionally call index_text without a None-check.
"""

from __future__ import annotations

from pathlib import Path

from .runner import test_group, require
from .mocks import TempDir, unique_session


def _pm(d: Path, **kw):
    from engram.project_memory import ProjectMemory
    return ProjectMemory(
        project_id="idx_test",
        project_type="programming_assistant",
        base_dir=d,
        llm_engine=None,
        session_id=unique_session(),
        **kw,
    )


# ── No-op when semantic disabled ───────────────────────────────────────────────

@test_group("P1-4: index_text No-op")
def test_index_text_no_semantic_returns_stats_not_raises():
    """index_text without semantic → ExtractionStats, never RuntimeError."""
    from engram.memory.extraction import ExtractionStats
    with TempDir() as d:
        pm = _pm(d)
        # index_text must never raise, regardless of whether semantic is enabled
        stats = pm.index_text("Python asyncio enables concurrent I/O operations.")
        assert isinstance(stats, ExtractionStats)


@test_group("P1-4: index_text No-op")
def test_index_text_no_semantic_zero_counts():
    """No-op stats have zero entity/sentence/relation counts."""
    with TempDir() as d:
        pm = _pm(d)
        stats = pm.index_text("Some text about asyncio and coroutines.")
        # When semantic disabled: all zero.
        # When enabled (kuzu installed): counts reflect extraction.
        if pm.extractor is None:
            assert stats.entities == 0
            assert stats.sentences == 0
            assert stats.relations == 0
        assert stats.llm_calls == 0


@test_group("P1-4: index_text No-op")
def test_index_text_no_semantic_method_is_disabled():
    """No-op stats carry method='disabled' so callers can distinguish them."""
    with TempDir() as d:
        pm = _pm(d)
        stats = pm.index_text("Anything at all.")
        if pm.extractor is None:
            assert stats.method == "disabled"
        else:
            assert stats.method in ("tfidf", "spacy")


@test_group("P1-4: index_text No-op")
def test_index_documents_no_semantic_returns_stats_not_raises():
    """index_documents without semantic → ExtractionStats, never RuntimeError."""
    from engram.memory.extraction import ExtractionStats
    with TempDir() as d:
        pm = _pm(d)
        stats = pm.index_documents(["Doc one.", "Doc two.", "Doc three."])
        assert isinstance(stats, ExtractionStats)


@test_group("P1-4: index_text No-op")
def test_index_documents_no_semantic_preserves_doc_count():
    """No-op stats record the number of documents passed in."""
    with TempDir() as d:
        pm = _pm(d)
        docs = ["Doc one.", "Doc two.", "Doc three.", "Doc four."]
        stats = pm.index_documents(docs)
        assert stats.documents == len(docs)
        if pm.extractor is None:
            assert stats.method == "disabled"


@test_group("P1-4: index_text No-op")
def test_index_text_empty_string_no_raise():
    """Empty string is a valid no-op call."""
    with TempDir() as d:
        pm = _pm(d)
        stats = pm.index_text("")
        assert stats.entities == 0


@test_group("P1-4: index_text No-op")
def test_index_documents_empty_list_no_raise():
    """Empty document list is a valid no-op call."""
    with TempDir() as d:
        pm = _pm(d)
        stats = pm.index_documents([])
        assert stats.documents == 0


@test_group("P1-4: index_text No-op")
def test_index_text_unconditional_call_pattern():
    """Callers can unconditionally call index_text — no try/except needed."""
    with TempDir() as d:
        pm = _pm(d)
        # Pattern: call unconditionally, inspect stats after
        stats = pm.index_text("Python asyncio is useful for concurrent I/O.")
        if stats.method == "disabled":
            pass   # semantic not available — fine, no-op
        else:
            assert stats.entities >= 0   # semantic ran


@test_group("P1-4: index_text No-op")
def test_index_text_to_dict_works_on_no_op_stats():
    """ExtractionStats.to_dict() works on disabled stats."""
    with TempDir() as d:
        pm = _pm(d)
        stats = pm.index_text("Some text.")
        d_out = stats.to_dict()
        assert isinstance(d_out, dict)
        assert "entities" in d_out
        assert "method" in d_out
        if pm.extractor is None:
            assert d_out["entities"] == 0
            assert d_out["method"] == "disabled"


# ── Diagnostics snapshot is unaffected ────────────────────────────────────────

@test_group("P1-4: index_text No-op")
def test_diagnostics_snapshot_works_without_semantic():
    """get_diagnostics_snapshot() still works after no-op index_text."""
    with TempDir() as d:
        pm = _pm(d)
        pm.index_text("Some content to index.")
        snap = pm.get_diagnostics_snapshot()
        assert isinstance(snap, dict)
        assert "project_id" in snap


# ── With real semantic (kuzu) the methods actually extract ─────────────────────

@test_group("P1-4: index_text No-op")
def test_index_text_with_semantic_extracts_entities():
    """When kuzu is available, index_text actually populates the graph."""
    require("kuzu", "sklearn")
    from engram.project_memory import ProjectMemory
    with TempDir() as d:
        pm = ProjectMemory(
            project_id="sem_idx_test",
            project_type="programming_assistant",
            base_dir=d,
            llm_engine=None,
            session_id=unique_session(),
        )
        assert pm.extractor is not None, "Expected extractor with semantic memory"
        stats = pm.index_text(
            "Python asyncio provides event loop management for concurrent I/O. "
            "asyncio coroutines are defined with async def syntax. "
            "The event loop schedules coroutines and handles I/O callbacks."
        )
        assert stats.method != "disabled"
        assert stats.sentences >= 2
        assert stats.entities >= 1


@test_group("P1-4: index_text No-op")
def test_index_documents_with_semantic_returns_aggregate_stats():
    """index_documents returns aggregate stats across all documents."""
    require("kuzu", "sklearn")
    from engram.project_memory import ProjectMemory
    with TempDir() as d:
        pm = ProjectMemory(
            project_id="docs_idx_test",
            project_type="programming_assistant",
            base_dir=d,
            llm_engine=None,
            session_id=unique_session(),
        )
        docs = [
            "Python asyncio enables concurrent I/O operations.",
            "FastAPI builds REST APIs with Python type hints.",
            "SQLAlchemy provides ORM support for Python databases.",
        ]
        stats = pm.index_documents(docs)
        assert stats.method != "disabled"
        assert stats.documents == 3
        assert stats.sentences >= 3
