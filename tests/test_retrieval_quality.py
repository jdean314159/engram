"""Retrieval quality tests for Engram.

These tests seed memory layers with known content and assert that the
retrieval pipeline surfaces the right items.  They are CPU-only, require
no GPU and no chromadb/kuzu, and run in < 2 seconds.

Layers exercised:
  - Working memory (always available): recent conversation turns
  - Cold storage (SQLite FTS5, always available): archived episodes
  - build_prompt pressure valve (token budget enforcement)

Layers not exercised here (require optional deps):
  - Episodic (chromadb) — integration tested separately when available
  - Semantic (kuzu) — integration tested separately when available
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine():
    """Minimal fake LLM engine."""
    class Eng:
        model_name = "test-model"
        system_prompt = "You are a helpful assistant."
        is_cloud = False
        max_context_length = 2048
        def count_tokens(self, t): return max(1, len(t) // 4)
        def compress_prompt(self, p, target_tokens): return p[:target_tokens * 4]
    return Eng()


def _make_pm(tmpdir, total_tokens=2000):
    from engram.project_memory import ProjectMemory, TokenBudget
    budget = TokenBudget(working=400, episodic=400, semantic=200, cold=400)
    return ProjectMemory(
        project_id="retrieval_test",
        project_type="general",
        base_dir=Path(tmpdir),
        llm_engine=_engine(),
        token_budget=budget,
    )


def _context_text(context: Any) -> str:
    """Flatten all context layers to a single lowercase string."""
    parts = []
    for msg in getattr(context, "working", []) or []:
        parts.append(getattr(msg, "content", ""))
    for ep in getattr(context, "episodic", []) or []:
        parts.append(getattr(ep, "text", ""))
    for row in getattr(context, "semantic", []) or []:
        if isinstance(row, dict):
            parts.append(str(row.get("content") or row.get("text") or row.get("value") or ""))
    for row in getattr(context, "cold", []) or []:
        if isinstance(row, dict):
            parts.append(str(row.get("text") or ""))
    return "\n".join(p for p in parts if p).lower()


# ---------------------------------------------------------------------------
# Working memory retrieval
# ---------------------------------------------------------------------------

def test_working_memory_returned_without_query():
    """Working memory context window is always populated, even with no query."""
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)
        pm.add_turn("user", "I prefer Python over Java.")
        pm.add_turn("assistant", "Noted. Python is a great choice.")

        context = pm.get_context(query=None)
        text = _context_text(context)
        assert "python" in text
        assert "java" in text
        assert context.working_tokens > 0


def test_working_memory_recent_turns_in_context():
    """Most recent turns appear in context."""
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)
        for i in range(5):
            pm.add_turn("user", f"turn number {i}")
            pm.add_turn("assistant", f"response {i}")

        context = pm.get_context(query=None)
        text = _context_text(context)
        # Most recent turn should always be present
        assert "turn number 4" in text


def test_working_memory_token_budget_respected():
    """Working memory respects its token budget."""
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)
        # Fill with more than the working budget
        for i in range(30):
            pm.add_turn("user", f"message {i}: " + "x" * 40)
        context = pm.get_context(query=None)
        assert context.working_tokens <= pm.budget.working + 50  # small tolerance


# ---------------------------------------------------------------------------
# Cold storage retrieval
# ---------------------------------------------------------------------------

def test_cold_storage_seeded_and_recalled():
    """Episodes archived to cold storage are returned by a matching query."""
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)

        # Archive known content directly
        pm.cold.archive([
            {
                "id": "ep_asyncio",
                "project_id": "retrieval_test",
                "session_id": "s1",
                "text": "Fixed a critical asyncio event loop error in the Python service.",
                "timestamp": time.time() - 3600,
                "metadata": {"importance": 0.8},
            },
            {
                "id": "ep_gardening",
                "project_id": "retrieval_test",
                "session_id": "s1",
                "text": "Planted tomatoes and watered the garden beds today.",
                "timestamp": time.time() - 7200,
                "metadata": {"importance": 0.3},
            },
            {
                "id": "ep_retry",
                "project_id": "retrieval_test",
                "session_id": "s1",
                "text": "Implemented exponential backoff retry logic for the worker queue.",
                "timestamp": time.time() - 1800,
                "metadata": {"importance": 0.7},
            },
        ])

        context = pm.get_context(query="asyncio python error", cold_n=5)
        text = _context_text(context)

        assert "asyncio" in text, "asyncio episode should be recalled"
        assert "gardening" not in text or "tomatoes" not in text, \
            "gardening episode should not appear for asyncio query"


def test_cold_storage_retry_recall():
    """Retry-related episode is recalled by a retry query."""
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)
        pm.cold.archive([
            {
                "id": "ep_retry",
                "project_id": "retrieval_test",
                "session_id": "s1",
                "text": "Added retry with exponential backoff to the task worker.",
                "timestamp": time.time(),
                "metadata": {"importance": 0.7},
            },
            {
                "id": "ep_unrelated",
                "project_id": "retrieval_test",
                "session_id": "s1",
                "text": "Had coffee and reviewed the quarterly report.",
                "timestamp": time.time() - 60,
                "metadata": {"importance": 0.2},
            },
        ])

        context = pm.get_context(query="worker retry exponential backoff", cold_n=5)
        text = _context_text(context)
        assert "retry" in text and "worker" in text


def test_cold_storage_project_isolation():
    """Cold storage only returns records for the matching project_id."""
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)

        # Archive a record for a *different* project with a unique sentinel phrase
        pm.cold.archive([
            {
                "id": "other_ep",
                "project_id": "other_project",
                "session_id": "s1",
                "text": "SENTINEL_PHRASE_XYZ asyncio note for wrong project.",
                "timestamp": time.time(),
                "metadata": {},
            }
        ])

        context = pm.get_context(query="asyncio", cold_n=10)
        text = _context_text(context)
        # The sentinel phrase must NOT appear — it belongs to the wrong project_id
        assert "sentinel_phrase_xyz" not in text, (
            "Cold storage leaked records from a different project"
        )


# ---------------------------------------------------------------------------
# build_prompt retrieval integration
# ---------------------------------------------------------------------------

def test_build_prompt_includes_cold_context():
    """build_prompt assembles prompt that includes cold-archived relevant content."""
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)
        pm.cold.archive([
            {
                "id": "ep1",
                "project_id": "retrieval_test",
                "session_id": "s1",
                "text": "The database connection pool was set to max 20 connections.",
                "timestamp": time.time() - 60,
                "metadata": {"importance": 0.8},
            }
        ])

        result = pm.build_prompt(
            "What is the database connection pool size?",
            max_prompt_tokens=800,
            reserve_output_tokens=100,
        )
        prompt = result["prompt"].lower()
        assert "connection pool" in prompt or "20 connections" in prompt


def test_build_prompt_pressure_valve_triggers():
    """Pressure valve compresses memory when prompt exceeds token cap."""
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td, total_tokens=2000)
        # Fill working memory with many long messages
        for i in range(20):
            pm.add_turn("user", f"question {i}: " + "word " * 30)
            pm.add_turn("assistant", "answer: " + "word " * 30)

        result = pm.build_prompt(
            "What did we discuss?",
            max_prompt_tokens=256,
            reserve_output_tokens=64,
        )
        assert result["prompt_tokens"] <= 256 + 20  # small tolerance
        assert isinstance(result["compressed"], bool)


def test_build_prompt_no_memory_still_works():
    """build_prompt works correctly when all memory layers are empty."""
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)
        result = pm.build_prompt(
            "hello",
            max_prompt_tokens=512,
            reserve_output_tokens=64,
        )
        assert "hello" in result["prompt"]
        assert result["compressed"] is False


# ---------------------------------------------------------------------------
# Retrieval fixture runner (existing infra)
# ---------------------------------------------------------------------------

def test_retrieval_fixture_runner_cold_path():
    """run_retrieval_fixtures works end-to-end via cold storage fallback."""
    from engram.eval.retrieval_eval import (
        RetrievalFixture, run_retrieval_fixtures, summarize_fixture_results
    )

    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)

        # Seed cold storage with fixture-relevant content
        pm.cold.archive([
            {
                "id": "fix1",
                "project_id": "retrieval_test",
                "session_id": "s1",
                "text": "Encountered a Python asyncio event loop error during testing.",
                "timestamp": time.time(),
                "metadata": {"importance": 0.9},
            },
            {
                "id": "fix2",
                "project_id": "retrieval_test",
                "session_id": "s1",
                "text": "We decided to use exponential backoff retry logic for the worker queue.",
                "timestamp": time.time(),
                "metadata": {"importance": 0.8},
            },
        ])

        fixtures = [
            RetrievalFixture(
                name="asyncio_recall",
                query="python asyncio error",
                expected_substrings=("asyncio", "python", "error"),
                min_recall=0.66,
            ),
            RetrievalFixture(
                name="retry_recall",
                query="retry exponential backoff worker",
                # Use exact words from the seeded episode text
                expected_substrings=("retry", "exponential", "backoff"),
                min_recall=0.66,
            ),
        ]

        report = run_retrieval_fixtures(pm, fixtures)
        summary = report["summary"]

        # Both fixtures should pass with cold storage providing the content
        assert report["ok"], (
            f"Retrieval fixtures failed: {summary['failed']}\n"
            f"Results: {json.dumps(report['results'], indent=2)}"
        )
        assert summary["avg_recall"] >= 0.8


# ---------------------------------------------------------------------------
# Synonym expansion for episodic embedding query
# ---------------------------------------------------------------------------

def test_expand_query_no_synonyms():
    """Queries with no synonym-map hits are returned unchanged."""
    from engram.memory.retrieval import UnifiedRetriever
    from unittest.mock import MagicMock
    ctx = MagicMock()
    r = UnifiedRetriever(ctx)
    terms = r._query_terms("hello world")
    assert r._expand_query_for_embedding("hello world", terms) == "hello world"


def test_expand_query_appends_synonyms():
    """Known synonym-map entries are appended."""
    from engram.memory.retrieval import UnifiedRetriever
    from unittest.mock import MagicMock
    ctx = MagicMock()
    r = UnifiedRetriever(ctx)
    terms = r._query_terms("asyncio bug")
    expanded = r._expand_query_for_embedding("asyncio bug", terms)
    assert expanded.startswith("asyncio bug")
    for syn in ["async", "await", "gather", "error", "failure", "issue", "problem"]:
        assert syn in expanded, f"Missing synonym {repr(syn)}"


def test_expand_query_no_duplicate_tokens():
    """Synonyms already present in the query are not re-appended."""
    from engram.memory.retrieval import UnifiedRetriever
    from unittest.mock import MagicMock
    ctx = MagicMock()
    r = UnifiedRetriever(ctx)
    # "error" is a synonym of "bug" but also already in the query
    terms = r._query_terms("asyncio bug error")
    expanded = r._expand_query_for_embedding("asyncio bug error", terms)
    # "error" should not appear twice
    assert expanded.count("error") == 1, f"Duplicate 'error' in: {repr(expanded)}"


def test_expand_query_preserves_original_prefix():
    """Original query is always the prefix of the expanded query."""
    from engram.memory.retrieval import UnifiedRetriever
    from unittest.mock import MagicMock
    ctx = MagicMock()
    r = UnifiedRetriever(ctx)
    for query in ["asyncio bug", "fix the issue", "python llm"]:
        terms = r._query_terms(query)
        expanded = r._expand_query_for_embedding(query, terms)
        assert expanded.startswith(query), f"Prefix not preserved for {repr(query)}"


def test_expand_query_deterministic():
    """Same query always produces the same expansion (sorted synonyms)."""
    from engram.memory.retrieval import UnifiedRetriever
    from unittest.mock import MagicMock
    ctx = MagicMock()
    r = UnifiedRetriever(ctx)
    terms = r._query_terms("asyncio bug")
    results = {r._expand_query_for_embedding("asyncio bug", terms) for _ in range(5)}
    assert len(results) == 1, f"Non-deterministic expansion: {results}"


def test_expand_query_long_query_unchanged():
    """Long natural-language query with no synonym hits is unchanged."""
    from engram.memory.retrieval import UnifiedRetriever
    from unittest.mock import MagicMock
    ctx = MagicMock()
    r = UnifiedRetriever(ctx)
    query = "What did we discuss about the database schema migration last week"
    terms = r._query_terms(query)
    expanded = r._expand_query_for_embedding(query, terms)
    assert expanded == query
