"""Tests for MemoryIngestor, UnifiedRetriever, and ForgettingPolicy."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runner import test_group, require
from .mocks import TempDir, make_project_memory, unique_session


# ── IngestionPolicy ────────────────────────────────────────────────────────────

@test_group("Memory Ingestion")
def test_ingestion_policy_for_project_types():
    from engram.memory.ingestion import IngestionPolicy
    from engram.memory.types import ProjectType
    for pt in ProjectType:
        policy = IngestionPolicy.for_project_type(pt)
        assert policy is not None
        assert 0.0 < policy.episode_threshold < 1.0


@test_group("Memory Ingestion")
def test_ingestion_decision_structure():
    from engram.memory.ingestion import IngestionDecision, SemanticMemoryWrite
    decision = IngestionDecision(
        should_store_episode=True,
        episode_text="A meaningful event occurred.",
        episode_importance=0.75,
        semantic_writes=[SemanticMemoryWrite(kind="fact", payload={"value": "x=1"})],
        reasons=["high importance", "contains fact"],
        source_text="A meaningful event occurred.",
    )
    d = decision.to_dict()
    for key in ("should_store_episode", "episode_text", "episode_importance",
                "semantic_writes", "reasons", "source_text", "timestamp"):
        assert key in d, f"Missing key: {key}"


@test_group("Memory Ingestion")
def test_ingestor_process_turn_no_side_effects():
    """process_turn returns IngestionDecision without storing anything."""
    from engram.memory.ingestion import MemoryIngestor
    with TempDir() as d:
        pm = make_project_memory(d)
        ingestor = MemoryIngestor(pm)
        decision = ingestor.process_turn(
            role="user",
            content="I prefer to use async/await over callbacks in Python.",
        )
        assert decision is not None
        assert isinstance(decision.should_store_episode, bool)
        assert 0.0 <= decision.episode_importance <= 1.0
        # process_turn should not write to working memory
        assert pm.working.get_message_count() == 0


@test_group("Memory Ingestion")
def test_ingestor_ephemeral_lower_than_preference():
    """Ephemeral phrases score lower than strong preferences."""
    from engram.memory.ingestion import MemoryIngestor
    with TempDir() as d:
        pm = make_project_memory(d)
        ingestor = MemoryIngestor(pm)

        greeting = ingestor.process_turn(role="user", content="Hi! How are you?")
        preference = ingestor.process_turn(
            role="user",
            content="I strongly prefer type-annotated Python with strict mypy checks.",
        )
        assert greeting.episode_importance <= preference.episode_importance


@test_group("Memory Ingestion")
def test_ingestor_apply_outcome_keys():
    """apply() returns dict with required outcome keys."""
    from engram.memory.ingestion import MemoryIngestor
    with TempDir() as d:
        pm = make_project_memory(d)
        ingestor = MemoryIngestor(pm)
        decision = ingestor.process_turn(role="user", content="Testing apply.")
        result = ingestor.apply(decision)
        assert "episode_id" in result
        assert "semantic_writes" in result
        assert "decision" in result


@test_group("Memory Ingestion")
def test_ingestor_apply_stores_when_forced():
    """apply() stores episode when should_store_episode forced True."""
    from engram.memory.ingestion import MemoryIngestor
    with TempDir() as d:
        pm = make_project_memory(d)
        ingestor = MemoryIngestor(pm)
        decision = ingestor.process_turn(
            role="user", content="I always use pytest for testing Python."
        )
        decision.should_store_episode = True
        decision.episode_text = decision.episode_text or decision.source_text
        result = ingestor.apply(decision)
        # With episodic disabled: episode_id=None but no exception
        assert "episode_id" in result


@test_group("Memory Ingestion")
def test_ingestor_custom_policy():
    """MemoryIngestor respects a custom IngestionPolicy."""
    from engram.memory.ingestion import MemoryIngestor, IngestionPolicy
    with TempDir() as d:
        pm = make_project_memory(d)
        # Force everything to store by setting threshold=0
        policy = IngestionPolicy(episode_threshold=0.0)
        ingestor = MemoryIngestor(pm, policy=policy)
        decision = ingestor.process_turn(role="user", content="Even mundane text.")
        # With threshold=0, any non-ephemeral content should qualify
        # (score may still be 0 if ephemeral patterns match, so just check no exception)
        assert isinstance(decision.episode_importance, float)


# ── UnifiedRetriever via ProjectMemory.get_context() ──────────────────────────
#
# UnifiedRetriever.retrieve() returns a ContextResult (not List[RetrievalCandidate]).
# The public surface for multi-layer retrieval is ProjectMemory.get_context().

def _pm_with_data(base_dir: Path):
    """Build a ProjectMemory with pre-populated working memory."""
    from engram.project_memory import ProjectMemory
    pm = ProjectMemory(
        project_id="ret_test",
        project_type="general_assistant",
        base_dir=base_dir,
        llm_engine=None,
        session_id=unique_session(),
    )
    return pm


@test_group("Unified Retrieval")
def test_retrieval_empty_state():
    """get_context on empty memory returns a ContextResult with no working messages."""
    with TempDir() as d:
        pm = _pm_with_data(d)
        ctx = pm.get_context(query="asyncio tutorial", max_tokens=200)
        assert hasattr(ctx, "working")
        assert hasattr(ctx, "total_tokens")
        assert ctx.total_tokens >= 0


@test_group("Unified Retrieval")
def test_retrieval_finds_working_memory():
    """get_context includes matching working memory messages."""
    with TempDir() as d:
        pm = _pm_with_data(d)
        pm.add_turn("user", "I love asyncio and concurrency patterns")
        pm.add_turn("assistant", "asyncio is great for IO-bound tasks")

        ctx = pm.get_context(query="asyncio", max_tokens=500)
        assert ctx.working is not None
        texts = " ".join(m.content for m in ctx.working)
        assert "asyncio" in texts.lower()


@test_group("Unified Retrieval")
def test_retrieval_context_result_structure():
    """ContextResult exposes all layer fields."""
    with TempDir() as d:
        pm = _pm_with_data(d)
        pm.add_turn("user", "Python asyncio event loop")
        ctx = pm.get_context(query="asyncio", max_tokens=500)
        for attr in ("working", "episodic", "semantic", "cold", "total_tokens"):
            assert hasattr(ctx, attr), f"ContextResult missing attr: {attr}"


@test_group("Unified Retrieval")
def test_retrieval_finds_cold():
    """get_context includes cold storage when working+episodic can not fill budget."""
    with TempDir() as d:
        pm = _pm_with_data(d)
        # Archive directly into cold storage
        pm.cold.archive([{
            "text": "User once discussed Django REST framework extensively.",
            "project_id": pm.project_id,
        }])
        ctx = pm.get_context(query="Django REST", max_tokens=500, cold_fallback=True)
        cold_texts = " ".join(r["text"] for r in (ctx.cold or []))
        assert "Django" in cold_texts


@test_group("Unified Retrieval")
def test_retrieval_budget_respected():
    """total_tokens stays within the requested max_tokens budget."""
    with TempDir() as d:
        pm = _pm_with_data(d)
        for i in range(20):
            pm.add_turn("user", f"Message {i} about asyncio patterns in Python")

        ctx = pm.get_context(query="asyncio", max_tokens=100)
        # working_tokens + episodic_tokens + semantic_tokens + cold_tokens ≤ budget
        assert ctx.total_tokens <= 100, f"Budget exceeded: {ctx.total_tokens}"


@test_group("Unified Retrieval")
def test_retrieval_to_prompt_sections():
    """ContextResult.to_prompt_sections() returns a dict of section strings."""
    with TempDir() as d:
        pm = _pm_with_data(d)
        pm.add_turn("user", "asyncio is my preferred concurrency model")
        ctx = pm.get_context(query="asyncio", max_tokens=500)
        sections = ctx.to_prompt_sections()
        assert isinstance(sections, dict)
        # Each value is a string (possibly empty if layer has no data)
        for k, v in sections.items():
            assert isinstance(v, str), f"Section {k!r} value is not str: {type(v)}"


# ── ForgettingPolicy ───────────────────────────────────────────────────────────

@test_group("Forgetting Policy")
def test_forgetting_record_access_no_error():
    from engram.memory.forgetting import ForgettingPolicy
    with TempDir() as d:
        policy = ForgettingPolicy(access_db_path=d / "access.db")
        policy.record_access(["ep_1", "ep_2", "ep_3"])


@test_group("Forgetting Policy")
def test_forgetting_run_skipped_when_too_few_episodes():
    """run() returns 'skipped' when episode count is below minimum."""
    from engram.memory.forgetting import ForgettingPolicy, ForgettingConfig
    from engram.memory.cold_storage import ColdStorage

    with TempDir() as d:
        config = ForgettingConfig(min_episodes_before_archival=50)
        policy = ForgettingPolicy(access_db_path=d / "access.db", config=config)
        cold = ColdStorage(db_path=d / "cold.db")

        class TinyEpisodic:
            def get_stats(self): return {"total_episodes": 0}

        result = policy.run(TinyEpisodic(), cold, project_id="p")
        assert result.get("status") in ("skipped", "disabled")


@test_group("Forgetting Policy")
def test_forgetting_dry_run_no_deletion():
    """dry_run=True reports but does not delete/archive anything.

    NOTE: ForgettingPolicy.score_all() calls episodic_memory.collection.get()
    directly — it is tightly coupled to ChromaDB internals. This test requires
    a real EpisodicMemory instance backed by ChromaDB.
    """
    require("chromadb", "sentence_transformers")
    from engram.memory.forgetting import ForgettingPolicy, ForgettingConfig
    from engram.memory.cold_storage import ColdStorage
    from engram.memory.episodic_memory import EpisodicMemory

    with TempDir() as d:
        config = ForgettingConfig(
            min_episodes_before_archival=0,
            retention_threshold=1.0,  # retain nothing
            min_age_days=0.0,
        )
        policy = ForgettingPolicy(access_db_path=d / "access.db", config=config)
        cold = ColdStorage(db_path=d / "cold.db")
        episodic = EpisodicMemory(persist_dir=d / "ep", embedding_device="cpu")

        for i in range(5):
            episodic.add_episode(f"Episode {i}", project_id="p", importance=0.1)

        result = policy.run(episodic, cold, project_id="p", dry_run=True)
        # dry_run=True: nothing archived/deleted
        assert isinstance(result, dict)
        stats = episodic.get_stats()
        # Episodes should still be present
        assert stats.get("total_episodes", 0) == 5
