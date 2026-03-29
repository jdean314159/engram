"""Integration tests for ProjectMemory — end-to-end pipeline."""

from __future__ import annotations

import time
from pathlib import Path

from .runner import test_group, require
from .mocks import TempDir, MockEngine, make_project_memory, unique_session


def _pm(d, **kw):
    from engram.project_memory import ProjectMemory
    return ProjectMemory(
        project_id="test",
        project_type="general_assistant",
        base_dir=d,
        llm_engine=MockEngine(),
        session_id=unique_session(),
        **kw,
    )


# ── Lifecycle ──────────────────────────────────────────────────────────────────

@test_group("ProjectMemory: Integration")
def test_pm_init():
    with TempDir() as d:
        pm = make_project_memory(d)
        assert pm.project_id == "test_project"
        assert pm.session_id is not None


@test_group("ProjectMemory: Integration")
def test_pm_add_turn_and_count():
    with TempDir() as d:
        pm = _pm(d)
        pm.add_turn("user", "What is asyncio?")
        pm.add_turn("assistant", "asyncio is a Python async I/O library.")
        assert pm.working.get_message_count() == 2


@test_group("ProjectMemory: Integration")
def test_pm_get_recent_turns():
    with TempDir() as d:
        pm = _pm(d)
        pm.add_turn("user", "Hello")
        pm.add_turn("assistant", "Hi there")
        turns = pm.get_recent_turns(n=10)
        assert len(turns) == 2
        roles = {t.role for t in turns}
        assert "user" in roles
        assert "assistant" in roles


@test_group("ProjectMemory: Integration")
def test_pm_build_prompt_returns_dict():
    """build_prompt assembles context + prompt without calling LLM."""
    with TempDir() as d:
        pm = _pm(d)
        pm.add_turn("user", "I love dark mode")
        pm.add_turn("assistant", "Noted — dark mode it is.")
        result = pm.build_prompt("What are my preferences?")
        assert isinstance(result, dict)
        assert "prompt" in result
        assert "context" in result
        assert "prompt_tokens" in result


@test_group("ProjectMemory: Integration")
def test_pm_prompt_contains_working_memory():
    """build_prompt injects working memory context into the prompt."""
    with TempDir() as d:
        pm = _pm(d)
        pm.add_turn("user", "I prefer dark mode in all tools")
        pm.add_turn("assistant", "Noted.")
        result = pm.build_prompt("What are my preferences?")
        assert "dark mode" in result["prompt"].lower()


@test_group("ProjectMemory: Integration")
def test_pm_new_session():
    with TempDir() as d:
        pm = _pm(d)
        pm.add_turn("user", "Session 1 message")
        old_session = pm.session_id

        new_sid = unique_session()
        pm.new_session(new_sid)
        assert pm.session_id == new_sid
        assert pm.session_id != old_session

        pm.add_turn("user", "Session 2 message")
        new_msgs = pm.get_recent_turns(n=5)
        assert all(m.session_id == new_sid for m in new_msgs)


@test_group("ProjectMemory: Integration")
def test_pm_get_context_returns_context_result():
    with TempDir() as d:
        pm = _pm(d)
        pm.add_turn("user", "I prefer pytest for all testing")
        pm.add_turn("assistant", "Got it.")
        ctx = pm.get_context(query="testing frameworks", max_tokens=500)
        assert hasattr(ctx, "working")
        assert hasattr(ctx, "total_tokens")
        assert ctx.total_tokens >= 0


@test_group("ProjectMemory: Integration")
def test_pm_get_context_working_layer():
    """get_context includes working memory messages."""
    with TempDir() as d:
        pm = _pm(d)
        pm.add_turn("user", "asyncio is my preferred concurrency model")
        ctx = pm.get_context(query="asyncio", max_tokens=500)
        assert ctx.working is not None
        assert len(ctx.working) > 0
        texts = " ".join(m.content for m in ctx.working)
        assert "asyncio" in texts.lower()


@test_group("ProjectMemory: Integration")
def test_pm_get_stats():
    with TempDir() as d:
        pm = _pm(d)
        pm.add_turn("user", "Hello")
        stats = pm.get_stats()
        assert "project_id" in stats
        assert "working" in stats
        assert stats["working"]["message_count"] >= 1


@test_group("ProjectMemory: Integration")
def test_pm_all_project_types():
    from engram.memory.types import ProjectType
    from engram.project_memory import ProjectMemory
    with TempDir() as d:
        for pt in ProjectType:
            with ProjectMemory(
                project_id=f"t_{pt.value}",
                project_type=pt,
                base_dir=d,
                llm_engine=MockEngine(),
                session_id=unique_session(),
            ) as pm:
                assert pm.project_id == f"t_{pt.value}"


@test_group("ProjectMemory: Integration")
def test_pm_clear_session():
    with TempDir() as d:
        pm = _pm(d)
        pm.add_turn("user", "Message 1")
        pm.add_turn("user", "Message 2")
        assert pm.working.get_message_count() == 2
        pm.clear_session()
        assert pm.working.get_message_count() == 0


@test_group("ProjectMemory: Integration")
def test_pm_store_and_search_episodes():
    """store_episode + search_episodes round-trip when episodic is available."""
    require("chromadb", "sentence_transformers")
    from engram.project_memory import ProjectMemory
    with TempDir() as d:
        pm = ProjectMemory(
            project_id="ep_test",
            project_type="programming_assistant",
            base_dir=d,
            llm_engine=MockEngine(),
            session_id=unique_session(),
        )
        ep_id = pm.store_episode(
            "User always uses type annotations in Python",
            metadata={"source": "test"},
            importance=0.9,
            bypass_filter=True,
        )
        assert ep_id is not None

        results = pm.search_episodes("type annotations", n=5)
        assert len(results) >= 1
        assert any("type" in r.text.lower() for r in results)


@test_group("ProjectMemory: Integration")
def test_pm_index_text_and_diagnostics():
    """index_text requires semantic memory (kuzu); diagnostics always works."""
    require("kuzu")
    with TempDir() as d:
        from engram.project_memory import ProjectMemory
        pm = ProjectMemory(
            project_id="idx_test",
            project_type="programming_assistant",
            base_dir=d,
            llm_engine=MockEngine(),
            session_id=unique_session(),
        )
        stats = pm.index_text(
            "Python asyncio enables concurrent I/O operations efficiently."
        )
        assert hasattr(stats, "sentences") or isinstance(stats, dict)

        snap = pm.get_diagnostics_snapshot()
        assert isinstance(snap, dict)
        assert "project_id" in snap


# ── Lifecycle manager ──────────────────────────────────────────────────────────

@test_group("ProjectMemory: Lifecycle")
def test_lifecycle_manager_init():
    from engram.memory.lifecycle import MemoryLifecycleManager, LifecycleConfig
    with TempDir() as d:
        pm = make_project_memory(d)
        # ProjectMemory exposes its ctx as _retriever.ctx or similar;
        # MemoryLifecycleManager takes a ctx (MemoryContext) directly.
        # Access it via internal attribute if available.
        ctx = getattr(pm, "_ctx", pm)
        mgr = MemoryLifecycleManager(ctx=ctx, config=LifecycleConfig())
        assert mgr is not None


@test_group("ProjectMemory: Lifecycle")
def test_lifecycle_promote_episode_no_error():
    from engram.memory.lifecycle import MemoryLifecycleManager, LifecycleConfig
    with TempDir() as d:
        pm = make_project_memory(d)
        ctx = getattr(pm, "_ctx", pm)
        mgr = MemoryLifecycleManager(ctx=ctx, config=LifecycleConfig())
        result = mgr.promote_episode(
            episode_id="ep_1",
            text="User always prefers async patterns",
            importance=0.85,
        )
        assert isinstance(result, dict)


@test_group("ProjectMemory: Lifecycle")
def test_pm_run_maintenance():
    """run_maintenance executes without error when episodic is absent."""
    with TempDir() as d:
        pm = _pm(d)
        result = pm.run_maintenance(dry_run=True)
        assert isinstance(result, dict)


# ── Telemetry ──────────────────────────────────────────────────────────────────

@test_group("ProjectMemory: Telemetry")
def test_telemetry_disabled_no_error():
    from engram.telemetry.core import Telemetry
    tel = Telemetry(sink=None, enabled=False)
    tel.emit("test_event", "nothing happened", x=1)


@test_group("ProjectMemory: Telemetry")
def test_telemetry_logging_sink_captures_events():
    from engram.telemetry.core import Telemetry
    from engram.telemetry.sinks import LoggingSink
    events = []

    class CaptureSink(LoggingSink):
        def emit(self, event):
            events.append(event)

    tel = Telemetry(sink=CaptureSink(), enabled=True)
    tel.emit("test", "a message", value=42)
    assert len(events) == 1
    assert events[0].kind == "test"
    assert events[0].fields["value"] == 42


@test_group("ProjectMemory: Telemetry")
def test_telemetry_jsonl_sink_writes_file():
    import json
    from engram.telemetry.core import Telemetry
    from engram.telemetry.sinks import JsonlFileSink
    with TempDir() as d:
        sink = JsonlFileSink(path=d / "events.jsonl")
        tel = Telemetry(sink=sink, enabled=True)
        tel.emit("retrieval", "retrieved 3 episodes", count=3)

        lines = (d / "events.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        ev = json.loads(lines[0])
        assert ev["kind"] == "retrieval"
        assert ev["fields"]["count"] == 3
