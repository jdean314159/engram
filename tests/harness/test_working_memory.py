"""Tests for WorkingMemory (Layer 1) — SQLite-backed, no optional deps."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .runner import test_group
from .mocks import TempDir, unique_session


def _wm(max_tokens: int = 20000):
    """Create a WorkingMemory with a unique session to prevent shared-cache pollution."""
    from engram.memory.working_memory import WorkingMemory
    return WorkingMemory(db_path=None, session_id=unique_session(), max_tokens=max_tokens)


# ── basic CRUD ──────────────────────────────────────────────────────────────────

@test_group("Working Memory")
def test_wm_add_and_retrieve():
    wm = _wm()
    msg = wm.add("user", "Hello, world!")
    assert msg.id is not None
    assert msg.role == "user"
    assert msg.content == "Hello, world!"
    recent = wm.get_recent(n=5)
    assert len(recent) == 1
    assert recent[0].content == "Hello, world!"


@test_group("Working Memory")
def test_wm_multiple_roles():
    wm = _wm()
    wm.add("user", "What is 2+2?")
    wm.add("assistant", "It is 4.")
    wm.add("system", "You are helpful.")
    assert wm.get_message_count() == 3
    recent = wm.get_recent(n=10)
    roles = {m.role for m in recent}
    assert roles == {"user", "assistant", "system"}


@test_group("Working Memory")
def test_wm_token_budget_eviction():
    """FIFO eviction fires when token budget is exceeded."""
    # "AAAA" = 1 token via len//4; budget=3 → keep at most 3 tokens
    wm = _wm(max_tokens=3)
    for _ in range(10):
        wm.add("user", "AAAA")

    count = wm.get_token_count()
    assert count <= 3, f"Expected ≤3 tokens after eviction, got {count}"


@test_group("Working Memory")
def test_wm_search_keyword():
    wm = _wm()
    wm.add("user", "I love Python asyncio")
    wm.add("assistant", "asyncio is great for concurrency")
    wm.add("user", "Tell me about threading")

    results = wm.search("asyncio")
    assert len(results) == 2
    for r in results:
        assert "asyncio" in r.content


@test_group("Working Memory")
def test_wm_search_role_filter():
    wm = _wm()
    wm.add("user", "asyncio question here")
    wm.add("assistant", "asyncio answer here")

    user_only = wm.search("asyncio", role_filter="user")
    assert len(user_only) == 1
    assert user_only[0].role == "user"


@test_group("Working Memory")
def test_wm_context_window_respects_budget():
    wm = _wm(max_tokens=100)
    for _ in range(50):
        wm.add("user", "AAAA")   # 1 token each

    window = wm.get_context_window(max_tokens=5)
    total = sum(m.token_count for m in window)
    assert total <= 5, f"Context window exceeded budget: {total}"


@test_group("Working Memory")
def test_wm_session_isolation():
    """Different session_ids do not bleed into each other (shared in-memory db)."""
    from engram.memory.working_memory import WorkingMemory
    sid_a, sid_b = unique_session(), unique_session()
    wm_a = WorkingMemory(db_path=None, session_id=sid_a, max_tokens=10000)
    wm_b = WorkingMemory(db_path=None, session_id=sid_b, max_tokens=10000)

    wm_a.add("user", "Message for A")
    wm_b.add("user", "Message for B")

    assert wm_a.get_message_count() == 1
    assert wm_b.get_message_count() == 1
    assert wm_a.get_recent()[0].content == "Message for A"
    assert wm_b.get_recent()[0].content == "Message for B"


@test_group("Working Memory")
def test_wm_clear_session():
    wm = _wm()
    wm.add("user", "One")
    wm.add("user", "Two")
    assert wm.get_message_count() == 2
    wm.clear_session()
    assert wm.get_message_count() == 0


@test_group("Working Memory")
def test_wm_stats():
    wm = _wm()
    wm.add("user", "Hello")
    wm.add("assistant", "Hi there")
    stats = wm.get_stats()

    assert stats["message_count"] == 2
    assert stats["token_count"] > 0
    assert stats["token_budget"] > 0
    assert 0.0 <= stats["utilization"] <= 1.0
    assert "session_id" in stats


@test_group("Working Memory")
def test_wm_persistence_across_instances():
    """Messages survive across WorkingMemory instances sharing the same db file."""
    with TempDir() as d:
        from engram.memory.working_memory import WorkingMemory
        db = d / "working.db"
        sid = unique_session()

        wm1 = WorkingMemory(db_path=db, session_id=sid)
        wm1.add("user", "Persistent message")
        wm1.close()

        wm2 = WorkingMemory(db_path=db, session_id=sid)
        recent = wm2.get_recent()
        assert len(recent) == 1
        assert recent[0].content == "Persistent message"
        wm2.close()


@test_group("Working Memory")
def test_wm_thread_safety():
    """Concurrent writes from multiple threads don't corrupt state."""
    wm = _wm(max_tokens=1_000_000)
    errors = []

    def writer(tid: int):
        try:
            for i in range(20):
                wm.add("user", f"Thread {tid} message {i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert wm.get_message_count() == 100


@test_group("Working Memory")
def test_wm_metadata_roundtrip():
    wm = _wm()
    meta = {"importance": 0.9, "tags": ["python", "asyncio"], "nested": {"x": 1}}
    wm.add("user", "Complex metadata", metadata=meta)
    recent = wm.get_recent()
    assert recent[0].metadata == meta


@test_group("Working Memory")
def test_wm_get_recent_newest_first():
    """get_recent returns messages newest-first."""
    wm = _wm()
    for i in range(5):
        wm.add("user", f"Message {i}")
        time.sleep(0.002)

    recent = wm.get_recent(n=5)
    contents = [m.content for m in recent]
    assert contents[0] == "Message 4"
    assert contents[-1] == "Message 0"


@test_group("Working Memory")
def test_wm_custom_token_counter():
    """WorkingMemory accepts a custom token counter function."""
    from engram.memory.working_memory import WorkingMemory
    counter_calls = []

    def my_counter(text: str) -> int:
        counter_calls.append(text)
        return len(text.split())  # word count instead of len//4

    sid = unique_session()
    wm = WorkingMemory(db_path=None, session_id=sid, max_tokens=100,
                       token_counter=my_counter)
    wm.add("user", "one two three four")
    assert len(counter_calls) >= 1
    assert wm.get_token_count() == 4  # 4 words


@test_group("Working Memory")
def test_wm_get_context_excludes_system():
    wm = _wm()
    wm.add("system", "System prompt: " + "x" * 100)
    wm.add("user", "User message")

    without_sys = wm.get_context_window(max_tokens=5000, include_system=False)
    roles = {m.role for m in without_sys}
    assert "system" not in roles

    with_sys = wm.get_context_window(max_tokens=5000, include_system=True)
    roles_with = {m.role for m in with_sys}
    assert "system" in roles_with
