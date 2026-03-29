"""Validation tests for P1 Bug #5 — WorkingMemory in-memory isolation.

The bug: WorkingMemory(db_path=None) previously used
  file::memory:?cache=shared
which is a single process-global SQLite database shared by every instance.
Session-ID scoping provided data isolation but not database isolation, causing
cross-instance contamination in concurrent usage and tests.

The fix: each instance now gets its own named URI:
  file:wm_{safe_session_id}?mode=memory&cache=shared
giving full database-level isolation while retaining the shared-cache thread
safety that thread-local connections require within a single instance.

Tests verify:
1. Each instance gets a distinct db_path derived from its session_id.
2. Writes to instance A are invisible to instance B (and vice versa).
3. get_message_count() on each instance reflects only its own writes.
4. Thread safety within a single instance is preserved.
5. Multiple ProjectMemory instances in the same process don't bleed state.
6. get_stats() token counts are per-instance, not cumulative.
7. clear_session() on one instance doesn't affect another.
8. Session IDs with special characters are sanitised safely.
"""

from __future__ import annotations

import threading
from pathlib import Path

from .runner import test_group
from .mocks import TempDir, unique_session


def _wm(session_id=None, max_tokens=100_000):
    from engram.memory.working_memory import WorkingMemory
    sid = session_id or unique_session()
    return WorkingMemory(db_path=None, session_id=sid, max_tokens=max_tokens)


# ── URI isolation ──────────────────────────────────────────────────────────────

@test_group("P1-5: WorkingMemory Isolation")
def test_each_instance_gets_distinct_db_path():
    """Two instances with different session_ids get different db_path URIs."""
    wm1 = _wm()
    wm2 = _wm()
    assert wm1.db_path != wm2.db_path, (
        f"Expected distinct db_paths, got {wm1.db_path!r} for both"
    )


@test_group("P1-5: WorkingMemory Isolation")
def test_db_path_contains_session_id():
    """The db_path URI encodes the session_id so it is unique per instance."""
    from engram.memory.working_memory import WorkingMemory
    sid = "mysession_abc123"
    wm = WorkingMemory(db_path=None, session_id=sid)
    assert "mysession_abc123" in wm.db_path or "mysession" in wm.db_path, (
        f"session_id not reflected in db_path: {wm.db_path!r}"
    )


@test_group("P1-5: WorkingMemory Isolation")
def test_db_path_is_uri_not_global_shared():
    """db_path must not be the old global shared URI."""
    wm = _wm()
    assert wm.db_path != "file::memory:?cache=shared", (
        "Old global shared URI still in use — fix not applied"
    )
    assert "mode=memory" in wm.db_path or "memory" in wm.db_path


# ── Write isolation ────────────────────────────────────────────────────────────

@test_group("P1-5: WorkingMemory Isolation")
def test_writes_to_a_invisible_to_b():
    """Messages written to instance A must not appear in instance B."""
    wm_a = _wm()
    wm_b = _wm()

    wm_a.add("user", "Secret message for A only")
    wm_b.add("user", "Secret message for B only")

    a_msgs = [m.content for m in wm_a.get_recent()]
    b_msgs = [m.content for m in wm_b.get_recent()]

    assert all("A only" in c for c in a_msgs), f"A has unexpected content: {a_msgs}"
    assert all("B only" in c for c in b_msgs), f"B has unexpected content: {b_msgs}"
    assert not any("B only" in c for c in a_msgs), "A contaminated by B"
    assert not any("A only" in c for c in b_msgs), "B contaminated by A"


@test_group("P1-5: WorkingMemory Isolation")
def test_message_counts_are_per_instance():
    """get_message_count() reflects each instance's own writes only."""
    wm1 = _wm()
    wm2 = _wm()
    wm3 = _wm()

    for _ in range(3):
        wm1.add("user", "wm1 message")
    for _ in range(7):
        wm2.add("user", "wm2 message")
    # wm3 has no messages

    assert wm1.get_message_count() == 3,  f"wm1: expected 3, got {wm1.get_message_count()}"
    assert wm2.get_message_count() == 7,  f"wm2: expected 7, got {wm2.get_message_count()}"
    assert wm3.get_message_count() == 0,  f"wm3: expected 0, got {wm3.get_message_count()}"


@test_group("P1-5: WorkingMemory Isolation")
def test_token_counts_are_per_instance():
    """get_token_count() reflects only the current instance's tokens."""
    wm1 = _wm()
    wm2 = _wm()

    wm1.add("user", "AAAA")   # 1 token via len//4
    wm1.add("user", "AAAA")

    # wm2 gets many more messages
    for _ in range(20):
        wm2.add("user", "AAAA BBBB CCCC DDDD")  # 4 tokens each

    assert wm1.get_token_count() == 2
    assert wm2.get_token_count() == 80


@test_group("P1-5: WorkingMemory Isolation")
def test_clear_session_does_not_affect_other_instance():
    """clear_session() on instance A must not delete B's messages."""
    wm_a = _wm()
    wm_b = _wm()

    wm_a.add("user", "A message")
    wm_b.add("user", "B message")

    wm_a.clear_session()

    assert wm_a.get_message_count() == 0
    assert wm_b.get_message_count() == 1


@test_group("P1-5: WorkingMemory Isolation")
def test_search_does_not_bleed_across_instances():
    """Keyword search is confined to the calling instance's database."""
    wm1 = _wm()
    wm2 = _wm()

    wm1.add("user", "asyncio event loop management")
    wm2.add("user", "threading and multiprocessing patterns")

    results1 = wm1.search("asyncio")
    results2 = wm2.search("asyncio")  # wm2 has no asyncio content

    assert len(results1) >= 1
    assert len(results2) == 0, f"wm2 found asyncio content it shouldn't have: {results2}"


# ── Thread safety preserved ────────────────────────────────────────────────────

@test_group("P1-5: WorkingMemory Isolation")
def test_thread_safety_within_single_instance():
    """Concurrent writes within one instance produce the correct total count."""
    wm = _wm(max_tokens=1_000_000)
    errors = []

    def writer(tid):
        try:
            for i in range(20):
                wm.add("user", f"Thread {tid} message {i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert not errors, f"Thread errors: {errors}"
    assert wm.get_message_count() == 100


@test_group("P1-5: WorkingMemory Isolation")
def test_concurrent_instances_no_interference():
    """Multiple instances writing concurrently stay isolated."""
    instances = [_wm() for _ in range(4)]
    errors = []

    def write_to(wm, tid):
        try:
            for i in range(25):
                wm.add("user", f"Instance {tid} message {i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write_to, args=(wm, i))
               for i, wm in enumerate(instances)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert not errors, f"Thread errors: {errors}"
    for i, wm in enumerate(instances):
        count = wm.get_message_count()
        assert count == 25, f"Instance {i}: expected 25 messages, got {count}"


# ── ProjectMemory multiple instances ──────────────────────────────────────────

@test_group("P1-5: WorkingMemory Isolation")
def test_two_project_memory_instances_isolated():
    """Two ProjectMemory instances in the same process don't share working memory."""
    from engram.project_memory import ProjectMemory

    with TempDir() as d:
        with ProjectMemory(
            project_id="project_one", project_type="general_assistant",
            base_dir=d, llm_engine=None, session_id=unique_session(),
        ) as pm1, ProjectMemory(
            project_id="project_two", project_type="general_assistant",
            base_dir=d, llm_engine=None, session_id=unique_session(),
        ) as pm2:
            pm1.add_turn("user", "Message for project one")
            pm2.add_turn("user", "Message for project two")

            assert pm1.working.get_message_count() == 1
            assert pm2.working.get_message_count() == 1

            turns1 = [t.content for t in pm1.get_recent_turns()]
            turns2 = [t.content for t in pm2.get_recent_turns()]

            assert all("project one" in c for c in turns1)
            assert all("project two" in c for c in turns2)
            assert not any("project two" in c for c in turns1)
            assert not any("project one" in c for c in turns2)


# ── Special character sanitisation ────────────────────────────────────────────

@test_group("P1-5: WorkingMemory Isolation")
def test_session_id_with_special_chars_sanitised():
    """Session IDs with spaces, slashes, etc. are sanitised in the URI."""
    from engram.memory.working_memory import WorkingMemory
    tricky_ids = [
        "session with spaces",
        "session/with/slashes",
        "session?with=query&chars",
        "session'with\"quotes",
        "normal_session-123",
    ]
    paths = set()
    for sid in tricky_ids:
        wm = WorkingMemory(db_path=None, session_id=sid)
        # URI must not contain the raw problematic chars
        assert " " not in wm.db_path,  f"Space in URI: {wm.db_path!r}"
        assert "//" not in wm.db_path.replace("file:", ""), \
            f"Double slash in URI body: {wm.db_path!r}"
        # Must be usable — adding a message must not raise
        wm.add("user", "test message")
        assert wm.get_message_count() == 1
        paths.add(wm.db_path)

    # Each session_id should produce a distinct URI
    assert len(paths) == len(tricky_ids), "Sanitised URIs collided"


@test_group("P1-5: WorkingMemory Isolation")
def test_same_session_id_same_db():
    """Two instances with the same session_id share the same database (intentional)."""
    from engram.memory.working_memory import WorkingMemory
    sid = unique_session()
    wm1 = WorkingMemory(db_path=None, session_id=sid)
    wm2 = WorkingMemory(db_path=None, session_id=sid)

    assert wm1.db_path == wm2.db_path, "Same session_id should give same db_path"

    wm1.add("user", "Written by wm1")
    # wm2 must see it — same session, same DB
    assert wm2.get_message_count() == 1
    assert wm2.get_recent()[0].content == "Written by wm1"
