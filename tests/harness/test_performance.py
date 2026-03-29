"""Performance benchmarks for Engram components.

Documented latency targets:
  - Working memory add/search:   < 5ms
  - Cold storage archive/search: < 20ms
  - Embedding cache hit:         < 1ms
  - Episodic search (cold):      < 200ms (first hit loads model)
  - Full context assembly:       < 50ms

Tests fail only on extreme violations (10× target) to avoid CI flakiness
from disk/model warmup variability.
"""

from __future__ import annotations

import time
from statistics import mean
from typing import Callable, List

from .runner import test_group, require
from .mocks import TempDir, fake_embed, unique_session


def _measure_ms(fn: Callable, n: int = 20) -> List[float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _p95(times: List[float]) -> float:
    return sorted(times)[int(len(times) * 0.95)]


def _assert_p95(times: List[float], limit_ms: float, label: str) -> None:
    p = _p95(times)
    avg = mean(times)
    assert p <= limit_ms, (
        f"{label}: p95={p:.1f}ms exceeds limit={limit_ms}ms (avg={avg:.1f}ms)"
    )


def _report(label: str, times: List[float]) -> None:
    print(f"\n           {label}: avg={mean(times):.2f}ms  p95={_p95(times):.2f}ms")


# ── Working memory ─────────────────────────────────────────────────────────────

@test_group("Performance")
def test_perf_working_memory_add():
    from engram.memory.working_memory import WorkingMemory
    wm = WorkingMemory(db_path=None, session_id=unique_session(), max_tokens=1_000_000)
    for i in range(5): wm.add("user", f"warmup {i}")

    times = _measure_ms(lambda: wm.add("user", "Benchmark message content here"))
    _assert_p95(times, 50, "WM.add")
    _report("WM.add", times)


@test_group("Performance")
def test_perf_working_memory_search():
    from engram.memory.working_memory import WorkingMemory
    wm = WorkingMemory(db_path=None, session_id=unique_session(), max_tokens=1_000_000)
    for i in range(200):
        wm.add("user", f"Message {i} about asyncio and Python concurrency")

    times = _measure_ms(lambda: wm.search("asyncio", limit=10))
    _assert_p95(times, 50, "WM.search")
    _report("WM.search", times)


@test_group("Performance")
def test_perf_working_memory_get_recent():
    from engram.memory.working_memory import WorkingMemory
    wm = WorkingMemory(db_path=None, session_id=unique_session(), max_tokens=1_000_000)
    for i in range(500): wm.add("user", f"msg {i}")

    times = _measure_ms(lambda: wm.get_recent(n=20))
    _assert_p95(times, 50, "WM.get_recent(20)")
    _report("WM.get_recent", times)


@test_group("Performance")
def test_perf_working_memory_context_window():
    from engram.memory.working_memory import WorkingMemory
    wm = WorkingMemory(db_path=None, session_id=unique_session(), max_tokens=1_000_000)
    for i in range(500): wm.add("user", f"msg {i}")

    times = _measure_ms(lambda: wm.get_context_window(max_tokens=1000))
    _assert_p95(times, 100, "WM.get_context_window")
    _report("WM.context_window", times)


# ── Cold storage ───────────────────────────────────────────────────────────────

@test_group("Performance")
def test_perf_cold_storage_archive():
    from engram.memory.cold_storage import ColdStorage
    with TempDir() as d:
        cs = ColdStorage(db_path=d / "cold.db")
        docs = [{"text": "Benchmark cold storage text for performance",
                 "project_id": "p"}]
        times = _measure_ms(lambda: cs.archive(docs), n=30)
        _assert_p95(times, 200, "ColdStorage.archive")
        _report("Cold.archive", times)


@test_group("Performance")
def test_perf_cold_storage_retrieve():
    from engram.memory.cold_storage import ColdStorage
    with TempDir() as d:
        cs = ColdStorage(db_path=d / "cold.db")
        docs = [{"text": f"Doc {i} about Python asyncio concurrency patterns",
                 "project_id": "p"} for i in range(50)]
        cs.archive(docs)

        times = _measure_ms(lambda: cs.retrieve("asyncio", n=10, project_id="p"))
        _assert_p95(times, 200, "ColdStorage.retrieve")
        _report("Cold.retrieve", times)


# ── Embedding cache ────────────────────────────────────────────────────────────

@test_group("Performance")
def test_perf_embedding_cache_hit():
    from engram.memory.embedding_cache import EmbeddingCache
    cache = EmbeddingCache(cache_dir=None, enabled=True)
    vec = fake_embed("cache hit benchmark text")
    cache.put("cache hit benchmark text", vec)

    times = _measure_ms(lambda: cache.get("cache hit benchmark text"), n=100)
    _assert_p95(times, 5, "EmbeddingCache.get (hit)")
    _report("Cache.hit", times)


# ── Token counting ─────────────────────────────────────────────────────────────

@test_group("Performance")
def test_perf_token_counting():
    require("pydantic")  # engine.base imports pydantic
    from engram.engine.base import _count_tokens
    text = "This is a benchmark sentence for token counting performance testing. " * 10
    times = _measure_ms(lambda: _count_tokens(text), n=200)
    _assert_p95(times, 50, "_count_tokens")
    _report("token_count", times)


# ── Full context assembly ──────────────────────────────────────────────────────

@test_group("Performance")
def test_perf_build_prompt():
    from engram.project_memory import ProjectMemory
    from .mocks import MockEngine
    with TempDir() as d:
        with ProjectMemory(
            project_id="perf_test",
            project_type="general_assistant",
            base_dir=d,
            llm_engine=MockEngine(),
            session_id=unique_session(),
        ) as pm:
            for i in range(50):
                pm.add_turn("user", f"Message {i}: asyncio event loop pattern")

            times = _measure_ms(
                lambda: pm.build_prompt("asyncio", max_prompt_tokens=1000), n=20
            )
            _assert_p95(times, 500, "build_prompt (working only)")
            _report("build_prompt", times)


# ── Episodic (optional live) ───────────────────────────────────────────────────

@test_group("Performance")
def test_perf_episodic_search():
    require("chromadb", "sentence_transformers")
    from engram.memory.episodic_memory import EpisodicMemory
    with TempDir() as d:
        em = EpisodicMemory(persist_dir=d / "ep", collection_name="perf",
                            embedding_device="cpu")
        for i in range(30):
            em.add_episode(f"Episode {i}: Python asyncio event loop", project_id="p")

        em.search("asyncio", n=5, project_id="p")  # warmup

        times = _measure_ms(
            lambda: em.search("asyncio concurrency", n=5, project_id="p"),
            n=10
        )
        _assert_p95(times, 500, "EpisodicMemory.search")
        _report("Episodic.search", times)
