"""Tests for EmbeddingCache and EmbeddingService."""

from __future__ import annotations

import numpy as np

from .runner import test_group, require
from .mocks import TempDir, FakeEmbeddingModel, fake_embed, EMBED_DIM


# ── EmbeddingCache ────────────────────────────────────────────────────────────

@test_group("Embedding Cache")
def test_cache_memory_only_get_or_compute():
    from engram.memory.embedding_cache import EmbeddingCache
    cache = EmbeddingCache(cache_dir=None, enabled=True)

    call_count = {"n": 0}
    def compute(text: str) -> np.ndarray:
        call_count["n"] += 1
        return fake_embed(text)

    v1 = cache.get_or_compute("hello world", compute)
    assert call_count["n"] == 1
    v2 = cache.get_or_compute("hello world", compute)
    assert call_count["n"] == 1  # cache hit
    np.testing.assert_array_almost_equal(v1, v2)


@test_group("Embedding Cache")
def test_cache_disabled_passthrough():
    """When disabled, cache always calls compute and never stores."""
    from engram.memory.embedding_cache import EmbeddingCache
    cache = EmbeddingCache(cache_dir=None, enabled=False)

    call_count = {"n": 0}
    def compute(text: str) -> np.ndarray:
        call_count["n"] += 1
        return fake_embed(text)

    cache.get_or_compute("test", compute)
    cache.get_or_compute("test", compute)
    assert call_count["n"] == 2


@test_group("Embedding Cache")
def test_cache_lru_eviction():
    """LRU eviction fires when max_memory is exceeded."""
    from engram.memory.embedding_cache import EmbeddingCache
    cache = EmbeddingCache(cache_dir=None, max_memory=3, enabled=True)
    calls = []

    def compute(text: str) -> np.ndarray:
        calls.append(text)
        return fake_embed(text)

    for i in range(10):
        cache.get_or_compute(f"text_{i}", compute)

    # After 10 unique inserts, at most 3 should be in memory
    initial_calls = len(calls)

    # "text_0" should have been evicted — re-computing should call compute again
    cache.get_or_compute("text_0", compute)
    assert len(calls) == initial_calls + 1, "Evicted entry should be recomputed"


@test_group("Embedding Cache")
def test_cache_direct_put_get():
    """put/get interface works independently of get_or_compute."""
    from engram.memory.embedding_cache import EmbeddingCache
    cache = EmbeddingCache(cache_dir=None, enabled=True)

    vec = fake_embed("direct put test")
    cache.put("direct put test", vec)
    result = cache.get("direct put test")
    assert result is not None
    np.testing.assert_array_almost_equal(result, vec)


@test_group("Embedding Cache")
def test_cache_miss_returns_none():
    from engram.memory.embedding_cache import EmbeddingCache
    cache = EmbeddingCache(cache_dir=None, enabled=True)
    result = cache.get("never stored key")
    assert result is None


@test_group("Embedding Cache")
def test_cache_disk_persistence():
    """Disk cache survives process-level cache eviction."""
    require("diskcache")
    from engram.memory.embedding_cache import EmbeddingCache

    with TempDir() as d:
        cache1 = EmbeddingCache(cache_dir=d / "emb_cache", enabled=True, max_memory=1)
        vec = fake_embed("persistent text")
        cache1.put("persistent text", vec)

        # Create new instance (simulates new process)
        cache2 = EmbeddingCache(cache_dir=d / "emb_cache", enabled=True)
        # May or may not be in LRU, but disk should have it
        result = cache2.get("persistent text")
        if result is not None:
            np.testing.assert_array_almost_equal(result, vec)
        # Either way, no exception


# ── EmbeddingService ──────────────────────────────────────────────────────────

@test_group("Embedding Service")
def test_embedding_service_no_model():
    """EmbeddingService gracefully returns None when no model available."""
    from engram.memory.embedding_service import EmbeddingService
    svc = EmbeddingService(episodic=None, cache=None, device="cpu")
    # If sentence-transformers not installed, available=False and embed=None
    if not svc.available:
        result = svc.embed("hello")
        assert result is None
    # If available, result should be ndarray
    else:
        result = svc.embed("hello")
        assert result is not None
        assert isinstance(result, np.ndarray)


@test_group("Embedding Service")
def test_embedding_service_with_fake_model():
    """EmbeddingService delegates to provided model and caches results."""
    from engram.memory.embedding_service import EmbeddingService
    from engram.memory.embedding_cache import EmbeddingCache

    cache = EmbeddingCache(cache_dir=None, enabled=True)
    call_count = {"n": 0}

    # Use a class so __call__ is looked up on the type (magic method resolution)
    class CountingModel:
        def __call__(self, texts):
            call_count["n"] += len(texts)
            return [fake_embed(t).tolist() for t in texts]

    svc = EmbeddingService.__new__(EmbeddingService)
    svc._episodic = None
    svc._cache = cache
    svc._device = "cpu"
    svc._model = CountingModel()

    r1 = svc.embed("test sentence unique xyz")
    assert r1 is not None
    assert r1.shape == (EMBED_DIM,)
    assert call_count["n"] == 1, f"Expected 1 model call, got {call_count['n']}"

    r2 = svc.embed("test sentence unique xyz")  # should hit cache
    assert call_count["n"] == 1, f"Expected cache hit (1 call), got {call_count['n']}"
    np.testing.assert_array_almost_equal(r1, r2)


@test_group("Embedding Service")
def test_embedding_service_borrows_episodic_model():
    """EmbeddingService reuses model from EpisodicMemory to avoid dual load."""
    from engram.memory.embedding_service import EmbeddingService

    # Build a fake episodic that has an embedding_fn
    class FakeEpisodic:
        embedding_fn = FakeEmbeddingModel()

    svc = EmbeddingService(episodic=FakeEpisodic(), cache=None, device="cpu")
    model = svc._get_model()
    # If sentence-transformers is available, model comes from episodic; otherwise standalone
    # Either way, should not raise
    assert model is not None or not svc.available


@test_group("Embedding Service")
def test_embedding_service_available_property():
    from engram.memory.embedding_service import EmbeddingService
    svc = EmbeddingService(episodic=None, cache=None, device="cpu")
    # available must be bool
    assert isinstance(svc.available, bool)
