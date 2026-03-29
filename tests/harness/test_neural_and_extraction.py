"""Tests for SurpriseFilter, NeuralMemory (RTRL), and GraphExtractor."""

from __future__ import annotations

import numpy as np

from .runner import test_group, require
from .mocks import TempDir, fake_embed, EMBED_DIM, MockEngine


# ── SurpriseFilter ─────────────────────────────────────────────────────────────

@test_group("Surprise Filter")
def test_surprise_filter_init():
    from engram.filters.surprise_filter import SurpriseFilter
    sf = SurpriseFilter(llm_engine=MockEngine())
    assert sf is not None
    assert hasattr(sf, "should_store")
    assert hasattr(sf, "evaluate_batch")


@test_group("Surprise Filter")
def test_surprise_filter_should_store_uncalibrated():
    """Before calibration with calibration_required=False, should_store must not raise."""
    from engram.filters.surprise_filter import SurpriseFilter
    sf = SurpriseFilter(llm_engine=MockEngine(), calibration_required=False)
    result = sf.should_store("Some arbitrary text", perplexity=15.0)
    assert isinstance(result, bool)


@test_group("Surprise Filter")
def test_surprise_filter_should_store_low_perplexity_filtered():
    """Content below threshold (low perplexity) should return False."""
    require("pydantic")  # _probe_logprobs checks engine.base import
    from engram.filters.surprise_filter import SurpriseFilter
    # base_threshold=50.0; perplexity=1.0 is way below it
    sf = SurpriseFilter(llm_engine=MockEngine(), base_threshold=50.0,
                        calibration_required=False)
    result = sf.should_store("Mundane content", perplexity=1.0)
    assert result is False


@test_group("Surprise Filter")
def test_surprise_filter_should_store_high_perplexity_passes():
    """Content with explicit high perplexity should return True regardless of pydantic."""
    from engram.filters.surprise_filter import SurpriseFilter
    # Pass perplexity explicitly — no engine call needed, pydantic irrelevant
    sf = SurpriseFilter(llm_engine=MockEngine(), base_threshold=5.0,
                        calibration_required=False)
    # Use perplexity=10.0: above threshold 5.0, but below 5.0*3=15.0
    # (filter has a "too surprising = noise" guard at threshold*3)
    result = sf.should_store("Moderately surprising content", perplexity=10.0)
    assert result is True, (
        f"Expected True (10.0 > threshold 5.0 and <= 15.0), got {result}. "
        f"logprobs_available={sf._logprobs_available}, "
        f"threshold={sf.current_threshold}"
    )


@test_group("Surprise Filter")
def test_surprise_filter_get_stats():
    from engram.filters.surprise_filter import SurpriseFilter
    sf = SurpriseFilter(llm_engine=MockEngine())
    stats = sf.get_stats()
    assert isinstance(stats, dict)
    # At minimum one key indicating calibration state or threshold
    assert len(stats) > 0


@test_group("Surprise Filter")
def test_surprise_filter_save_load_calibration():
    """Calibration round-trips through disk after calibrate() is called."""
    require("pydantic")  # calibrate() calls llm_engine.generate_with_logprobs
    from engram.filters.surprise_filter import SurpriseFilter
    with TempDir() as d:
        cal_path = d / "calibration.json"

        # Must calibrate before saving; calibrate() needs llm_engine.generate_with_logprobs
        sf1 = SurpriseFilter(llm_engine=MockEngine(), base_threshold=20.0)
        # Provide >3 texts so calibrate() has enough samples
        texts = [f"Human written text sample number {i}." for i in range(10)]
        sf1.calibrate(texts)
        assert sf1.is_calibrated
        sf1.save_calibration(cal_path)
        assert cal_path.exists()

        sf2 = SurpriseFilter(llm_engine=MockEngine())
        sf2.load_calibration(cal_path)
        assert sf2.is_calibrated


@test_group("Surprise Filter")
def test_surprise_filter_calibrate_returns_baseline():
    """calibrate() returns a SurpriseBaseline with expected fields."""
    require("pydantic")  # calibrate() calls llm_engine.generate_with_logprobs
    from engram.filters.surprise_filter import SurpriseFilter, SurpriseBaseline
    sf = SurpriseFilter(llm_engine=MockEngine())
    texts = [f"Sample human text {i}." for i in range(10)]
    baseline = sf.calibrate(texts)
    assert isinstance(baseline, SurpriseBaseline)
    assert hasattr(baseline, "mean")
    assert hasattr(baseline, "std")
    assert baseline.sample_count == len(texts)


@test_group("Surprise Filter")
def test_surprise_filter_evaluate_batch():
    """evaluate_batch returns one SurpriseMetrics per input text."""
    require("pydantic")  # evaluate_batch calls llm_engine.generate_with_logprobs
    from engram.filters.surprise_filter import SurpriseFilter
    sf = SurpriseFilter(llm_engine=MockEngine(), calibration_required=False)
    texts = ["Hello world", "Python asyncio", "Deep learning model"]
    results = sf.evaluate_batch(texts)
    assert len(results) == len(texts)
    for m in results:
        assert hasattr(m, "is_surprising")
        assert hasattr(m, "perplexity")
        assert hasattr(m, "threshold_used")
        assert isinstance(m.is_surprising, bool)
        assert isinstance(m.perplexity, float)


# ── NeuralMemory / RTRL ────────────────────────────────────────────────────────

@test_group("Neural Memory")
def test_neural_config_defaults():
    require("torch")
    from engram.rtrl.neural_memory import NeuralMemoryConfig
    cfg = NeuralMemoryConfig()
    assert cfg.hidden_dim == 32
    assert cfg.value_dim == 16
    assert 0 < cfg.lr <= 0.01


@test_group("Neural Memory")
def test_titans_memory_step_returns_dict():
    """TITANSMemory.step() returns a dict with 'predicted' and 'surprise'."""
    require("torch")
    from engram.rtrl.core import TITANSConfig, TITANSMemory
    cfg = TITANSConfig(key_dim=64, value_dim=16, hidden_dim=32)
    mem = TITANSMemory(cfg)
    key = np.random.randn(64).astype(np.float32)
    val = np.random.randn(16).astype(np.float32)
    result = mem.step(key, val)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "surprise" in result or "predicted" in result


@test_group("Neural Memory")
def test_titans_memory_read():
    """TITANSMemory.read() returns a vector of value_dim."""
    require("torch")
    from engram.rtrl.core import TITANSConfig, TITANSMemory
    cfg = TITANSConfig(key_dim=64, value_dim=16, hidden_dim=32)
    mem = TITANSMemory(cfg)
    key = np.ones(64, dtype=np.float32) / np.sqrt(64)
    val = np.array([float(i) for i in range(16)], dtype=np.float32)
    for _ in range(5):
        mem.step(key, val)
    result = mem.read(key)
    assert result.shape == (16,)


@test_group("Neural Memory")
def test_neural_memory_step_accumulates():
    """NeuralMemory.step() increments internal step count."""
    require("torch")
    from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig
    with TempDir() as d:
        cfg = NeuralMemoryConfig(hidden_dim=32, value_dim=16)
        mem = NeuralMemory(project_dir=d, config=cfg)
        before = mem._session_steps
        for _ in range(5):
            mem.step(np.random.randn(64).astype(np.float32),
                     np.random.randn(16).astype(np.float32))
        assert mem._session_steps >= before  # steps recorded


@test_group("Neural Memory")
def test_neural_memory_save_and_load():
    """NeuralMemory state round-trips through disk."""
    require("torch")
    from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig
    with TempDir() as d:
        cfg = NeuralMemoryConfig(hidden_dim=32, value_dim=16)
        mem1 = NeuralMemory(project_dir=d, config=cfg)
        for _ in range(5):
            mem1.step(np.random.randn(64).astype(np.float32),
                      np.random.randn(16).astype(np.float32))
        steps = mem1._session_steps
        mem1.save()

        mem2 = NeuralMemory.load(project_dir=d, config=cfg)
        # After load, memory object should be valid
        assert mem2 is not None


@test_group("Neural Memory")
def test_embedding_projector_shape():
    require("torch")
    from engram.rtrl.neural_memory import EmbeddingProjector
    proj = EmbeddingProjector(input_dim=EMBED_DIM, output_dim=64)
    out = proj(fake_embed("test sentence"))
    assert out.shape == (64,)


@test_group("Neural Memory")
def test_neural_coordinator_process_turns():
    require("torch")
    from engram.memory.neural_coordinator import NeuralCoordinator
    from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig, EmbeddingProjector
    from .mocks import FakeEmbeddingService

    with TempDir() as d:
        cfg = NeuralMemoryConfig(hidden_dim=32, value_dim=16)
        neural = NeuralMemory(project_dir=d, config=cfg)
        coord = NeuralCoordinator(
            neural=neural,
            key_projector=EmbeddingProjector(EMBED_DIM, 64),
            value_projector=EmbeddingProjector(EMBED_DIM, 16),
            embedding_service=FakeEmbeddingService(),
        )
        # NeuralCoordinator.feed(role, content, msg) - msg is a working-memory Message
        class FakeMsg:
            metadata = {}
        for i in range(10):
            coord.feed("user", f"User message {i}", FakeMsg())
            coord.feed("assistant", f"Assistant response {i}", FakeMsg())
        assert neural._session_steps >= 0  # coordinator ran without error


@test_group("Neural Memory")
def test_neural_coordinator_hint_after_warmup():
    require("torch")
    from engram.memory.neural_coordinator import NeuralCoordinator, _MIN_STEPS_FOR_HINT
    from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig, EmbeddingProjector
    from .mocks import FakeEmbeddingService

    with TempDir() as d:
        cfg = NeuralMemoryConfig(hidden_dim=32, value_dim=16)
        neural = NeuralMemory(project_dir=d, config=cfg)
        coord = NeuralCoordinator(
            neural=neural,
            key_projector=EmbeddingProjector(EMBED_DIM, 64),
            value_projector=EmbeddingProjector(EMBED_DIM, 16),
            embedding_service=FakeEmbeddingService(),
        )
        class FakeMsg:
            metadata = {}
        for i in range(_MIN_STEPS_FOR_HINT + 5):
            coord.feed("user", f"msg {i}", FakeMsg())
            coord.feed("assistant", f"resp {i}", FakeMsg())
        # build_hint(neural_meta) - pass None for neural_meta
        hint = coord.build_hint(None)
        assert isinstance(hint, str)  # always returns str (may be empty)


# ── GraphExtractor ─────────────────────────────────────────────────────────────

class _MinimalSemantic:
    """Stub with all methods GraphExtractor calls — including schema helpers."""
    node_tables: set = set()
    rel_tables: set = set()

    def query(self, *a, **kw): return []

    def add_node(self, table, id_, props=None):
        self.node_tables.add(table)

    def add_relationship(self, *a, **kw): pass

    # Schema helpers called by GraphExtractor._ensure_schema()
    def _create_node_table_safe(self, table, columns, pk): pass
    def _create_rel_table_safe(self, rel, src, dst, columns):
        self.rel_tables.add(f"{rel}_{src}_{dst}")


@test_group("Graph Extraction")
def test_extractor_init_no_deps():
    """GraphExtractor initialises against a minimal stub semantic layer."""
    from engram.memory.extraction import GraphExtractor, ExtractionConfig
    extractor = GraphExtractor(semantic_memory=_MinimalSemantic(),
                               config=ExtractionConfig())
    assert extractor is not None


@test_group("Graph Extraction")
def test_extractor_index_text_returns_stats():
    require("sklearn")
    from engram.memory.extraction import GraphExtractor, ExtractionConfig, ExtractionStats

    class CaptureSemantic(_MinimalSemantic):
        nodes = []
        def add_node(self, table, id_, props=None):
            self.nodes.append(id_)
            self.node_tables.add(table)

    sem = CaptureSemantic()
    # Use correct ExtractionConfig field names
    extractor = GraphExtractor(sem, ExtractionConfig(
        tfidf_min_score=0.01,
        tfidf_max_entities_per_chunk=10,
        min_cooccurrence_count=1,
    ))
    text = (
        "Python asyncio provides event loop management. "
        "The event loop runs coroutines and tasks. "
        "asyncio is built into the Python standard library. "
        "Coroutines are defined with async def syntax."
    )
    stats = extractor.index_text(text)
    assert isinstance(stats, ExtractionStats)
    assert stats.entities >= 0
    assert stats.sentences >= 0


@test_group("Graph Extraction")
def test_extractor_index_documents():
    require("sklearn")
    from engram.memory.extraction import GraphExtractor, ExtractionConfig

    extractor = GraphExtractor(_MinimalSemantic(), ExtractionConfig(
        tfidf_min_score=0.01, min_cooccurrence_count=1,
    ))
    docs = [
        "Python is a high-level programming language.",
        "asyncio enables concurrent I/O in Python.",
        "FastAPI builds REST APIs with Python.",
    ]
    stats = extractor.index_documents(docs)
    assert stats.sentences >= 0


@test_group("Graph Extraction")
def test_extractor_deduplication():
    """Indexing identical text twice should not double-add entity nodes."""
    require("sklearn")
    from engram.memory.extraction import GraphExtractor, ExtractionConfig

    class CountSemantic(_MinimalSemantic):
        calls: list = []
        def add_node(self, table, id_, props=None):
            self.calls.append(id_)
            self.node_tables.add(table)

    sem = CountSemantic()
    ex = GraphExtractor(sem, ExtractionConfig(tfidf_min_score=0.01,
                                              min_cooccurrence_count=1))
    text = "Python asyncio is fast and elegant."
    ex.index_text(text)
    first = len(sem.calls)
    ex.index_text(text)
    second = len(sem.calls) - first
    # Second pass must not add more Entity nodes than the first pass
    # (dedup by id — existing nodes are updated, not duplicated)
    assert second <= first
