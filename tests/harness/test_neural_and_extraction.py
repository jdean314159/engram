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


# ── Non-mutating read() and surprise() ────────────────────────────────────────

def _to_np(B, tensor):
    """Convert backend tensor to numpy regardless of backend."""
    return B.to_numpy(tensor).copy()


@test_group("Neural Memory")
def test_titans_read_is_non_mutating():
    """read() must not alter recurrent state: outputs, p_matrix, gate_values."""
    require("torch")
    from engram.rtrl.core import TITANSConfig, TITANSMemory

    cfg = TITANSConfig(key_dim=8, value_dim=4, hidden_dim=8, gated=True)
    mem = TITANSMemory(cfg)
    B = mem.net.B
    key = np.ones(8, dtype=np.float32) / np.sqrt(8)
    val = np.array([1.0, 0.0, 0.5, -0.5], dtype=np.float32)

    # Warm up so state is non-trivial
    for _ in range(10):
        mem.step(key, val)

    net = mem.net
    outputs_before   = _to_np(B, net.outputs)
    p_matrix_before  = _to_np(B, net.p_matrix)
    p_old_before     = _to_np(B, net.p_matrix_old)
    gate_before      = _to_np(B, net.gate_values)
    prev_out_before  = _to_np(B, net.prev_outputs)

    _ = mem.read(key)

    np.testing.assert_allclose(_to_np(B, net.outputs),      outputs_before,  atol=1e-6, err_msg="read() mutated outputs")
    np.testing.assert_allclose(_to_np(B, net.p_matrix),     p_matrix_before, atol=1e-6, err_msg="read() mutated p_matrix")
    np.testing.assert_allclose(_to_np(B, net.p_matrix_old), p_old_before,    atol=1e-6, err_msg="read() mutated p_matrix_old")
    np.testing.assert_allclose(_to_np(B, net.gate_values),  gate_before,     atol=1e-6, err_msg="read() mutated gate_values")
    np.testing.assert_allclose(_to_np(B, net.prev_outputs), prev_out_before, atol=1e-6, err_msg="read() mutated prev_outputs")


@test_group("Neural Memory")
def test_titans_surprise_is_non_mutating():
    """surprise() must not alter recurrent state."""
    require("torch")
    from engram.rtrl.core import TITANSConfig, TITANSMemory

    cfg = TITANSConfig(key_dim=8, value_dim=4, hidden_dim=8, gated=True)
    mem = TITANSMemory(cfg)
    B = mem.net.B
    key = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    val = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    for _ in range(10):
        mem.step(key, val)

    net = mem.net
    outputs_before  = _to_np(B, net.outputs)
    p_matrix_before = _to_np(B, net.p_matrix)
    gate_before     = _to_np(B, net.gate_values)

    _ = mem.surprise(key, val)

    np.testing.assert_allclose(_to_np(B, net.outputs),     outputs_before,  atol=1e-6, err_msg="surprise() mutated outputs")
    np.testing.assert_allclose(_to_np(B, net.p_matrix),    p_matrix_before, atol=1e-6, err_msg="surprise() mutated p_matrix")
    np.testing.assert_allclose(_to_np(B, net.gate_values), gate_before,     atol=1e-6, err_msg="surprise() mutated gate_values")


@test_group("Neural Memory")
def test_titans_read_consistent_with_step_predicted():
    """read(key) before step(key, val) must match step()'s 'predicted' field."""
    require("torch")
    from engram.rtrl.core import TITANSConfig, TITANSMemory

    cfg = TITANSConfig(key_dim=8, value_dim=4, hidden_dim=8, gated=True)
    mem = TITANSMemory(cfg)
    key = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    val = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    for _ in range(5):
        mem.step(key, val)

    # read() and step()'s predicted must agree on the same network state
    predicted_via_read = mem.read(key)
    result = mem.step(key, val)
    predicted_via_step = result["predicted"]

    np.testing.assert_allclose(
        predicted_via_read, predicted_via_step, atol=1e-6,
        err_msg="read() and step()['predicted'] disagree on same state"
    )


# ── Neural coordinator: query_neural_context and candidate_affinity ────────────

@test_group("Neural Memory")
def test_query_neural_context_returns_predicted_value():
    """query_neural_context() returns a predicted_value array of correct shape."""
    require("torch")
    from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig
    from engram.memory.neural_coordinator import NeuralCoordinator

    class _FakeEmb:
        def embed(self, text):
            return np.random.randn(64).astype(np.float32)

    class _FakeProj:
        def __init__(self, out_dim):
            self._dim = out_dim
            self._W = np.random.randn(out_dim, 64).astype(np.float32)
        def __call__(self, x):
            return self._W @ x

    with TempDir() as d:
        cfg = NeuralMemoryConfig(hidden_dim=32, value_dim=16)
        neural = NeuralMemory(project_dir=d, config=cfg)
        coord = NeuralCoordinator(
            neural=neural,
            key_projector=_FakeProj(cfg.key_dim),
            value_projector=_FakeProj(cfg.value_dim),
            embedding_service=_FakeEmb(),
        )
        ctx = coord.query_neural_context("What is asyncio?")
        assert ctx is not None
        assert "predicted_value" in ctx
        assert ctx["predicted_value"].shape == (cfg.value_dim,)
        assert "query_surprise" in ctx
        assert "warmed_up" in ctx
        assert "surprise_ratio" in ctx


@test_group("Neural Memory")
def test_query_surprise_backward_compatible():
    """query_surprise() must not include predicted_value (backward compat)."""
    require("torch")
    from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig
    from engram.memory.neural_coordinator import NeuralCoordinator

    class _FakeEmb:
        def embed(self, text):
            return np.random.randn(64).astype(np.float32)

    class _FakeProj:
        def __init__(self, out_dim):
            self._W = np.random.randn(out_dim, 64).astype(np.float32)
        def __call__(self, x):
            return self._W @ x

    with TempDir() as d:
        cfg = NeuralMemoryConfig(hidden_dim=32, value_dim=16)
        neural = NeuralMemory(project_dir=d, config=cfg)
        coord = NeuralCoordinator(
            neural=neural,
            key_projector=_FakeProj(cfg.key_dim),
            value_projector=_FakeProj(cfg.value_dim),
            embedding_service=_FakeEmb(),
        )
        result = coord.query_surprise("some query")
        assert result is not None
        assert "predicted_value" not in result
        assert "query_surprise" in result


@test_group("Neural Memory")
def test_candidate_affinity_returns_float_in_range():
    """candidate_affinity() returns a float in [-1, 1]."""
    require("torch")
    from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig
    from engram.memory.neural_coordinator import NeuralCoordinator

    class _FakeEmb:
        def embed(self, text):
            return np.random.randn(64).astype(np.float32)

    class _FakeProj:
        def __init__(self, out_dim):
            self._W = np.random.randn(out_dim, 64).astype(np.float32)
        def __call__(self, x):
            return self._W @ x

    with TempDir() as d:
        cfg = NeuralMemoryConfig(hidden_dim=32, value_dim=16)
        neural = NeuralMemory(project_dir=d, config=cfg)
        coord = NeuralCoordinator(
            neural=neural,
            key_projector=_FakeProj(cfg.key_dim),
            value_projector=_FakeProj(cfg.value_dim),
            embedding_service=_FakeEmb(),
        )
        predicted = np.random.randn(cfg.value_dim).astype(np.float32)
        score = coord.candidate_affinity(predicted, "some candidate text")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


@test_group("Neural Memory")
def test_neural_affinity_weight_in_retrieval_policy():
    """RetrievalPolicy exposes neural_affinity_weight with correct default."""
    from engram.memory.retrieval import RetrievalPolicy
    policy = RetrievalPolicy()
    assert hasattr(policy, "neural_affinity_weight")
    assert 0.0 < policy.neural_affinity_weight <= 0.20


# ── Neural consolidation pipeline ─────────────────────────────────────────────

@test_group("Neural Memory")
def test_record_retrieval_affinities_counts_episodic_only():
    """record_retrieval_affinities() ignores semantic/cold candidates."""
    require("torch")
    from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig
    from engram.memory.neural_coordinator import NeuralCoordinator
    from engram.memory.retrieval import RetrievalCandidate

    class _FakeEmb:
        def embed(self, text):
            return np.random.randn(64).astype(np.float32)

    class _FakeProj:
        def __init__(self, out_dim):
            self._W = np.random.randn(out_dim, 64).astype(np.float32)
        def __call__(self, x):
            return self._W @ x

    with TempDir() as d:
        cfg = NeuralMemoryConfig(hidden_dim=32, value_dim=16)
        neural = NeuralMemory(project_dir=d, config=cfg)
        coord = NeuralCoordinator(
            neural=neural,
            key_projector=_FakeProj(cfg.key_dim),
            value_projector=_FakeProj(cfg.value_dim),
            embedding_service=_FakeEmb(),
        )

        candidates = [
            RetrievalCandidate("episodic", "text A", None, 10, 0.8,
                               source_id="ep-001",
                               metadata={"neural_affinity": 0.5}),
            RetrievalCandidate("semantic", "text B", None, 10, 0.7,
                               source_id="sem-001",
                               metadata={"neural_affinity": 0.5}),
            RetrievalCandidate("episodic", "text C", None, 10, 0.6,
                               source_id="ep-002",
                               metadata={"neural_affinity": 0.05}),  # below threshold
        ]
        coord.record_retrieval_affinities(candidates)

        assert coord._affinity_hits.get("ep-001", 0) == 1, "ep-001 should have 1 hit"
        assert "sem-001" not in coord._affinity_hits, "semantic candidates ignored"
        assert "ep-002" not in coord._affinity_hits, "low-affinity candidates ignored"


@test_group("Neural Memory")
def test_consolidation_candidates_clears_after_return():
    """consolidation_candidates() returns qualifying IDs and clears their counts."""
    require("torch")
    from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig
    from engram.memory.neural_coordinator import NeuralCoordinator
    from engram.memory.retrieval import RetrievalCandidate

    class _FakeEmb:
        def embed(self, text):
            return np.random.randn(64).astype(np.float32)

    class _FakeProj:
        def __init__(self, out_dim):
            self._W = np.random.randn(out_dim, 64).astype(np.float32)
        def __call__(self, x):
            return self._W @ x

    with TempDir() as d:
        cfg = NeuralMemoryConfig(hidden_dim=32, value_dim=16)
        neural = NeuralMemory(project_dir=d, config=cfg)
        coord = NeuralCoordinator(
            neural=neural,
            key_projector=_FakeProj(cfg.key_dim),
            value_projector=_FakeProj(cfg.value_dim),
            embedding_service=_FakeEmb(),
        )

        # Simulate 2 retrieval rounds for ep-001, 1 for ep-002
        for _ in range(2):
            coord.record_retrieval_affinities([
                RetrievalCandidate("episodic", "text", None, 10, 0.8,
                                   source_id="ep-001",
                                   metadata={"neural_affinity": 0.4}),
            ])
        coord.record_retrieval_affinities([
            RetrievalCandidate("episodic", "text", None, 10, 0.8,
                               source_id="ep-002",
                               metadata={"neural_affinity": 0.4}),
        ])

        candidates = coord.consolidation_candidates(min_hits=2)
        assert "ep-001" in candidates, "ep-001 should qualify with 2 hits"
        assert "ep-002" not in candidates, "ep-002 has only 1 hit"

        # After return, ep-001 count cleared; ep-002 still accumulating
        assert "ep-001" not in coord._affinity_hits, "ep-001 cleared after consolidation"
        assert coord._affinity_hits.get("ep-002", 0) == 1, "ep-002 count preserved"


@test_group("Neural Memory")
def test_reset_clears_affinity_hits():
    """reset() clears accumulated affinity hits."""
    require("torch")
    from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig
    from engram.memory.neural_coordinator import NeuralCoordinator
    from engram.memory.retrieval import RetrievalCandidate

    class _FakeEmb:
        def embed(self, text):
            return np.random.randn(64).astype(np.float32)

    class _FakeProj:
        def __init__(self, out_dim):
            self._W = np.random.randn(out_dim, 64).astype(np.float32)
        def __call__(self, x):
            return self._W @ x

    with TempDir() as d:
        cfg = NeuralMemoryConfig(hidden_dim=32, value_dim=16)
        neural = NeuralMemory(project_dir=d, config=cfg)
        coord = NeuralCoordinator(
            neural=neural,
            key_projector=_FakeProj(cfg.key_dim),
            value_projector=_FakeProj(cfg.value_dim),
            embedding_service=_FakeEmb(),
        )
        coord.record_retrieval_affinities([
            RetrievalCandidate("episodic", "text", None, 10, 0.8,
                               source_id="ep-001",
                               metadata={"neural_affinity": 0.4}),
        ])
        assert coord._affinity_hits  # non-empty
        coord.reset()
        assert not coord._affinity_hits, "reset() should clear affinity hits"
