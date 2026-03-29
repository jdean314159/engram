"""Regression tests for Issue 1 refactoring — architecture components.

Tests that the extracted components behave correctly in isolation and that
ProjectMemory correctly wires them together.

CPU-only, no network, no GPU, no LLM calls.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_engine():
    e = MagicMock()
    e.model_name = "test-model"
    e.system_prompt = "You are helpful."
    e.is_cloud = False
    e.max_context_length = 512
    e.count_tokens.side_effect = lambda t: max(1, len(t) // 4)
    e.compress_prompt.side_effect = lambda p, target_tokens: p[:target_tokens * 4]
    return e


def _make_pm(tmpdir):
    from engram.project_memory import ProjectMemory
    return ProjectMemory(
        project_id="arch_test",
        project_type="general",
        base_dir=Path(tmpdir),
        llm_engine=_dummy_engine(),
    )


# ---------------------------------------------------------------------------
# EmbeddingService
# ---------------------------------------------------------------------------

def test_embedding_service_no_model():
    """EmbeddingService returns None when sentence-transformers unavailable."""
    from engram.memory.embedding_service import EmbeddingService
    svc = EmbeddingService(episodic=None, cache=None, device="cpu")
    # sentence-transformers not installed in test env → None
    assert svc.available is False
    result = svc.embed("hello world")
    assert result is None


def test_embedding_service_with_fake_model():
    """EmbeddingService uses cache on second call."""
    from engram.memory.embedding_service import EmbeddingService
    from engram.memory.embedding_cache import EmbeddingCache

    cache = EmbeddingCache(cache_dir=None, enabled=True)  # memory-only

    fake_vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    call_count = {"n": 0}

    class FakeModel:
        def __call__(self, texts):
            call_count["n"] += 1
            return [fake_vec.tolist() for _ in texts]

    svc = EmbeddingService.__new__(EmbeddingService)
    svc._episodic = None
    svc._cache = cache
    svc._device = "cpu"
    svc._model = FakeModel()

    r1 = svc.embed("test text")
    assert call_count["n"] == 1
    np.testing.assert_array_equal(r1, fake_vec)

    r2 = svc.embed("test text")  # should hit cache
    assert call_count["n"] == 1  # no second model call
    np.testing.assert_array_equal(r2, fake_vec)


# ---------------------------------------------------------------------------
# NeuralCoordinator
# ---------------------------------------------------------------------------

def test_resolve_neural_fingerprint():
    from engram.memory.neural_coordinator import resolve_neural_fingerprint

    class Eng:
        model_name = "qwen-7b"

    assert resolve_neural_fingerprint(None) == "no-engine"
    assert resolve_neural_fingerprint(Eng()) == "qwen-7b"


def test_neural_coordinator_build_hint_no_data():
    from engram.memory.neural_coordinator import NeuralCoordinator

    coord = NeuralCoordinator.__new__(NeuralCoordinator)
    coord._neural = MagicMock()
    coord._key_proj = MagicMock()
    coord._val_proj = MagicMock()
    coord._emb = MagicMock()
    coord._pending_user_key = None

    assert coord.build_hint(None) == ""
    assert coord.build_hint({"total_steps": 5}) == ""  # under threshold


def test_neural_coordinator_build_hint_familiar():
    from engram.memory.neural_coordinator import NeuralCoordinator

    coord = NeuralCoordinator.__new__(NeuralCoordinator)
    coord._neural = MagicMock()
    coord._pending_user_key = None

    meta = {"total_steps": 50, "query_surprise": 0.1, "avg_surprise": 0.5}
    hint = coord.build_hint(meta)
    assert "familiar" in hint.lower()


def test_neural_coordinator_build_hint_novel():
    from engram.memory.neural_coordinator import NeuralCoordinator

    coord = NeuralCoordinator.__new__(NeuralCoordinator)
    meta = {"total_steps": 50, "query_surprise": 1.2, "avg_surprise": 0.5}
    hint = coord.build_hint(meta)
    assert "novel" in hint.lower()


def test_neural_coordinator_consecutive_user_turns_logged(capfd=None):
    """Second consecutive user turn overwrites pending key (no crash)."""
    from engram.memory.neural_coordinator import NeuralCoordinator

    fake_neural = MagicMock()
    fake_neural.step.return_value = {"surprise": 0.1, "wrote": True}
    fake_key_proj = MagicMock(return_value=np.array([1.0, 2.0]))
    fake_val_proj = MagicMock(return_value=np.array([3.0, 4.0]))
    fake_emb = MagicMock()
    fake_emb.embed.return_value = np.array([0.5, 0.5])

    coord = NeuralCoordinator(
        neural=fake_neural,
        key_projector=fake_key_proj,
        value_projector=fake_val_proj,
        embedding_service=fake_emb,
    )

    msg = MagicMock()
    msg.metadata = {}
    coord.feed("user", "first question", msg)
    assert coord._pending_user_key is not None

    # Second user turn without assistant reply
    coord.feed("user", "second question", msg)
    assert coord._pending_user_key is not None  # overwritten, not None


# ---------------------------------------------------------------------------
# MemoryContext
# ---------------------------------------------------------------------------

def test_memory_context_session_cell():
    from engram.memory.memory_context import MemoryContext

    cell = ["sess1"]
    ctx = MemoryContext(
        working=MagicMock(),
        episodic=None,
        semantic=None,
        cold=MagicMock(),
        neural_coord=None,
        embedding_service=MagicMock(),
        budget=MagicMock(),
        token_counter=lambda t: 1,
        project_id="p1",
        project_type="general",
        telemetry=MagicMock(),
        search_episodes=MagicMock(),
        store_episode=MagicMock(),
        _session_cell=cell,
    )

    assert ctx.session_id == "sess1"
    cell[0] = "sess2"
    assert ctx.session_id == "sess2"  # automatically reflects change


def test_memory_context_session_setter():
    from engram.memory.memory_context import MemoryContext

    cell = ["default"]
    ctx = MemoryContext(
        working=MagicMock(), episodic=None, semantic=None,
        cold=MagicMock(), neural_coord=None, embedding_service=MagicMock(),
        budget=MagicMock(), token_counter=lambda t: 1,
        project_id="p", project_type="general",
        telemetry=MagicMock(), search_episodes=MagicMock(),
        store_episode=MagicMock(), _session_cell=cell,
    )
    ctx.session_id = "new_session"
    assert cell[0] == "new_session"


# ---------------------------------------------------------------------------
# ProjectMemory integration
# ---------------------------------------------------------------------------

def test_project_memory_has_ctx():
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)
        from engram.memory.memory_context import MemoryContext
        assert isinstance(pm._ctx, MemoryContext)


def test_orchestrators_hold_ctx():
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)
        assert pm.retriever.project_memory is pm._ctx
        assert pm.ingestor.project_memory is pm._ctx
        assert pm.lifecycle.project_memory is pm._ctx


def test_new_session_syncs_cell():
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)
        pm.new_session("s2")
        assert pm._ctx.session_id == "s2"
        assert pm.session_id == "s2"


def test_new_session_syncs_working():
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)
        pm.new_session("s3")
        assert pm._ctx.working is pm.working


def test_build_prompt_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)
        pm.add_turn("user", "hello")
        pm.add_turn("assistant", "hi")
        result = pm.build_prompt("what next?", max_prompt_tokens=256,
                                 reserve_output_tokens=64)
        assert isinstance(result["prompt"], str)
        assert "what next?" in result["prompt"]
        assert isinstance(result["compressed"], bool)


def test_get_stats_no_neural():
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)
        stats = pm.get_stats()
        assert "working" in stats
        assert "embedding_cache" in stats
        assert "neural" not in stats  # neural disabled


def test_close_does_not_raise():
    with tempfile.TemporaryDirectory() as td:
        pm = _make_pm(td)
        pm.close()  # should not raise


# ---------------------------------------------------------------------------
# _build_layers
# ---------------------------------------------------------------------------

def test_build_layers_minimal():
    """_build_layers with no optional deps returns sensible defaults."""
    from engram.project_memory import _build_layers, TokenBudget

    with tempfile.TemporaryDirectory() as td:
        result = _build_layers(
            project_dir=Path(td),
            project_id="layers_test",
            project_type="general",
            session_id="s1",
            budget=TokenBudget(),
            token_counter=lambda t: len(t) // 4,
            neural_config=None,
            forgetting_config=None,
        )
        assert result["working"] is not None
        assert result["cold"] is not None
        assert result["episodic"] is None   # chromadb not installed
        assert result["semantic"] is None   # kuzu not installed
        assert result["neural"] is None     # not requested
        assert result["embedding_cache"] is not None
        assert result["forgetting"] is not None
        result["working"].close()
        result["cold"].close()
        result["embedding_cache"].close()
        result["forgetting"].close()
