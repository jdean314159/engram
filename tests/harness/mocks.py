"""
Shared mocks and fixtures for Engram test harness.

All mocks are dependency-free (no chromadb, kuzu, torch required)
so unit tests always run regardless of optional dependencies.
"""

from __future__ import annotations

import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


# ── LLM Engine mock ───────────────────────────────────────────────────────────

class MockEngine:
    """Minimal LLMEngine duck-type — no network calls, no pydantic."""

    model_name = "mock-model"
    served_model_name = "mock-model"
    is_cloud = False
    max_context_length = 4096
    system_prompt = "You are a helpful assistant."

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def compress_prompt(self, prompt: str, target_tokens: int) -> str:
        return prompt[: target_tokens * 4]

    def generate(self, prompt: str, **kwargs) -> str:
        return "Mock response."

    async def agenerate(self, prompt: str, **kwargs) -> str:
        return "Mock response."

    def generate_with_logprobs(self, prompt: str, **kwargs):
        from engram.engine.base import LogprobResult, TokenLogprob
        return LogprobResult(
            text="Mock response.",
            token_logprobs=[TokenLogprob("mock", -2.0) for _ in range(3)],
        )


def make_engine() -> MockEngine:
    return MockEngine()


# ── Embedding mock ────────────────────────────────────────────────────────────

EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension


def fake_embed(text: str) -> np.ndarray:
    """Deterministic fake embedding — same text → same vector."""
    seed = hash(text) % (2 ** 31)
    rng = np.random.RandomState(seed)
    vec = rng.randn(EMBED_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec


class FakeEmbeddingModel:
    """Mimics a SentenceTransformer embedding callable."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, texts: List[str]) -> List[List[float]]:
        self.call_count += len(texts)
        return [fake_embed(t).tolist() for t in texts]

    def encode(self, texts, convert_to_numpy=False, show_progress_bar=False):
        vecs = [fake_embed(t) for t in texts]
        if convert_to_numpy:
            return np.stack(vecs)
        return vecs


class FakeEmbeddingService:
    """EmbeddingService replacement that never touches a real model."""
    available = True

    def embed(self, text: str) -> Optional[np.ndarray]:
        return fake_embed(text)


# ── Ephemeral directory fixtures ──────────────────────────────────────────────

class TempDir:
    """Context manager yielding a temporary directory Path."""

    def __enter__(self) -> Path:
        self._td = tempfile.TemporaryDirectory()
        return Path(self._td.name)

    def __exit__(self, *_):
        # Force garbage collection before directory cleanup so that kuzu
        # connections (which hold file handles inside the temp dir) are
        # released before we try to delete the directory.
        import gc
        gc.collect()
        try:
            self._td.cleanup()
        except Exception:
            pass  # Windows may refuse to delete open files — ignore


# ── Unique session IDs ────────────────────────────────────────────────────────

def unique_session() -> str:
    """Return a globally unique session ID.

    Critical: WorkingMemory uses file::memory:?cache=shared so all
    in-memory instances share one SQLite file. Tests MUST use unique
    session_ids to prevent cross-test state contamination.
    """
    return f"test_{uuid.uuid4().hex}"


# ── ProjectMemory factory ─────────────────────────────────────────────────────

def make_project_memory(base_dir: Path, project_id: str = "test_project",
                        project_type: str = "general_assistant",
                        neural: bool = False) -> Any:
    """Build a ProjectMemory with mock engine and unique session ID."""
    from engram.project_memory import ProjectMemory

    kwargs: Dict[str, Any] = dict(
        project_id=project_id,
        project_type=project_type,
        base_dir=base_dir,
        llm_engine=make_engine(),
        session_id=unique_session(),
    )
    if neural:
        try:
            from engram.rtrl.neural_memory import NeuralMemoryConfig
            kwargs["neural_config"] = NeuralMemoryConfig()
        except ImportError:
            pass

    return ProjectMemory(**kwargs)


# ── Fake SemanticMemory ───────────────────────────────────────────────────────

class FakeSemanticMemory:
    """In-memory stub satisfying SemanticLayerProtocol."""

    def __init__(self):
        self._store: List[Dict[str, Any]] = []
        self.node_tables: set = set()

    def search_generic_memories(self, query: str, *, limit: int = 20,
                                per_type_limit: int = 60,
                                include_graph: bool = True,
                                graph_sentence_limit: int = 6) -> List[Dict[str, Any]]:
        q = query.lower()
        results = []
        for row in self._store:
            text = row.get("text", row.get("value", ""))
            if q in text.lower():
                results.append({**row, "match_score": 0.7})
        return results[:limit]

    def add_fact(self, text: str, metadata: Optional[Dict] = None) -> str:
        row = {"type": "fact", "text": text, "timestamp": time.time(),
               "match_score": 0.5, **(metadata or {})}
        self._store.append(row)
        return str(uuid.uuid4())

    def add_node(self, *a, **kw): pass
    def add_relationship(self, *a, **kw): pass
    def query(self, *a, **kw): return []

    def get_stats(self) -> Dict[str, Any]:
        return {"enabled": True, "status": "stub", "node_count": len(self._store)}


# ── Fake EpisodicMemory ───────────────────────────────────────────────────────

class FakeEpisodicMemory:
    """In-memory stub satisfying the episodic interface."""

    def __init__(self):
        self._episodes: List[Any] = []
        self._id_counter = 0

    def add_episode(self, text: str, metadata: Optional[Dict] = None,
                    session_id: Optional[str] = None, project_id: Optional[str] = None,
                    importance: float = 0.5, episode_id: Optional[str] = None) -> str:
        """Match real EpisodicMemory.add_episode() signature — returns str id."""
        from dataclasses import dataclass

        @dataclass
        class Ep:
            id: str
            text: str
            metadata: dict
            session_id: Optional[str]
            project_id: Optional[str]
            importance: float
            timestamp: float

        self._id_counter += 1
        eid = episode_id or f"ep_{self._id_counter}"
        ep = Ep(
            id=eid, text=text, metadata=metadata or {},
            session_id=session_id, project_id=project_id,
            importance=importance, timestamp=time.time(),
        )
        self._episodes.append(ep)
        return eid  # real API returns str id

    # Alias for backwards compat with tests that use .add()
    def add(self, text: str, **kw) -> Any:
        ep_id = self.add_episode(text, **kw)
        return next(e for e in self._episodes if e.id == ep_id)

    def get_stats(self) -> Dict[str, Any]:
        return {"total_episodes": len(self._episodes)}

    def search(self, query: str, n: int = 5, n_results: int = 0,
               project_id: Optional[str] = None) -> List[Any]:
        """Match real EpisodicMemory.search() — keyword arg is n= not n_results=."""
        limit = n or n_results or 5
        q = query.lower()
        return [ep for ep in self._episodes if q in ep.text.lower()][:limit]

    def get_recent_episodes(self, n: int = 10, days_back: int = 7,
                            project_id: Optional[str] = None) -> List[Any]:
        return list(reversed(self._episodes))[:n]

    # Alias for old tests
    def get_recent(self, n: int = 10, project_id: Optional[str] = None) -> List[Any]:
        return self.get_recent_episodes(n=n, project_id=project_id)

    def delete_episode(self, episode_id: str) -> bool:
        before = len(self._episodes)
        self._episodes = [e for e in self._episodes if e.id != episode_id]
        return len(self._episodes) < before

    def delete(self, episode_id: str) -> bool:
        return self.delete_episode(episode_id)

    def count(self, project_id: Optional[str] = None) -> int:
        return len(self._episodes)
