from dataclasses import dataclass

from engram.memory.lifecycle import MemoryLifecycleManager
from engram.memory.retrieval import UnifiedRetriever
from engram.rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig


@dataclass
class FakeEpisode:
    id: str
    text: str
    importance: float = 0.5
    timestamp: float = 1_700_000_000.0
    metadata: dict = None
    session_id: str = "sess"


class FakeTelemetry:
    def __init__(self):
        self.events = []

    def emit(self, *args, **kwargs):
        self.events.append((args, kwargs))


class FakeSemantic:
    def __init__(self):
        self.facts = []
        self.preferences = []
        self.events = []

    def add_fact(self, **kwargs):
        self.facts.append(kwargs)

    def add_preference(self, **kwargs):
        self.preferences.append(kwargs)

    def add_event(self, **kwargs):
        self.events.append(kwargs)

    def list_facts(self, limit=100):
        return list(self.facts)[:limit]

    def list_preferences(self, limit=100):
        return list(self.preferences)[:limit]

    def list_events(self, limit=100):
        return list(self.events)[:limit]


class FakeCold:
    def __init__(self):
        self.rows = []

    def archive(self, memories):
        self.rows.extend(memories)
        return len(memories)

    def retrieve(self, query, n=5):
        return self.rows[:n]


class FakeEpisodic:
    def __init__(self, episodes):
        self.episodes = episodes
        self.deleted = []

    def get_recent_episodes(self, n=10, days_back=7, project_id=None):
        return list(self.episodes)[:n]

    def delete_episode(self, episode_id):
        self.deleted.append(episode_id)


class FakeWorking:
    def get_context_window(self, max_tokens=None):
        return []


class FakeProjectMemory:
    neural_coord = None  # guard: retrieval.py uses getattr
    def __init__(self):
        self.project_id = "proj"
        self.session_id = "sess"
        self.semantic = FakeSemantic()
        self.telemetry = FakeTelemetry()
        self.cold = FakeCold()
        self.working = FakeWorking()
        self._token_counter = lambda text: max(1, len((text or '').split()) // 2)
        self.budget = type("Budget", (), {"working": 20, "episodic": 20, "semantic": 20, "cold": 20})()
        self.neural = None
        self._key_projector = None
        self._value_projector = None
        self.episodic = FakeEpisodic([
            FakeEpisode(id="ep1", text="We decided to standardize on Python async patterns.", importance=0.92, metadata={"tag": "decision"}),
            FakeEpisode(id="ep2", text="Minor note from an older session.", importance=0.30, timestamp=1_600_000_000.0, metadata={}),
        ])

    def search_episodes(self, query, n=5, min_importance=0.0, days_back=None):
        return []


def test_lifecycle_promotes_and_archives():
    pm = FakeProjectMemory()
    mgr = MemoryLifecycleManager(pm)
    report = mgr.run_maintenance()
    assert report.promoted_events >= 1
    assert report.promoted_facts >= 1
    assert report.archived_episodes >= 1
    assert pm.semantic.events
    assert pm.semantic.facts
    assert pm.cold.rows


def test_retriever_includes_semantic_events():
    pm = FakeProjectMemory()
    pm.semantic.events.append({"id": "evt1", "summary": "Python async patterns were standardized", "importance": 0.9})
    retriever = UnifiedRetriever(pm)
    context = retriever.retrieve(query="python async patterns", max_tokens=60, cold_fallback=False)
    assert any(row.get("type") == "event" for row in context.semantic)


def test_neural_memory_exposes_role_and_resets_on_fingerprint_change(tmp_path):
    cfg = NeuralMemoryConfig(enabled=True, key_dim=4, value_dim=2, hidden_dim=4, embedding_dim=8, model_fingerprint="model-a")
    mem = NeuralMemory(project_dir=tmp_path, config=cfg)
    desc = mem.describe_role()
    assert desc["memory_role"] == "auxiliary_embedding_memory"
    changed = mem.ensure_compatible("model-b")
    assert changed is True
    assert mem.get_stats()["model_fingerprint"] == "model-b"
