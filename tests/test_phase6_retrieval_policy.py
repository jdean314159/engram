from dataclasses import dataclass

from engram.memory.ingestion import MemoryIngestor
from engram.memory.retrieval import UnifiedRetriever
from engram.memory.working_memory import Message


@dataclass
class FakeEpisode:
    id: str
    text: str
    importance: float = 0.5
    timestamp: float = 1_700_000_000.0


class FakeTelemetry:
    def emit(self, *args, **kwargs):
        return None


class FakeSemantic:
    def __init__(self):
        self.facts = []
        self.preferences = []
        self.events = []

    def add_fact(self, **kwargs):
        self.facts.append(kwargs)

    def add_preference(self, **kwargs):
        self.preferences.append(kwargs)

    def list_facts(self, limit=100):
        return list(self.facts)[:limit]

    def list_preferences(self, limit=100):
        return list(self.preferences)[:limit]

    def list_events(self, limit=100):
        return list(self.events)[:limit]


class FakeWorking:
    def get_context_window(self, max_tokens=None):
        return [Message(role="user", content="current question", token_count=4)]


class FakeCold:
    def retrieve(self, query, n=5):
        return [{"id": "cold1", "text": "Archived issue triage notes for Python service failures."}]


class FakeProjectMemory:
    neural_coord = None  # guard: retrieval.py uses getattr
    def __init__(self):
        self.semantic = FakeSemantic()
        self.episodic = object()
        self.telemetry = FakeTelemetry()
        self.project_id = "proj"
        self.session_id = "sess"
        self.budget = type("Budget", (), {"working": 20, "episodic": 20, "semantic": 20, "cold": 20})()
        self._token_counter = lambda text: max(1, len((text or "").split()) // 2)
        self.working = FakeWorking()
        self.cold = FakeCold()
        self.neural = None
        self._key_projector = None
        self._value_projector = None
        self._episodes = []

    def search_episodes(self, query, n=5, min_importance=0.0, days_back=None):
        return list(self._episodes)[:n]

    def store_episode(self, text, metadata=None, importance=0.5, bypass_filter=False):
        self._episodes.append(FakeEpisode(id=f"ep{len(self._episodes)+1}", text=text, importance=importance))
        return self._episodes[-1].id


def test_retrieval_expands_query_terms_and_finds_semantic_fact():
    pm = FakeProjectMemory()
    pm.semantic.facts.append({
        "id": "fact1",
        "content": "The service hit an error when the Python worker crashed.",
        "confidence": 0.9,
    })
    retriever = UnifiedRetriever(pm)
    context = retriever.retrieve(query="python bug", max_tokens=80, cold_fallback=False)
    assert any("error" in row.get("content", "").lower() for row in context.semantic)



def test_retrieval_prefers_single_best_preference_per_category():
    pm = FakeProjectMemory()
    pm.semantic.preferences.extend([
        {"id": "p1", "category": "language", "value": "Python", "strength": 0.9, "timestamp": 1_800_000_000.0},
        {"id": "p2", "category": "language", "value": "Java", "strength": 0.3, "timestamp": 1_600_000_000.0},
    ])
    retriever = UnifiedRetriever(pm)
    context = retriever.retrieve(query="preferred language", max_tokens=80, cold_fallback=False)
    prefs = [row for row in context.semantic if row.get("type") == "preference"]
    assert len(prefs) == 1
    assert prefs[0]["value"] == "Python"



def test_ingestion_skips_ephemeral_and_dedupes_semantic_writes():
    pm = FakeProjectMemory()
    pm.semantic.facts.append({
        "id": "existing-fact",
        "content": "asyncio.gather heavily",
        "confidence": 0.8,
    })
    ingestor = MemoryIngestor(pm)
    decision = ingestor.process_turn(
        role="user",
        content="Temporary test message, do not remember this. I use asyncio.gather heavily.",
    )
    assert decision.should_store_episode is False
    assert decision.semantic_writes == []
