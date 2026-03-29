from dataclasses import dataclass

from engram.memory.ingestion import MemoryIngestor
from engram.memory.retrieval import RetrievalCandidate, UnifiedRetriever
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

    def add_fact(self, **kwargs):
        self.facts.append(kwargs)

    def add_preference(self, **kwargs):
        self.preferences.append(kwargs)

    def list_facts(self, limit=100):
        return [
            {"id": "fact1", "content": "The user uses asyncio.gather for concurrent Python tasks.", "confidence": 0.8},
            {"id": "fact2", "content": "The user enjoys Haskell.", "confidence": 0.4},
        ][:limit]

    def list_preferences(self, limit=100):
        return [
            {"id": "pref1", "category": "language", "value": "Python", "strength": 0.9},
        ][:limit]


class FakeWorking:
    def __init__(self):
        self.messages = [
            Message(role="assistant", content="assistant reply about gather", token_count=4),
            Message(role="user", content="question about asyncio gather", token_count=4),
        ]

    def get_context_window(self, max_tokens=None):
        return list(self.messages)


class FakeCold:
    def retrieve(self, query, n=5):
        return [{"id": "cold1", "text": "Archived note about asyncio gather error handling."}]


class FakeProjectMemory:
    neural_coord = None  # guard: retrieval.py uses getattr
    def __init__(self):
        self.semantic = FakeSemantic()
        self.episodic = object()
        self.telemetry = FakeTelemetry()
        self.project_id = "proj"
        self.session_id = "sess"
        self.budget = type("Budget", (), {"working": 20, "episodic": 20, "semantic": 20, "cold": 20})()
        self._token_counter = lambda text: max(1, len(text.split()) // 2)
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


def test_ingestor_extracts_preference_and_stores_episode():
    pm = FakeProjectMemory()
    ingestor = MemoryIngestor(pm)
    decision = ingestor.process_turn(
        role="user",
        content="Please remember that I prefer Python for asyncio work and I use asyncio.gather heavily.",
        metadata={"remember": True},
    )
    assert decision.should_store_episode is True
    outcome = ingestor.apply(decision)
    assert outcome["episode_id"] == "ep1"
    assert pm.semantic.preferences
    assert pm.semantic.facts


def test_unified_retriever_deduplicates_and_reranks():
    pm = FakeProjectMemory()
    pm._episodes = [
        FakeEpisode(id="ep1", text="The user uses asyncio.gather for concurrent Python tasks.", importance=0.9),
        FakeEpisode(id="ep2", text="Unrelated gardening note.", importance=0.8),
    ]
    retriever = UnifiedRetriever(pm)
    context = retriever.retrieve(query="python asyncio gather", max_tokens=80, cold_fallback=True)
    texts = [ep.text for ep in context.episodic] + [item.get("content", "") for item in context.semantic] + [row.get("text", "") for row in context.cold]
    joined = "\n".join(texts).lower()
    assert "asyncio.gather" in joined
    # duplicate semantic fact should not be added alongside identical episodic text
    matching = [t for t in texts if "concurrent python tasks" in t.lower()]
    assert len(matching) == 1
