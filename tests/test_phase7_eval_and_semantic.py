from dataclasses import dataclass

from engram.eval.profile_runtime import profile_project_memory
from engram.eval.retrieval_eval import RetrievalFixture, evaluate_context
from engram.memory.ingestion import IngestionPolicy, MemoryIngestor
from engram.memory.retrieval import UnifiedRetriever
from engram.memory.working_memory import Message


@dataclass
class FakeEpisode:
    id: str
    text: str
    importance: float = 0.5
    timestamp: float = 1_700_000_000.0


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
        self.node_tables = {"Entity", "Sentence"}

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

    def search_generic_memories(self, query, limit=20, per_type_limit=60, include_graph=True, graph_sentence_limit=6):
        rows = [
            {"type": "fact", "id": "fact1", "content": "The Python service hit an error in asyncio.gather.", "confidence": 0.9, "match_score": 0.75},
            {"type": "graph_context", "id": "sent1", "content": "Related sentence mentions the worker failure and retry loop.", "confidence": 0.62, "match_score": 0.58},
            {"type": "preference", "id": "pref1", "category": "language", "value": "Python", "strength": 0.92, "match_score": 0.8},
        ]
        return rows[:limit]

    def get_stats(self):
        return {"node_counts": {"Fact": len(self.facts), "Preference": len(self.preferences)}}


class FakeWorking:
    def __init__(self):
        self.messages = []

    def add(self, role, content, metadata=None):
        msg = Message(role=role, content=content, token_count=max(1, len(content.split()) // 2), metadata=metadata)
        self.messages.append(msg)
        return msg

    def get_context_window(self, max_tokens=None):
        return list(reversed(self.messages[-3:]))

    def get_recent(self, n=2):
        return list(reversed(self.messages[-n:]))


class FakeCold:
    def retrieve(self, query, n=5):
        return [{"id": "cold1", "text": "Archived note about Python incidents."}]

    def get_stats(self):
        return {"total_rows": 1}


class FakeProjectMemory:
    neural_coord = None  # guard: retrieval.py uses getattr
    def __init__(self):
        self.semantic = FakeSemantic()
        self.episodic = object()
        self.telemetry = FakeTelemetry()
        self.project_id = "proj"
        self.session_id = "sess"
        self.project_type = "language_tutor"
        self.budget = type("Budget", (), {"working": 20, "episodic": 20, "semantic": 20, "cold": 20, "total": 80})()
        self._token_counter = lambda text: max(1, len((text or "").split()) // 2)
        self.working = FakeWorking()
        self.cold = FakeCold()
        self.neural = None
        self._key_projector = None
        self._value_projector = None
        self._episodes = []
        self.retriever = UnifiedRetriever(self)

    def add_turn(self, role, content, metadata=None):
        return self.working.add(role, content, metadata)

    def get_context(self, query=None, max_tokens=80):
        return self.retriever.retrieve(query=query, max_tokens=max_tokens)

    def search_episodes(self, query, n=5, min_importance=0.0, days_back=None):
        return list(self._episodes)[:n]

    def store_episode(self, text, metadata=None, importance=0.5, bypass_filter=False):
        self._episodes.append(FakeEpisode(id=f"ep{len(self._episodes)+1}", text=text, importance=importance))
        return self._episodes[-1].id

    def get_diagnostics_snapshot(self):
        return {"semantic": self.semantic.get_stats(), "cold": self.cold.get_stats()}


def test_language_tutor_policy_extracts_spanish_preference():
    pm = FakeProjectMemory()
    ingestor = MemoryIngestor(pm, policy=IngestionPolicy.for_project_type("language_tutor"))
    decision = ingestor.process_turn(role="user", content="Prefiero Python para practicar asyncio.")
    assert any(write.kind == "preference" for write in decision.semantic_writes)


def test_unified_retriever_uses_search_generic_memories_and_graph_context():
    pm = FakeProjectMemory()
    context = pm.get_context(query="python bug")
    semantic_types = {row.get("type") for row in context.semantic}
    assert "graph_context" in semantic_types
    assert "fact" in semantic_types


def test_profile_and_eval_helpers_return_diagnostics():
    pm = FakeProjectMemory()
    report = profile_project_memory(
        pm,
        turns=[("user", "I use Python for work."), ("assistant", "Noted.")],
        queries=["preferred language"],
    )
    assert report["add_turn"]["count"] == 2
    assert "diagnostics" in report

    fixture = RetrievalFixture(name="pref", query="preferred language", expected_substrings=("python",))
    result = evaluate_context(pm.get_context(query="preferred language"), fixture)
    assert result["passed"] is True
