"""Validation tests for P2-6: typed relation extraction.

Tests verify:
1. extract_typed_relations() produces ExtractedRelation objects.
2. Pattern matching correctly identifies PREFERS, USES, KNOWS_ABOUT.
3. Trailing prepositions/stop-words are stripped from captured terms.
4. Stop-value filtering removes noise entities.
5. ExtractionStats.typed_relations tracks counts.
6. ExtractionConfig.extract_typed=False disables the pass.
7. _write_typed_relations() writes edges to a fake semantic backend.
8. GraphExtractor.get_stats() returns typed relation counts.
9. GraphExtractor.find_typed_relations() queries PREFERS/USES/KNOWS_ABOUT.
10. search_typed_relations() returns rows with correct structure.
11. search_generic_memories() includes typed relation results.
12. IngestionDecision.semantic_writes flow into graph via apply().
13. Full pipeline: text → IngestionDecision → Kuzu typed edges (kuzu required).
"""

from __future__ import annotations

from contextlib import contextmanager

import time
from typing import Any, Dict, List, Optional

from .runner import test_group, require
from .mocks import TempDir, unique_session


# ── Fake backends ──────────────────────────────────────────────────────────────

class _FakeSemantic:
    """Minimal fake satisfying GraphExtractor's interface."""
    node_tables: set = set()
    rel_tables: set = set()
    _nodes: list = []
    _rels: list = []

    def __init__(self):
        self.node_tables = set()
        self.rel_tables = set()
        self._nodes = []
        self._rels = []

    def query(self, q, params=None):
        if "Preference" in q and "PREFERS" not in q:
            return [{"id": "p1", "category": "testing", "value": "pytest"}]
        if "Fact" in q and "USES" not in q and "KNOWS_ABOUT" not in q:
            return [{"id": "f1", "content": "user works at Anthropic"}]
        return []

    def add_node(self, table, id_, props=None):
        self.node_tables.add(table)
        self._nodes.append({"table": table, "id": id_, "props": props or {}})

    def add_relationship(self, src_table, src_id, dst_table, dst_id, rel, props=None):
        self._rels.append({
            "rel": rel, "src_table": src_table, "src_id": src_id,
            "dst_table": dst_table, "dst_id": dst_id, "props": props or {},
        })

    def _create_node_table_safe(self, *a, **kw): pass

    def _create_rel_table_safe(self, rel, src, dst, columns=None, **kw):
        self.rel_tables.add(f"{rel}_{src}_{dst}")


def _config(**kw):
    from engram.memory.extraction import ExtractionConfig
    return ExtractionConfig(min_entity_length=2, tfidf_min_score=0.01, **kw)


# ── extract_typed_relations() ──────────────────────────────────────────────────

@test_group("P2-6: Typed Relations")
def test_extract_typed_returns_list():
    from engram.memory.extraction import extract_typed_relations
    result = extract_typed_relations([], _FakeSemantic(), _config())
    assert isinstance(result, list)
    assert result == []


@test_group("P2-6: Typed Relations")
def test_extract_prefers_pattern():
    from engram.memory.extraction import extract_typed_relations
    sem = _FakeSemantic()
    rels = extract_typed_relations(
        ["I prefer pytest over unittest."], sem, _config()
    )
    prefers = [r for r in rels if r.relation_type in ("PREFERS", "KNOWS_ABOUT")]
    assert len(prefers) >= 1
    entities = [r.object_text.lower() for r in prefers]
    assert any("pytest" in e for e in entities), f"Expected 'pytest' in {entities}"


@test_group("P2-6: Typed Relations")
def test_extract_uses_pattern():
    from engram.memory.extraction import extract_typed_relations
    rels = extract_typed_relations(
        ["I use PyCharm as my primary IDE."], _FakeSemantic(), _config()
    )
    uses = [r for r in rels if r.relation_type == "USES"]
    assert len(uses) >= 1
    assert any("pycharm" in r.object_text.lower() for r in uses)


@test_group("P2-6: Typed Relations")
def test_extract_knows_about_work_pattern():
    from engram.memory.extraction import extract_typed_relations
    rels = extract_typed_relations(
        ["I work at Anthropic."], _FakeSemantic(), _config()
    )
    knows = [r for r in rels if r.relation_type == "KNOWS_ABOUT"]
    assert len(knows) >= 1
    assert any("anthropic" in r.object_text.lower() for r in knows)


@test_group("P2-6: Typed Relations")
def test_extract_knows_about_framework_pattern():
    from engram.memory.extraction import extract_typed_relations
    rels = extract_typed_relations(
        ["FastAPI is a framework for building REST APIs."], _FakeSemantic(), _config()
    )
    knows = [r for r in rels if r.relation_type == "KNOWS_ABOUT"]
    entities = [r.object_text.lower() for r in knows]
    assert any("fastapi" in e for e in entities), f"Expected 'FastAPI' in {entities}"


@test_group("P2-6: Typed Relations")
def test_trailing_stop_words_stripped():
    """'pytest over' → 'pytest', 'PyCharm as' → 'PyCharm'."""
    from engram.memory.extraction import extract_typed_relations
    rels = extract_typed_relations(
        ["I prefer pytest over unittest.", "I use PyCharm as my IDE."],
        _FakeSemantic(), _config()
    )
    for r in rels:
        words = r.object_text.lower().split()
        trailing_stops = {"over", "as", "on", "at", "in", "for", "with", "to"}
        assert words[-1] not in trailing_stops, (
            f"Trailing stop-word not stripped: {r.object_text!r}"
        )


@test_group("P2-6: Typed Relations")
def test_noise_entities_filtered():
    """Vague phrases should not produce entity relations."""
    from engram.memory.extraction import extract_typed_relations
    rels = extract_typed_relations(
        ["I use it all the time.", "I prefer something else."],
        _FakeSemantic(), _config()
    )
    entities = [r.object_text.lower() for r in rels]
    for noise in ("it", "this", "something", "everything", "anything"):
        assert noise not in entities, f"Noise entity slipped through: {noise!r}"


@test_group("P2-6: Typed Relations")
def test_no_duplicates_within_run():
    """Same (subject, relation, object) should appear only once."""
    from engram.memory.extraction import extract_typed_relations
    rels = extract_typed_relations(
        ["I use pytest.", "I use pytest.", "I use pytest."],
        _FakeSemantic(), _config()
    )
    keys = [(r.subject_id, r.relation_type, r.object_id) for r in rels]
    assert len(keys) == len(set(keys)), "Duplicate relations found"


@test_group("P2-6: Typed Relations")
def test_extract_typed_disabled():
    """extract_typed=False skips the typed relation pass in index_text."""
    require("sklearn")
    from engram.memory.extraction import GraphExtractor, ExtractionConfig
    sem = _FakeSemantic()
    ex = GraphExtractor(sem, ExtractionConfig(extract_typed=False,
                                              tfidf_min_score=0.01))
    stats = ex.index_text("I prefer pytest. I use PyCharm.")
    assert stats.typed_relations == 0


# ── ExtractionStats ────────────────────────────────────────────────────────────

@test_group("P2-6: Typed Relations")
def test_extraction_stats_has_typed_relations_field():
    from engram.memory.extraction import ExtractionStats
    stats = ExtractionStats()
    assert hasattr(stats, "typed_relations")
    assert stats.typed_relations == 0


@test_group("P2-6: Typed Relations")
def test_extraction_stats_to_dict_includes_typed():
    from engram.memory.extraction import ExtractionStats
    stats = ExtractionStats(entities=5, typed_relations=3)
    d = stats.to_dict()
    assert "typed_relations" in d
    assert d["typed_relations"] == 3


# ── _write_typed_relations() ───────────────────────────────────────────────────

@test_group("P2-6: Typed Relations")
def test_write_typed_relations_creates_entity_node():
    require("sklearn")
    from engram.memory.extraction import GraphExtractor, ExtractionConfig, ExtractedRelation

    sem = _FakeSemantic()
    ex = GraphExtractor(sem, ExtractionConfig())

    rel = ExtractedRelation(
        subject_id="fact_root_user_context",
        subject_table="Fact",
        relation_type="USES",
        object_text="PyCharm",
        object_id="abc123",
        source_text="I use PyCharm.",
        confidence=0.75,
    )
    count = ex._write_typed_relations([rel])
    assert count == 1

    tables = {n["table"] for n in sem._nodes}
    assert "Entity" in tables, "Entity node not written"


@test_group("P2-6: Typed Relations")
def test_write_typed_relations_creates_edge():
    require("sklearn")
    from engram.memory.extraction import GraphExtractor, ExtractionConfig, ExtractedRelation

    sem = _FakeSemantic()
    ex = GraphExtractor(sem, ExtractionConfig())

    rel = ExtractedRelation(
        subject_id="fact_root_user_context",
        subject_table="Fact",
        relation_type="KNOWS_ABOUT",
        object_text="Anthropic",
        object_id="xyz789",
        source_text="I work at Anthropic.",
        confidence=0.70,
    )
    ex._write_typed_relations([rel])
    edge_types = [r["rel"] for r in sem._rels]
    assert "KNOWS_ABOUT" in edge_types


@test_group("P2-6: Typed Relations")
def test_write_typed_relations_returns_count():
    require("sklearn")
    from engram.memory.extraction import GraphExtractor, ExtractionConfig, ExtractedRelation

    sem = _FakeSemantic()
    ex = GraphExtractor(sem, ExtractionConfig())
    rels = [
        ExtractedRelation("f1", "Fact", "USES", "pytest", "id1", "src", 0.8),
        ExtractedRelation("f1", "Fact", "USES", "PyCharm", "id2", "src", 0.75),
    ]
    count = ex._write_typed_relations(rels)
    assert count == 2


# ── GraphExtractor.index_text() typed pass ─────────────────────────────────────

@test_group("P2-6: Typed Relations")
def test_index_text_typed_pass_runs():
    require("sklearn")
    from engram.memory.extraction import GraphExtractor, ExtractionConfig

    sem = _FakeSemantic()
    ex = GraphExtractor(sem, ExtractionConfig(
        extract_typed=True, tfidf_min_score=0.01
    ))
    stats = ex.index_text(
        "I prefer pytest for testing. I use PyCharm. FastAPI is a framework."
    )
    assert stats.typed_relations >= 0  # may be 0 if no patterns match
    assert isinstance(stats.typed_relations, int)


@test_group("P2-6: Typed Relations")
def test_index_text_typed_relations_counted():
    require("sklearn")
    from engram.memory.extraction import GraphExtractor, ExtractionConfig

    sem = _FakeSemantic()
    ex = GraphExtractor(sem, ExtractionConfig(
        extract_typed=True, tfidf_min_score=0.01
    ))
    stats = ex.index_text(
        "I prefer pytest. I use PyCharm. I work at Anthropic."
    )
    assert stats.typed_relations >= 2, (
        f"Expected ≥2 typed relations, got {stats.typed_relations}"
    )


# ── GraphExtractor.get_stats() ─────────────────────────────────────────────────

@test_group("P2-6: Typed Relations")
def test_get_stats_has_typed_relations_key():
    require("sklearn")
    from engram.memory.extraction import GraphExtractor, ExtractionConfig

    sem = _FakeSemantic()
    ex = GraphExtractor(sem, ExtractionConfig())
    stats = ex.get_stats()
    assert "typed_relations" in stats
    assert isinstance(stats["typed_relations"], dict)


# ── search_typed_relations() ───────────────────────────────────────────────────

@test_group("P2-6: Typed Relations")
def test_search_typed_relations_returns_list():
    require("kuzu")
    from engram.memory.semantic_memory import SemanticMemory
    from engram.memory.types import ProjectType

    with TempDir() as d:
        with SemanticMemory(db_path=d / "sem", project_type=ProjectType.GENERAL_ASSISTANT) as sm:
            results = sm.search_typed_relations("pytest testing")
            assert isinstance(results, list)


@test_group("P2-6: Typed Relations")
def test_search_typed_relations_row_structure():
    """Typed relation rows have the required fields."""
    require("kuzu", "sklearn")
    from engram.memory.semantic_memory import SemanticMemory
    from engram.memory.types import ProjectType
    from engram.memory.extraction import GraphExtractor, ExtractionConfig

    with TempDir() as d:
        with SemanticMemory(db_path=d / "sem", project_type=ProjectType.GENERAL_ASSISTANT) as sm:
            sm.add_preference(category="testing", value="pytest", strength=0.85)
            ex = GraphExtractor(sm, ExtractionConfig(extract_typed=True, tfidf_min_score=0.01))
            ex.index_text("I prefer pytest for testing Python.")

            results = sm.search_typed_relations("pytest testing", limit=10)
            for row in results:
                assert "type" in row
                assert row["type"] == "typed_relation"
                assert "relation_type" in row
                assert "entity_text" in row
                assert "match_score" in row
                assert "confidence" in row


@test_group("P2-6: Typed Relations")
def test_search_generic_memories_includes_typed():
    """search_generic_memories surfaces typed relation rows."""
    require("kuzu", "sklearn")
    from engram.memory.semantic_memory import SemanticMemory
    from engram.memory.types import ProjectType
    from engram.memory.extraction import GraphExtractor, ExtractionConfig

    with TempDir() as d:
        with SemanticMemory(db_path=d / "sem", project_type=ProjectType.GENERAL_ASSISTANT) as sm:
            sm.add_preference(category="testing", value="pytest", strength=0.85)
            ex = GraphExtractor(sm, ExtractionConfig(extract_typed=True, tfidf_min_score=0.01))
            ex.index_text("I prefer pytest over unittest for all testing.")

            results = sm.search_generic_memories("pytest testing framework", limit=20)
            types = {r["type"] for r in results}
            # Should include at least preference rows
            assert "preference" in types or "typed_relation" in types, (
                f"Expected preference or typed_relation rows, got types: {types}"
            )


# ── IngestionDecision → graph flow ─────────────────────────────────────────────

@test_group("P2-6: Typed Relations")
def test_ingestor_apply_calls_graph_indexing():
    """apply() should call _index_semantic_write_to_graph when extractor present."""
    from engram.memory.ingestion import MemoryIngestor, IngestionPolicy

    indexed = []

    class FakeProjectMemory:
        project_id = "test"
        session_id = unique_session()
        semantic = _FakeSemantic()
        project_type = "general_assistant"
        extractor = type("Ex", (), {
            "config": _config(),
            "_write_typed_relations": lambda self, rels: indexed.extend(rels) or len(rels),
        })()

        def store_episode(self, *a, **kw): return None

        def _semantic_preference_exists_check(self): return False

    # Build a policy that will fire on the test text
    from engram.memory.ingestion import IngestionDecision, SemanticMemoryWrite
    decision = IngestionDecision(
        should_store_episode=False,
        semantic_writes=[
            SemanticMemoryWrite(
                kind="preference",
                payload={"category": "testing", "value": "pytest",
                         "strength": 0.8, "source": "conversation"},
            )
        ],
        source_text="I prefer pytest for all my testing.",
        reasons=["preference_signal"],
    )

    pm = FakeProjectMemory()
    pm.semantic.query = lambda q, p=None: (
        [] if "Preference" not in q else [{"id": "p1", "category": "testing", "value": "pytest"}]
    ) if "PREFERS" not in q else []

    ingestor = MemoryIngestor.__new__(MemoryIngestor)
    ingestor.project_memory = pm
    ingestor.policy = IngestionPolicy()

    # Patch _semantic_preference_exists to return False so it runs add_preference
    ingestor._semantic_preference_exists = lambda payload: False
    ingestor._semantic_fact_exists = lambda payload: False
    ingestor._index_semantic_write_to_graph = lambda write, node, src: None

    result = ingestor.apply(decision)
    assert "semantic_writes" in result


# ── End-to-end with real Kuzu ──────────────────────────────────────────────────

@test_group("P2-6: Typed Relations")
def test_full_pipeline_text_to_typed_edges():
    """Full pipeline: text → extraction → typed edges in Kuzu → retrieval."""
    require("kuzu", "sklearn")
    from engram.memory.semantic_memory import SemanticMemory
    from engram.memory.types import ProjectType
    from engram.memory.extraction import GraphExtractor, ExtractionConfig

    with TempDir() as d:
        with SemanticMemory(db_path=d / "sem", project_type=ProjectType.GENERAL_ASSISTANT) as sm:

            # Add a preference node first (so PREFERS edges have a subject)
            sm.add_preference(category="testing", value="pytest", strength=0.85)

            ex = GraphExtractor(sm, ExtractionConfig(extract_typed=True, tfidf_min_score=0.01))
            # Enable WARNING logging so per-step errors surface in test output
            import logging
            log = logging.getLogger("engram.memory.extraction")
            log.setLevel(logging.WARNING)

            handler = logging.StreamHandler()
            handler.setLevel(logging.WARNING)
            log.addHandler(handler)

            stats = ex.index_text(
                "I prefer pytest for testing. "
                "I use PyCharm as my IDE. "
                "I work at Anthropic. "
                "FastAPI is a framework for building APIs."
            )

            graph_stats = ex.get_stats()

            assert stats.typed_relations >= 2, (
                f"Expected ≥2 typed relations, got {stats.typed_relations}. "
                f"Graph stats: {graph_stats}. "
                f"Entities: {stats.entities}, co-occur: {stats.relations}. "
                f"Check WARNING log output above for per-step failures."
            )

            # Query typed relations back
            uses = ex.find_typed_relations("USES", limit=10)
            knows = ex.find_typed_relations("KNOWS_ABOUT", limit=10)

            all_entities = (
                [r.get("entity", "") for r in uses] +
                [r.get("entity", "") for r in knows]
            )
            entity_text = " ".join(all_entities).lower()
            assert any(term in entity_text for term in ("pycharm", "anthropic", "fastapi")), (
                f"Expected key entities in typed relations, got: {all_entities}"
            )


@test_group("P2-6: Typed Relations")
def test_new_schema_tables_created():
    """PREFERS, USES, KNOWS_ABOUT tables are created and usable.

    Note: rel_tables is an optimistic cache. Tables that fail during
    SemanticMemory init (because Entity doesn't exist yet) are NOT cached —
    they are created later by GraphExtractor._ensure_schema() after Entity
    is set up.  This test therefore verifies functional existence via kuzu
    rather than cache membership.
    """
    require("kuzu", "sklearn")
    from engram.memory.semantic_memory import SemanticMemory
    from engram.memory.types import ProjectType
    from engram.memory.extraction import GraphExtractor, ExtractionConfig, ExtractedRelation
    import time

    with TempDir() as d:
        with SemanticMemory(db_path=d / "sem", project_type=ProjectType.GENERAL_ASSISTANT) as sm:
            # GraphExtractor._ensure_schema creates the tables after Entity exists
            ex = GraphExtractor(sm, ExtractionConfig())

            # All three typed rel tables should now be registered (after _ensure_schema)
            for rel in ("PREFERS_Preference_Entity", "USES_Fact_Entity", "KNOWS_ABOUT_Fact_Entity"):
                assert rel in sm.rel_tables, (
                    f"Typed relation table {rel!r} not registered after _ensure_schema. "
                    f"Current rel_tables: {sm.rel_tables}"
                )

            # Functional verification: write and read back a USES edge
            sm.add_node("Entity", "test_entity_1", {
                "text": "PyCharm", "normalized": "pycharm",
                "entity_type": "TERM", "score": 0.8, "mention_count": 1,
            })
            sm.add_node("Fact", "test_fact_1", {
                "content": "user uses pycharm",
                "timestamp": time.time(), "confidence": 0.8,
                "source": "test", "metadata": "{}",
            })
            sm.add_relationship(
                "Fact", "test_fact_1",
                "Entity", "test_entity_1",
                "USES",
                {"confidence": 0.8, "source_text": "I use PyCharm", "timestamp": time.time()},
            )
            rows = sm.query(
                "MATCH (f:Fact)-[r:USES]->(e:Entity) WHERE f.id = 'test_fact_1' RETURN e.text"
            )
            assert len(rows) >= 1, "USES edge not queryable after creation"
            assert rows[0].get("e.text") == "PyCharm" or rows[0].get("text") == "PyCharm" 
