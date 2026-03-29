# Core Library Architecture

## Package structure

```
engram/
├── project_memory.py       Primary facade — all five layers behind one API
├── memory/
│   ├── working_memory.py   Layer 1: SQLite per-session context
│   ├── episodic_memory.py  Layer 2: ChromaDB semantic search
│   ├── semantic_memory.py  Layer 3: Kuzu graph database
│   ├── semantic_schema.py    Schema per project type (programming, tutor, etc.)
│   ├── semantic_search.py    SemanticSearchMixin: search_preferences, search_facts,
│   │                         search_typed_relations, search_generic_memories
│   ├── cold_storage.py     Layer 4: SQLite FTS5 archive
│   ├── extraction.py       GraphExtractor: TF-IDF/SpaCy → Kuzu, typed relations
│   ├── experiment_memory.py  SQLite run/experiment tracking
│   ├── forgetting.py       Episodic → cold lifecycle policy
│   ├── embedding_cache.py  Two-tier LRU + disk embedding cache
│   ├── embedding_service.py  Lazy shared SentenceTransformer model
│   ├── neural_coordinator.py RTRL coordination: feed, accessors, hint
│   ├── memory_context.py   Narrow dependency bundle for orchestrators
│   ├── ingestion.py        Heuristic ingestion decision pipeline
│   ├── retrieval.py        Cross-layer UnifiedRetriever
│   └── lifecycle.py        Episode promotion and archival rules
├── engine/
│   ├── base.py             LLMEngine ABC
│   ├── router.py           FailoverEngine with circuit breaker
│   ├── vllm_engine.py      vLLM / OpenAI-compat backend
│   ├── ollama_engine.py    Ollama backend
│   ├── claude_engine.py    Anthropic API backend
│   ├── openai_cloud_engine.py OpenAI API backend
│   └── llama_cpp_engine.py llama-server backend
├── strategies/
│   ├── direct_answer.py    Single-call baseline
│   ├── multi_candidate.py  Generate N, select by policy
│   ├── propose_then_verify.py Generate → verify → select
│   └── runner.py           Strategy runner with experiment logging
├── verifiers/              CandidateVerifier ABC + built-ins
├── reporting/run_reporter.py RunReporter over ExperimentMemory
├── filters/surprise_filter.py TITANS-inspired episodic gating
└── rtrl/
    ├── core.py             Subgrouped RTRL / TITANSMemory
    └── neural_memory.py    NeuralMemory wrapper (ProjectMemory-facing)
```

## Key design decisions

**`_build_layers()` is separate from `__init__`** — layer construction is a
pure function that returns a dict of layer instances. This makes it testable
without a full `ProjectMemory` and keeps `__init__` focused on wiring.

**`MemoryContext` is the internal dependency bundle** — orchestrators
(`UnifiedRetriever`, `MemoryLifecycleManager`, `MemoryIngestor`) receive a
`MemoryContext` dataclass rather than a full `ProjectMemory` reference. This
prevents private-attr coupling and enables independent testing.

**Typed relation tables are created by `GraphExtractor._ensure_schema()`** —
not by `SemanticMemory._init_core_schema()`. The semantic memory schema tries
to create them at init time but fails silently because the `Entity` node table
doesn't exist yet. `GraphExtractor._ensure_schema()` runs after `Entity` is
created and forces a retry via `rel_tables.discard()`.

**`_create_rel_table_safe()` does not cache failures** — only successes and
definitive "already exists" responses are cached. This allows `GraphExtractor`
to retry table creation that failed during SemanticMemory init.

## Layer initialization sequence

```
_build_layers():
  1. EmbeddingCache
  2. WorkingMemory (SQLite)
  3. EpisodicMemory (ChromaDB) — optional, degrades gracefully
  4. SemanticMemory (Kuzu) — optional, degrades gracefully on mmap failure
  5. ColdStorage (SQLite FTS5)
  6. EmbeddingService (shared model)
  7. NeuralMemory (torch) — optional, only if neural_config.enabled
  8. ForgettingPolicy
  9. GraphExtractor (if semantic available)
  10. ExperimentMemory

ProjectMemory.__init__():
  11. Unpack layers dict onto self
  12. Complete NeuralCoordinator (needs llm_engine fingerprint)
  13. SurpriseFilter (needs llm_engine)
  14. MemoryContext (narrow bundle for orchestrators)
  15. UnifiedRetriever, MemoryIngestor, MemoryLifecycleManager
```
