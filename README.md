# Engram

**Engram** is a local-first memory runtime for LLM applications.

It gives an LLM application persistent, structured memory across sessions
without requiring a hosted memory service or any changes to the model itself.
Engram works with **Ollama**, **vLLM**, **llama.cpp**, **Anthropic Claude**,
**OpenAI**, and any OpenAI-compatible server.

---

## What problem Engram solves

Most LLM applications handle memory in one of two ways:

- Keep a small rolling chat history in the prompt, or
- Retrieve chunks from a single vector store.

Neither ages well. Rolling history evicts older context silently. A single
store forces everything into one bucket with no concept of recency, importance,
or type.

Engram uses a **five-layer memory model** instead. Each layer has a distinct
purpose, a distinct storage backend, and a distinct place in the retrieval
pipeline.

---

## The five memory layers

| Layer | Storage | What it stores | Typical lifetime |
|---|---|---|---|
| **Working** | SQLite | Recent turns in the current session | Minutes to hours |
| **Episodic** | ChromaDB + embeddings | Important past interactions | Sessions to weeks |
| **Semantic** | Kuzu graph DB | Facts, preferences, typed relations | Indefinite |
| **Cold** | SQLite FTS5 | Archived episodes, keyword-searchable | Indefinite |
| **Neural** | File (torch tensors) | Learned associative patterns + surprise signal | Indefinite |

All five layers are optional at the dependency level. Working memory and cold
storage work with no additional packages. Episodic requires `chromadb` and
`sentence-transformers`. Semantic requires `kuzu`. Neural requires `torch`.

---

## Quickest path to value

```python
from pathlib import Path
from engram import ProjectMemory
from engram.engine import create_failover_engine

engine = create_failover_engine("default_local")

with ProjectMemory(
    project_id="my_assistant",
    project_type="general_assistant",
    base_dir=Path("./data/memory"),
    llm_engine=engine,
) as memory:
    # Single-call conversational interface
    result = memory.respond("What is asyncio?")
    print(result["answer"])
```

---

## Core interface: `ProjectMemory`

`ProjectMemory` is the single entry point. It coordinates all five layers,
owns the token budget, and manages the session.

### Two conversation patterns

**Pattern A — single call (recommended for most apps):**

```python
result = memory.respond("Explain async/await in Python.")
print(result["answer"])          # LLM response
print(result["prompt_tokens"])   # tokens used
```

`respond()` handles the full loop: records the user turn, assembles the
memory-enriched prompt, calls the LLM, records the assistant turn, logs the
run to `ExperimentMemory`, and returns the result.

**Pattern B — manual loop (for custom pipelines):**

```python
memory.add_turn("user", user_text)
built = memory.build_prompt(user_message=user_text, max_prompt_tokens=6000,
                            reserve_output_tokens=600)
reply = engine.generate(built["prompt"])
memory.add_turn("assistant", reply)
```

### Public methods

| Method | Description |
|---|---|
| `respond(user_message, ...)` | Full conversational loop in one call |
| `add_turn(role, content)` | Add a turn; feeds neural layer if enabled |
| `get_recent_turns(n)` | Recent working-memory turns |
| `store_episode(text, metadata, importance)` | Manually store an episode |
| `search_episodes(query, n)` | Semantic search over this project's episodes |
| `get_context(query, max_tokens)` | Retrieve context from all layers |
| `build_prompt(user_message, ...)` | Full prompt assembly with pressure valve |
| `new_session(session_id)` | Start a new working-memory session |
| `index_text(text, config)` | Extract entities/relations into semantic graph |
| `index_documents(documents, config)` | Batch graph indexing |
| `export_dataset(path, format, config)` | Export conversations as fine-tuning data |
| `run_maintenance(dry_run)` | Run forgetting policy explicitly |
| `calibrate_surprise_filter(texts)` | Calibrate episodic surprise threshold |
| `get_stats()` | Stats dict for all layers |
| `close()` | Flush and release all resources |

---

## Token budget and pressure valve

```python
from engram import ProjectMemory, TokenBudget

memory = ProjectMemory(
    ...,
    token_budget=TokenBudget(working=1000, episodic=800, semantic=400, cold=400),
)
```

The **pressure valve** fires when the assembled prompt would exceed the
engine's context window. It compresses or truncates the memory block, but
never drops the current user message.

---

## Typed relation extraction (semantic layer)

The semantic layer stores typed relations extracted from conversation text,
with no LLM calls required:

| Relation | Meaning | Example trigger |
|---|---|---|
| `PREFERS` | User prefers X | "I prefer pytest over unittest" |
| `USES` | User uses X | "I use PyCharm as my IDE" |
| `KNOWS_ABOUT` | User knows / works with X | "I work at Anthropic", "FastAPI is a framework" |

Relations are extracted by pattern matching and written to Kuzu during
`index_text()`. They surface in retrieval via `search_generic_memories()`.

```python
from engram.memory.extraction import GraphExtractor, ExtractionConfig

ex = GraphExtractor(memory.semantic, ExtractionConfig(extract_typed=True))
stats = ex.index_text("I prefer pytest. I use PyCharm. FastAPI is a framework.")
print(stats.typed_relations)   # edges written

uses = ex.find_typed_relations("USES", limit=10)
```

Project-specific relation types (`STRUGGLES_WITH`, `CORRECTED`, `LEARNED` for
a language tutor; `DEPENDS_ON`, `PRODUCED` for an agent coordinator) are
added in `_init_project_schema()`.

---

## The forgetting policy

Scores every episode on four factors and archives below-threshold ones to cold
storage:

- **Recency** — exponential decay (half-life 30 days by default)
- **Importance** — the 0–1 score set at episode creation
- **Access frequency** — log-scaled retrieval hit count
- **Neural surprise** — surprise signal at storage time (if neural enabled)

```python
from engram import ForgettingConfig

memory = ProjectMemory(
    ...,
    forgetting_config=ForgettingConfig(
        retention_threshold=0.3, min_age_days=7.0,
        weight_recency=0.35, weight_importance=0.30,
        weight_access=0.25,  weight_surprise=0.10,
    ),
)
result = memory.run_maintenance(dry_run=True)   # preview
```

The policy runs on a daemon thread and never blocks the chat path.

---

## Neural memory (Layer 5)

An optional **subgrouped RTRL network** (based on 1994 AFIT thesis, mapped to the
TITANS architecture). Disabled by default; requires `torch`.

### What it does

The network runs on every (user→assistant) turn pair and produces a **surprise
score** — how unexpected was this pair given what it has learned.

This score is used for:

1. **Episodic gating** via `SurpriseFilter` — only novel turns become episodes
2. **Dynamic episode importance** — high-surprise turns get higher importance
   scores, so the forgetting policy retains them longer
3. **Forgetting policy input** — `weight_surprise` in `ForgettingConfig`

### Signal validation

Before relying on the signal, validate it:

```bash
PYTHONPATH=. python run_tests.py --groups "Neural Memory"
# test_rtrl_surprise_correlates_with_novelty
# reports: Spearman r — should be >= 0.4 to use the signal
```

### Enabling

```python
from engram.rtrl.neural_memory import NeuralMemoryConfig

memory = ProjectMemory(
    ...,
    neural_config=NeuralMemoryConfig(
        enabled=True,
        value_dim=16, hidden_dim=32, key_dim=64,
        embedding_dim=384, lr=0.001, grad_clip_norm=5.0,
    ),
)
```

### Accessors

```python
coord = memory.neural_coord
coord.get_last_surprise()          # float: most recent turn's surprise EMA
coord.get_surprise_ema()           # float: EMA over recent turns
coord.is_warmed_up()               # bool: >= 20 turns, signal is reliable
coord.get_surprise_for_query(text) # float: novelty of arbitrary text
```

---

## Episodic surprise filter

Two rejection gates:

- **Below threshold** → too familiar, don't store
- **Above threshold × 3** → too noisy, don't store
- **In range** → genuinely surprising, store it

```python
memory.calibrate_surprise_filter(human_texts=my_samples)
# Calibration saved to project_dir/calibration.json, auto-loaded next init.
```

---

## Reasoning strategies

```python
from engram.strategies import DirectAnswerStrategy, MultiCandidateStrategy
from engram.strategies import ProposeThenVerifyStrategy
from engram.verifiers.simple import ExactMatchVerifier

result = DirectAnswerStrategy().run(memory, "Explain async/await")
result = MultiCandidateStrategy().run(memory, "Capital of France?", n_candidates=3)
result = ProposeThenVerifyStrategy().run(
    memory, "17 * 24?", n_candidates=3, verifier=ExactMatchVerifier("408")
)
```

All strategies log to `ExperimentMemory` automatically.

---

## Experiment tracking

```python
from engram.reporting.run_reporter import RunReporter

reporter = RunReporter(memory.experiments)
for s in reporter.strategy_summary():
    print(f"{s['strategy']:30s}  runs={s['run_count']}  avg_ms={s['avg_duration_ms']:.0f}")
```

---

## Engine configuration (YAML)

```yaml
engines:
  qwen3_8b:
    type: ollama
    model: qwen3:8b
    base_url: http://localhost:11434/v1
    max_context: 32768

  qwen32b_awq:
    type: vllm
    base_url: http://127.0.0.1:8000/v1
    max_context: 8192
    reasoning_visibility: auto   # strip <think>...</think>

  claude:
    type: anthropic
    model: claude-sonnet-4-5-20250929
    max_context: 200000

  llama_32b_split:
    type: llama_cpp
    gguf_path: /models/qwen2.5-32b-instruct-q4_k_m.gguf
    base_url: http://127.0.0.1:8081/v1
    n_gpu_layers: 40     # RTX 3090: 40 layers ≈ 14GB VRAM

profiles:
  default_local:
    engines: [qwen32b_awq, qwen3_8b]
    allow_cloud_failover: false
    max_attempts: 4
```

---

## Installation

```bash
pip install engram              # core only
pip install "engram[all]"       # all layers
```

| Extra | Adds |
|---|---|
| `engram[episodic]` | chromadb, sentence-transformers, diskcache |
| `engram[semantic]` | kuzu |
| `engram[neural]` | PyTorch |
| `engram[engines]` | Anthropic, OpenAI, vLLM, llama.cpp adapters |
| `engram[ui]` | Streamlit sandbox app |

Minimum Python: 3.10

---

## Testing

```bash
# All tests via the harness runner (preferred — pytest misfires on harness internals)
PYTHONPATH=. python run_tests.py

# Specific group
PYTHONPATH=. python run_tests.py --groups "Neural Memory"

# Fast mode
PYTHONPATH=. python run_tests.py --fast
```

Tests requiring optional deps (torch, chromadb, kuzu) are automatically
skipped when those deps are absent.

---

## Known limitations

- The RTRL network needs ≥ 20 warmup turns before the surprise signal is
  reliable. Disable for short sessions.
- Typed relations (`PREFERS`, `USES`, `KNOWS_ABOUT`) use regex pattern
  matching — they work on first-person declarative statements but miss
  paraphrases.
- `IngestionDecision.semantic_writes` are not yet automatically routed into
  typed graph edges during `add_turn()`. Use `index_text()` explicitly.

---

## License

Apache 2.0
