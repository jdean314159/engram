# Getting Started with Engram

Engram is a memory runtime for LLM applications. It handles what to remember,
how to retrieve it, and how to fit it into a prompt without blowing the token
budget.

## Before you start

You need a running local model backend. Engram works with Ollama, vLLM,
llama.cpp, or any OpenAI-compatible server. Configure your backend in
`~/.engram/llm_engines.yaml` (see `docs/tutorials/build-your-own-app.md`).

## The single fastest path

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
    result = memory.respond("What is asyncio?")
    print(result["answer"])
```

`respond()` does everything: records the user turn, retrieves relevant memory
from all layers, builds the prompt, calls the LLM, records the assistant reply,
and logs the run for later analysis.

## What you get for free

From the first call, Engram is already:

- Keeping recent turns in **working memory** (SQLite, always-on)
- Routing interesting exchanges into **episodic memory** (ChromaDB, if installed)
- Running a **forgetting policy** that archives low-value episodes over time
- Logging every call to **ExperimentMemory** for analysis

As the session grows, Engram automatically surfaces relevant past exchanges
in each new prompt, within the configured token budget.

## The four project types

Each project type pre-configures the semantic schema for a specific domain:

| Type | Semantic schema adds |
|---|---|
| `general_assistant` | Facts, preferences, events |
| `programming_assistant` | Concepts, code snippets, bugs, API knowledge |
| `language_tutor` | Vocabulary, grammar topics, error patterns |
| `file_organizer` | File types, organisation rules, location patterns |

The project type affects what gets indexed into the semantic graph and how
retrieval is weighted.

## Next steps

- **Understand the memory flow**: `docs/concepts/memory-flow.md`
- **Build an integration**: `docs/tutorials/build-your-own-app.md`
- **Inspect behavior live**: `streamlit run apps/sandbox/app.py`
- **Run the test suite**: `PYTHONPATH=. python run_tests.py`
