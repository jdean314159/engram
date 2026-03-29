# Build Your Own App with Engram

This tutorial walks through the practical steps of building an LLM application
that uses Engram for memory. It assumes you have a local model server running
(Ollama, vLLM, or llama.cpp).

## Step 1: Configure your engine

Create or edit `~/.engram/llm_engines.yaml`:

```yaml
engines:
  local_7b:
    type: ollama
    model: qwen3:8b
    base_url: http://localhost:11434/v1
    max_context: 32768
    system_prompt: You are a helpful assistant.

profiles:
  default_local:
    engines: [local_7b]
    allow_cloud_failover: false
    max_attempts: 2
```

Test the connection:

```bash
engram doctor --profile default_local
```

## Step 2: Create a ProjectMemory

Pick a `project_id` (any string — becomes a directory name) and a
`project_type` that matches your application:

```python
from pathlib import Path
from engram import ProjectMemory
from engram.engine import create_failover_engine

engine = create_failover_engine("default_local")

memory = ProjectMemory(
    project_id="my_coding_assistant",
    project_type="programming_assistant",  # or general_assistant, language_tutor, etc.
    base_dir=Path("~/.engram/projects").expanduser(),
    llm_engine=engine,
)
```

Each `project_id` gets its own isolated directory. No data leaks between
projects.

## Step 3: Choose your integration pattern

**For most apps — use `respond()`:**

```python
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    result = memory.respond(user_input)
    print(f"Assistant: {result['answer']}")

memory.close()
```

**For custom pipelines — use the manual loop:**

```python
memory.add_turn("user", user_input)

built = memory.build_prompt(
    user_message=user_input,
    query=user_input,              # or a refined search query
    max_prompt_tokens=5000,
    reserve_output_tokens=500,
)

reply = engine.generate(built["prompt"])
memory.add_turn("assistant", reply)
```

Use the manual loop when you need to:
- Inspect or modify the assembled context before generating
- Use a retrieval query different from the user message
- Handle multi-step generation with intermediate results

## Step 4: Seed the semantic graph (optional)

If your app has domain knowledge (documentation, previous conversations, source
files), index it into the semantic graph before the user session starts:

```python
# From a text file
with open("python_docs.txt") as f:
    stats = memory.index_text(f.read())
    print(f"Indexed: {stats.entities} entities, {stats.relations} co-occur edges,
           {stats.typed_relations} typed edges")

# From multiple documents
stats = memory.index_documents(["doc1...", "doc2...", "doc3..."])
```

This populates entity nodes and co-occurrence edges in Kuzu without any LLM
calls. The graph is queried during every `build_prompt()` call.

## Step 5: Configure the token budget

The default budget (800 working / 1800 episodic / 1800 semantic / 600 cold)
works well for 4K–8K context models. For larger models:

```python
from engram import TokenBudget

memory = ProjectMemory(
    ...,
    token_budget=TokenBudget(
        working=2000,    # more recent context
        episodic=3000,   # more long-term retrieval
        semantic=1500,
        cold=500,
    ),
)
```

For smaller models or tight latency requirements, reduce episodic and semantic
and increase the reserve for output:

```python
built = memory.build_prompt(
    user_message=text,
    max_prompt_tokens=3000,
    reserve_output_tokens=800,
)
```

## Step 6: Handle sessions

Each user session should have a distinct `session_id`:

```python
import uuid

# New session
memory.new_session(session_id=uuid.uuid4().hex)

# Or at construction time
memory = ProjectMemory(..., session_id=uuid.uuid4().hex)
```

Working memory is isolated per session. Episodic and semantic memory are shared
across sessions (they are the long-term memory).

## Step 7: Inspect what's happening

The sandbox app provides a live inspection UI:

```bash
streamlit run apps/sandbox/app.py
```

Or inspect programmatically:

```python
# What's in the last assembled context?
context = memory.get_context(query="asyncio", max_tokens=2000)
print(context.to_dict())

# What's in each layer?
stats = memory.get_stats()
print(stats)

# Recent experiment runs
from engram.reporting.run_reporter import RunReporter
reporter = RunReporter(memory.experiments)
for run in reporter.recent_run_summaries(limit=5):
    print(run["strategy"], run["status"], run["duration_ms"])
```

## Step 8: Close cleanly

Always close `ProjectMemory` when done. Use a context manager to ensure cleanup
even on exceptions:

```python
with ProjectMemory(...) as memory:
    # your app logic
    pass
# memory.close() called automatically
```

Or call `memory.close()` explicitly in a `finally` block.

---

## Common patterns

### Language tutor

```python
memory = ProjectMemory(
    project_id="spanish_tutor",
    project_type="language_tutor",
    base_dir=Path("~/.engram/projects").expanduser(),
    llm_engine=engine,
)

# After the session, check what errors were corrected:
ctx = memory.get_context(query="grammar errors corrections", max_tokens=500)
```

### Coding assistant that learns preferences

```python
# After a few sessions, the semantic layer will have captured:
# - "I prefer pytest" → PREFERS edge to pytest Entity
# - "I use PyCharm" → USES edge to PyCharm Entity
# These surface automatically in every build_prompt() call.
```

### Multi-project isolation

```python
# Each project_id is fully isolated
work_memory = ProjectMemory(project_id="work", ...)
personal_memory = ProjectMemory(project_id="personal", ...)
# No data crosses between them
```
