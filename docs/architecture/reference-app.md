# Sandbox Reference App

The Streamlit app in `apps/sandbox/` demonstrates Engram in a realistic chat
loop and serves as a template for building your own UI.

## Running it

```bash
streamlit run apps/sandbox/app.py
```

## Tabs

| Tab | What it shows |
|---|---|
| Chat | Live conversation with assembled context visible |
| Working Memory | Current session turns and token count |
| Episodic Memory | Stored episodes with importance scores |
| Semantic Memory | Graph node counts, preferences, facts, typed relations |
| Cold Storage | Archived episodes, FTS5 search |
| Neural Memory | Surprise EMA, write ratio, step count |
| Diagnostics | Last retrieval trace, engine failure info, reasoning filter status |
| Models | Engine runtime state, VRAM fit, launch commands |
| Maintenance | Forgetting policy controls (dry-run, live run) |

## What the sandbox validates

Running the sandbox against your local model setup confirms:

- Engine connectivity and failover behavior
- Memory layer population over a real session
- Token budget enforcement and pressure valve behavior
- Retrieval quality (visible in the assembled context tab)
- Forgetting policy (run maintenance after a few sessions)

## Using it as a template

The sandbox's `engine_factory.py` shows the canonical `ProjectMemory`
construction pattern. The chat loop in `app.py` is the reference implementation
for the manual `add_turn()` / `build_prompt()` / `engine.generate()` pattern.

Note: the sandbox chat loop uses the manual pattern rather than `respond()`.
Migrating it to `respond()` would simplify the loop and add automatic
experiment tracking.
