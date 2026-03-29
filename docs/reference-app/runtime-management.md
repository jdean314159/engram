# Runtime Management in the Sandbox

## Engine lifecycle

The sandbox manages engine lifetimes via `apps/sandbox/runtime_manager.py`.
Engines marked `managed: true` in YAML can be started and stopped from the
**Models** tab. The sandbox auto-generates the correct launch command from
the YAML config (vllm serve, llama-server, or ollama run).

## Session management

Each browser tab creates its own `UISession` with a unique `session_id`.
`ProjectMemory` is constructed once per session and held in Streamlit
`session_state`. Working memory is isolated per session; episodic and semantic
memory are shared across all sessions for the same `project_id`.

## Diagnostics

The **Diagnostics** tab surfaces:
- Last retrieval trace (which items from which layers entered the prompt)
- Engine failure classification (not_running, wrong_model, oom, context_overflow)
- Recovery guidance specific to the detected failure kind
- Reasoning filter metadata (chars stripped, detection status)
- VRAM usage and model fit estimates

## Known limitation

The sandbox hardcodes `session_id="ui"` in `engine_factory.py`. Multiple
browser tabs pointing to the same Streamlit server will share working memory.
Fix: derive a per-tab session ID from Streamlit's session info.
