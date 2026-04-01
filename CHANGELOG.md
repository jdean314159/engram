# Changelog

## 0.1.18 (2026-04)

### Added
- Normalized runtime metadata surface across engines via `LLMEngine.get_runtime_metadata()`
- `ProjectMemory.get_runtime_metadata()` for diagnostics, UI display, and experiment/report plumbing

### Changed
- Added stable backend labels/runtime-kind hints for Ollama, vLLM, llama.cpp, OpenAI, Claude, Gemini, and failover routing
- Incremented package version after release-readiness polish

---

## 0.1.17 (2026-03)

### Added
- **`GeminiEngine`** — Google Gemini support via the OpenAI-compatible AI Studio
  endpoint (`generativelanguage.googleapis.com/v1beta/openai/`).
  - Thin subclass of `OpenAICloudEngine` — no new protocol handling required
  - Auto-detects context window size from model name (1M tokens for Flash family)
  - Reads `GOOGLE_API_KEY` env var by default
  - Default model: `gemini-2.0-flash` (free tier: 15 RPM, 1M tokens/day)
  - Supports `gemini-2.0-flash`, `gemini-2.0-flash-lite`, `gemini-1.5-pro`,
    `gemini-1.5-flash`
  - Exported from `engram.engine` alongside `OllamaEngine`, `ClaudeEngine`
- **`OpenAICloudEngine`** added to public `__all__` exports

### Changed
- `engram/engine/__init__.py`: `GeminiEngine` and `OpenAICloudEngine` added to
  `__all__` and lazy `__getattr__` dispatch

---

## 0.1.16 (prior)

Initial public release with Working, Episodic, Semantic, Cold, and Neural
memory layers; OllamaEngine, ClaudeEngine, VLLMEngine, LlamaCppEngine adapters;
LANGUAGE_TUTOR and AGENT_SWARM project types; sandbox UI.
