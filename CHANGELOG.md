# Changelog

## 0.1.19 (2026-04)

### Fixed
- **`TITANSMemory.read()` and `surprise()` were state-mutating** — both called
  `self.net.forward()` without saving or restoring recurrent state, meaning a
  retrieval-time `query_surprise()` call silently perturbed hidden state that
  the next `step()` depended on.  Both methods now snapshot and restore
  `outputs`, `activations`, `p_matrix`, `p_matrix_old`, `gate_values`,
  `prev_outputs`, and `ln_std` via `try/finally`.  Added
  `ModernSubgroupedRTRL._snapshot_recurrent_state()` /
  `_restore_recurrent_state()` as the supporting infrastructure.
- **`saved_lr` `UnboundLocalError`** in `TITANSMemory.write()` and `step()` —
  the LR-restore guard `if cfg.surprise_modulated_lr:` was broader than the
  save guard `if cfg.surprise_modulated_lr and self._surprise_ema > 1e-10:`,
  causing a `NameError` on the first call after construction or `reset_full()`.
  Both restore conditions narrowed to match.

### Added
- **Neural retrieval reranking** — `UnifiedRetriever` now calls
  `NeuralCoordinator.query_neural_context()` *before* candidate scoring (was
  after selection).  The returned `predicted_value` vector (network's predicted
  response embedding via `read()`) drives a per-candidate cosine-similarity
  score added to both episodic and semantic scoring formulas.
  `RetrievalPolicy.neural_affinity_weight` (default `0.08`) controls the
  contribution.  Active only when the network is warmed up (≥50 steps).
- **`NeuralCoordinator.query_neural_context()`** — replaces the internal
  `query_surprise()` implementation.  Single embed call yields both the
  surprise score and the predicted value vector.  `query_surprise()` retained
  as a backward-compatible wrapper that strips `predicted_value`.
- **`NeuralCoordinator.candidate_affinity(predicted_value, candidate_text)`** —
  embeds a candidate, projects to value space, returns cosine similarity with
  the network's predicted response vector.
- **Neural→semantic consolidation pipeline** — episodes repeatedly retrieved
  with high neural affinity are now promoted to semantic memory:
  - `NeuralCoordinator.record_retrieval_affinities(selected)` accumulates
    per-episode hit counts (affinity threshold: 0.15 cosine similarity).
  - `NeuralCoordinator.consolidation_candidates(min_hits=2)` returns episode
    IDs that qualify and clears their counters.
  - `MemoryLifecycleManager.run_neural_consolidation(neural_coord, episodic)`
    fetches qualifying episodes and promotes them via `promote_episode()` with
    `source="neural_consolidation"`.
  - `ProjectMemory.run_lifecycle_maintenance()` now calls neural consolidation
    when both the neural coordinator and episodic layer are available.
  - `NeuralCoordinator.reset()` clears accumulated affinity hits.
- **`neural_affinity` field** added to episodic and semantic `RetrievalCandidate`
  metadata for observability.

### Tests
- `test_titans_read_is_non_mutating` — verifies 5 state tensors unchanged after `read()`
- `test_titans_surprise_is_non_mutating` — verifies 3 state tensors unchanged after `surprise()`
- `test_titans_read_consistent_with_step_predicted` — `read()` output matches `step()['predicted']` on same state
- `test_query_neural_context_returns_predicted_value` — shape and key checks on new coordinator method
- `test_query_surprise_backward_compatible` — `predicted_value` absent from legacy wrapper
- `test_candidate_affinity_returns_float_in_range` — cosine sim in `[-1, 1]`
- `test_neural_affinity_weight_in_retrieval_policy` — field present with sane default
- `test_record_retrieval_affinities_counts_episodic_only` — semantic/cold/low-affinity candidates ignored
- `test_consolidation_candidates_clears_after_return` — hit counters cleared after qualifying IDs returned
- `test_reset_clears_affinity_hits` — `reset()` empties hit tracker

---

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
