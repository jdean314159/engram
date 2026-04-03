"""Neural memory coordinator for Engram.

Encapsulates all RTRL/TITANS neural memory interactions that were previously
scattered across ProjectMemory:

  - Projecting embeddings to key/value space
  - Buffering user keys for paired user→assistant learning steps
  - Feeding turns into the NeuralMemory layer
  - Building familiarity/novelty hints for prompt injection
  - Computing query surprise for retrieval ranking

This is a pure coordinator — it owns no storage.  The NeuralMemory layer
it wraps owns persistence (weights + hidden state saved to disk on close).

Author: Jeffrey Dean
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimum steps before the network's signal is considered meaningful.
_MIN_STEPS_FOR_HINT = 50   # EMA with alpha=0.05 needs ~50 steps to stabilize
_FAMILIAR_RATIO = 0.7
_NOVEL_RATIO = 1.5


def resolve_neural_fingerprint(llm_engine: Optional[Any]) -> str:
    """Derive a stable string ID from the active LLM engine.

    Used to detect when the underlying model has changed between sessions,
    which would invalidate stored neural weights.
    """
    if llm_engine is None:
        return "no-engine"
    for attr in ("model_name", "served_model_name", "profile_name", "engine_name"):
        value = getattr(llm_engine, attr, None)
        if value:
            return str(value)
    return llm_engine.__class__.__name__


class NeuralCoordinator:
    """Coordinates RTRL neural memory for one ProjectMemory instance.

    Args:
        neural:            NeuralMemory instance (already initialised).
        key_projector:     EmbeddingProjector for user-turn keys.
        value_projector:   EmbeddingProjector for assistant-turn values.
        embedding_service: EmbeddingService used to embed text.
    """

    def __init__(
        self,
        neural: Any,               # NeuralMemory
        key_projector: Any,        # EmbeddingProjector
        value_projector: Any,      # EmbeddingProjector
        embedding_service: Any,    # EmbeddingService
    ) -> None:
        self._neural = neural
        self._key_proj = key_projector
        self._val_proj = value_projector
        self._emb = embedding_service
        self._pending_user_key = None  # buffered user embedding, waiting for assistant reply
        # Per-episode affinity hit counter for consolidation.
        # Incremented when a candidate is selected with neural_affinity above threshold.
        self._affinity_hits: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Conversation feeding
    # ------------------------------------------------------------------

    def feed(self, role: str, content: str, msg: Any) -> None:
        """Feed one conversation turn into neural memory.

        User turns are buffered as a pending key.  When the next assistant
        turn arrives, the (user_key → assistant_value) pair is used to
        step the RTRL network.  Surprise is attached to the message metadata.

        Args:
            role:    "user" | "assistant" | "system"
            content: Turn text.
            msg:     Working-memory Message object (metadata updated in-place).
        """
        embedding = self._emb.embed(content)
        if embedding is None:
            return

        if role == "user":
            if self._pending_user_key is not None:
                logger.debug(
                    "NeuralCoordinator: pending user key overwritten "
                    "(consecutive user turns without assistant reply)"
                )
            self._pending_user_key = self._key_proj(embedding)
            value = self._val_proj(embedding)
            result = self._neural.step(self._pending_user_key, value)
            logger.debug(
                "Neural (user self-assoc): surprise=%.4f wrote=%s",
                result["surprise"], result["wrote"],
            )

        elif role == "assistant" and self._pending_user_key is not None:
            value = self._val_proj(embedding)
            result = self._neural.step(self._pending_user_key, value)
            self._pending_user_key = None  # consumed

            if msg.metadata is None:
                msg.metadata = {}
            msg.metadata["neural_surprise"] = float(result["surprise"])
            msg.metadata["neural_wrote"] = result["wrote"]
            logger.debug(
                "Neural (user→assistant): surprise=%.4f wrote=%s",
                result["surprise"], result["wrote"],
            )
        # system turns: ignored

    # ------------------------------------------------------------------
    # Prompt hint
    # ------------------------------------------------------------------

    def build_hint(self, neural_meta: Optional[Dict]) -> str:
        """Return a familiarity/novelty hint string for prompt injection.

        Returns empty string when the network hasn't seen enough data or
        the signal isn't distinctive enough to be worth injecting.

        Args:
            neural_meta: The ``neural_meta`` dict from a ContextResult, or None.
        """
        if neural_meta is None:
            return ""

        total_steps = neural_meta.get("total_steps", 0)
        if total_steps < _MIN_STEPS_FOR_HINT:
            return ""

        query_surprise = neural_meta.get("query_surprise", 0.0)
        avg_surprise = neural_meta.get("avg_surprise", 0.0)
        if avg_surprise <= 0:
            return ""

        ratio = query_surprise / avg_surprise

        if ratio < _FAMILIAR_RATIO:
            return (
                f"This interaction pattern is familiar (confidence: high, "
                f"based on {total_steps} prior exchanges). "
                f"Prioritize consistency with previous responses on this topic."
            )
        if ratio > _NOVEL_RATIO:
            return (
                f"This appears to be a novel topic or unusual framing "
                f"(surprise: {ratio:.1f}x above average). "
                f"No strong prior pattern detected; reason carefully."
            )
        return ""

    def warm_up_from_history(
        self,
        episodes: List[Any],
        max_episodes: int = 60,
    ) -> int:
        """Pre-warm the RTRL network from past episodic memory.

        Replays recent episodes through the network as synthetic turns before
        the live conversation begins.  This gives the EMA baseline a head start
        so the surprise signal is meaningful from the first real turn rather
        than requiring 50+ live turns to stabilize.

        Episodes are played in chronological order (oldest first) so the
        network builds up familiarity progressively.  Only the episodic content
        is used — no message metadata is modified and no new episodes are stored.

        Args:
            episodes: List of Episode objects from episodic memory (most recent
                      first, as returned by get_recent_episodes).
            max_episodes: Cap on episodes to replay (default 60 — roughly 3
                          effective EMA windows at alpha=0.05).

        Returns:
            Number of episodes actually replayed.
        """
        if not episodes:
            return 0

        # Play in chronological order (oldest first)
        to_replay = list(reversed(episodes))[:max_episodes]
        replayed = 0

        class _SilentMsg:
            """Dummy message object — we don't want to write surprise metadata."""
            metadata: dict = None

        for episode in to_replay:
            text = getattr(episode, "text", "") or ""
            if not text.strip():
                continue

            embedding = self._emb.embed(text)
            if embedding is None:
                continue

            # Treat each episode as a complete user→assistant exchange:
            # the same text serves as both key (user) and value (assistant).
            # This is a simplification but sufficient for EMA baseline warmup.
            key = self._key_proj(embedding)
            value = self._val_proj(embedding)

            self._neural.step(key, value)
            replayed += 1

        logger.info(
            "NeuralCoordinator: pre-warmed from %d episodes (%d steps total)",
            replayed, self._neural.get_stats().get("total_steps", 0),
        )
        return replayed

    # ------------------------------------------------------------------
    # Surprise accessors (for importance scoring and forgetting policy)
    # ------------------------------------------------------------------

    def get_last_surprise(self) -> Optional[float]:
        """Return the surprise score from the most recent RTRL step.

        This is the per-turn surprise — how unexpected was the last
        (user→assistant) pair relative to what the network predicted.
        Used to set episode importance dynamically.

        Returns None when the network is disabled or has no steps yet.
        """
        if self._neural is None or self._neural._memory is None:
            return None
        mem = self._neural._memory
        if mem._step_count == 0:
            return None
        # _surprise_ema tracks the running EMA; for the last step value
        # we need the step result — stored in the last step's stats update.
        # Use surprise_ema as a proxy for recent surprise (more stable than
        # a single step's raw MSE).
        return float(mem._surprise_ema)

    def get_surprise_ema(self, window: int = 20) -> float:
        """Return the exponential moving average of surprise.

        More useful than avg_surprise for current-context decisions because
        it weights recent turns more heavily.  The EMA alpha is set in
        TITANSConfig (default 0.1, meaning ~10-turn effective window).

        Args:
            window: Ignored — EMA alpha governs the window. Parameter kept
                    for API symmetry with potential future implementations.

        Returns 0.0 when the network is disabled or has no steps.
        """
        if self._neural is None or self._neural._memory is None:
            return 0.0
        return float(self._neural._memory._surprise_ema)

    def get_surprise_for_query(self, query: str) -> Optional[float]:
        """Estimate surprise for an arbitrary query string.

        Embeds the query and asks the RTRL network how unexpected it is
        relative to the current hidden state.  Used by retrieval ranking
        to bias episodic results toward contextually-appropriate surprisingness.

        Returns None when embedding fails or network is disabled.
        """
        embedding = self._emb.embed(query)
        if embedding is None:
            return None
        key = self._key_proj(embedding)
        value = self._val_proj(embedding)
        try:
            return float(self._neural.surprise(key, value))
        except Exception as e:
            logger.debug("get_surprise_for_query failed: %s", e)
            return None

    def is_warmed_up(self) -> bool:
        """Return True once the network has enough steps to produce reliable signals."""
        stats = self._neural.get_stats() if self._neural else {}
        return stats.get("total_steps", 0) >= _MIN_STEPS_FOR_HINT

    # ------------------------------------------------------------------
    # Retrieval support
    # ------------------------------------------------------------------

    def query_neural_context(self, query: str) -> Optional[Dict]:
        """Compute surprise and predicted value vector for a retrieval query.

        Uses read() (non-mutating) to get the network's predicted response
        embedding for the query, plus the surprise score.  Intended to be
        called once per retrieve() call, before candidate scoring, so the
        predicted vector can drive per-candidate affinity scoring.

        Returns dict with keys:
            predicted_value  np.ndarray (value_dim,) — network's prediction
            query_surprise   float  — MSE between prediction and self-assoc target
            avg_surprise     float  — running EMA baseline
            surprise_ratio   float  — query_surprise / avg_surprise, clamped [0, 3]
            total_steps      int
            warmed_up        bool
            write_ratio      float
            memory_role      str
            model_fingerprint str | None

        Returns None if embedding fails.
        """
        embedding = self._emb.embed(query)
        if embedding is None:
            return None

        key = self._key_proj(embedding)
        value = self._val_proj(embedding)  # self-association as surprise target

        predicted_value = self._neural.read(key)   # non-mutating
        surprise = self._neural.surprise(key, value)
        stats = self._neural.get_stats()

        avg_surprise = stats.get("avg_surprise", 0.0)
        total_steps = stats.get("total_steps", 0)
        warmed_up = total_steps >= _MIN_STEPS_FOR_HINT

        surprise_ratio = 0.0
        if warmed_up and avg_surprise > 1e-10:
            surprise_ratio = min(surprise / avg_surprise, 3.0)

        return {
            "predicted_value": predicted_value,
            "query_surprise": float(surprise),
            "avg_surprise": avg_surprise,
            "surprise_ratio": float(surprise_ratio),
            "total_steps": total_steps,
            "warmed_up": warmed_up,
            "write_ratio": stats.get("write_ratio", 0.0),
            "memory_role": stats.get("memory_role", "auxiliary_embedding_memory"),
            "backend_agnostic": stats.get("backend_agnostic", True),
            "model_fingerprint": stats.get("model_fingerprint"),
        }

    def query_surprise(self, query: str) -> Optional[Dict]:
        """Backward-compatible wrapper around query_neural_context.

        Returns the same dict as before but delegates to query_neural_context
        so that read() and surprise() share one embed call.  Callers that
        already have a neural_ctx dict from query_neural_context should use
        that directly instead of calling this method.
        """
        ctx = self.query_neural_context(query)
        if ctx is None:
            return None
        # Strip predicted_value — it is not JSON-serialisable and wasn't
        # part of the original contract.
        return {k: v for k, v in ctx.items() if k != "predicted_value"}

    def candidate_affinity(
        self, predicted_value, candidate_text: str
    ) -> float:
        """Cosine similarity between the network's predicted response and a candidate.

        Measures how closely a retrieved memory matches what the RTRL network
        expects to be relevant for the current query.  Only meaningful when
        the network is warmed up (is_warmed_up() is True).

        Args:
            predicted_value: Value vector from read(), shape (value_dim,).
            candidate_text:  Text of the retrieval candidate to score.

        Returns:
            Cosine similarity in [-1, 1].  Returns 0.0 on embed failure or
            degenerate vectors.
        """
        import numpy as np
        try:
            embedding = self._emb.embed(candidate_text)
            if embedding is None:
                return 0.0
            candidate_value = self._val_proj(embedding)
            pn = np.linalg.norm(predicted_value)
            cn = np.linalg.norm(candidate_value)
            if pn < 1e-10 or cn < 1e-10:
                return 0.0
            return float(np.dot(predicted_value, candidate_value) / (pn * cn))
        except Exception:
            logger.debug("candidate_affinity failed", exc_info=True)
            return 0.0

    # ------------------------------------------------------------------
    # Neural consolidation (affinity-hit tracking → semantic promotion)
    # ------------------------------------------------------------------

    # Minimum cosine similarity for a retrieval event to count as a hit.
    _AFFINITY_HIT_THRESHOLD: float = 0.15

    def record_retrieval_affinities(self, candidates: list) -> None:
        """Accumulate per-episode affinity hits from a retrieval selection.

        Called by UnifiedRetriever after candidate selection.  Only episodic
        candidates with neural_affinity above the threshold are counted —
        this filters out low-signal or unwarmed-network retrievals.

        Args:
            candidates: List of RetrievalCandidate objects that were selected.
        """
        for cand in candidates:
            if cand.layer != "episodic":
                continue
            affinity = float(cand.metadata.get("neural_affinity", 0.0))
            if affinity < self._AFFINITY_HIT_THRESHOLD:
                continue
            ep_id = cand.source_id
            if ep_id:
                self._affinity_hits[ep_id] = self._affinity_hits.get(ep_id, 0) + 1
                logger.debug(
                    "NeuralCoordinator: affinity hit ep=%s affinity=%.3f hits=%d",
                    ep_id, affinity, self._affinity_hits[ep_id],
                )

    def consolidation_candidates(self, min_hits: int = 2) -> List[str]:
        """Return episode IDs that have accumulated enough affinity hits.

        Episodes that have been repeatedly retrieved with high neural affinity
        are stable, well-learned patterns — good candidates for promotion to
        semantic memory.

        The hit counts are cleared after this call so each episode is only
        returned once.  The lifecycle layer's digest-based deduplication
        prevents double-promotion if the same episode is nominated again.

        Args:
            min_hits: Minimum number of high-affinity retrievals required.

        Returns:
            List of episode source_ids meeting the threshold.
        """
        candidates = [
            ep_id for ep_id, hits in self._affinity_hits.items()
            if hits >= min_hits
        ]
        # Clear consumed hits; episodes below threshold remain for accumulation.
        for ep_id in candidates:
            del self._affinity_hits[ep_id]
        return candidates

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset session state (pending key + neural hidden state).

        Preserves learned weights — only the in-context hidden state is cleared.
        Affinity hit counts are also cleared so consolidation starts fresh.
        """
        self._pending_user_key = None
        self._affinity_hits.clear()
        self._neural.reset()

    # ------------------------------------------------------------------
    # Passthrough for stats / close
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        return self._neural.get_stats()

    def close(self) -> None:
        self._neural.close()
