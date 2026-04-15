"""
Engram Memory Eval — Engram Probe (Direct Import Mode)

Wraps ProjectMemory directly. No HTTP layer.

Injection strategy:
  1. add_turn("user", fact_text)   → feeds RTRL pending key
  2. add_turn("assistant", "...")  → steps RTRL, generates surprise score
  3. store_episode(bypass_filter=True) → guaranteed episodic storage,
     importance set from RTRL surprise if neural is warmed up

This mirrors real usage: neural signal is computed from conversational
context, then the episode is durably stored with RTRL-derived importance.

Retrieval:
  search_episodes(query, n=top_k) → list[Episode]

CONFIGURE:
  - ENGRAM_SYS_PATH: path to your Engram repo (added to sys.path)
  - ENGRAM_BASE_DIR: base_dir for ProjectMemory storage
  - ENGINE_PROFILE:  failover engine profile name in llm_engines.yaml,
                     or None to skip neural (disables RTRL signal)
"""

import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────
ENGRAM_SYS_PATH = "/home/cybernaif/ai_tools/engram"      # repo root
ENGRAM_BASE_DIR = "/home/cybernaif/ai_tools/engram/data/memory"
ENGINE_PROFILE  = "default_local"                   # from llm_engines.yaml; None = no engine
EVAL_PROJECT_ID = "eval_harness"
# ─────────────────────────────────────────────────────────────────────────────

if ENGRAM_SYS_PATH not in sys.path:
    sys.path.insert(0, ENGRAM_SYS_PATH)

from engram import ProjectMemory, ProjectType
from engram.rtrl.neural_memory import NeuralMemoryConfig
from engram.engine import create_failover_engine

from fact_generator import Fact


@dataclass
class InjectionResult:
    fact_id: str
    trial: int
    repetition: int
    elapsed_ms: float
    novelty_score: Optional[float]    # raw RTRL surprise (None if neural not warmed)
    novelty_ema: Optional[float]      # EMA baseline at injection time
    novelty_ratio: Optional[float]    # raw / ema (novelty relative to baseline)
    episode_id: Optional[str]         # episodic memory ID, None if not stored
    success: bool
    error: Optional[str] = None


@dataclass
class RetrievalResult:
    fact_id: str
    query: str
    query_type: str           # "direct" | "paraphrase" | "decoy"
    top_k: int
    retrieved_chunks: list    # list[str]
    elapsed_ms: float
    success: bool
    error: Optional[str] = None


class EngramProbe:
    """
    Direct-import wrapper around ProjectMemory for the eval harness.
    Not async — the harness calls are synchronous; asyncio.gather in
    trial_runner.py still works because these are CPU-bound operations
    that release quickly. If Ollama inference becomes a bottleneck,
    wrap with run_in_executor.
    """

    def __init__(self, config):
        self.config = config
        self._memory: Optional[ProjectMemory] = None
        self._session_counter = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        """Initialise ProjectMemory. Called once before any trial."""
        engine = None
        if ENGINE_PROFILE:
            try:
                engine = create_failover_engine(ENGINE_PROFILE)
            except Exception as e:
                print(f"[probe] Warning: engine init failed ({e}); running without neural layer")

        neural_cfg = NeuralMemoryConfig(enabled=True)

        self._memory = ProjectMemory(
            project_id=EVAL_PROJECT_ID,
            project_type=ProjectType.GENERAL_ASSISTANT,
            base_dir=Path(ENGRAM_BASE_DIR),
            llm_engine=engine,
            session_id=f"eval_session_{self._session_counter}",
            neural_config=neural_cfg,
        )
        print(f"[probe] ProjectMemory initialized: {self._memory}")

    async def stop(self):
        """Release resources."""
        if self._memory:
            self._memory.close()
            self._memory = None

    # ── Injection ─────────────────────────────────────────────────────────────

    async def inject_fact(
        self,
        fact: Fact,
        use_contradiction: bool = False,
        trial: int = 0,
        repetition: int = 0,
    ) -> InjectionResult:
        text = fact.contradiction if use_contradiction else fact.canonical
        t0 = time.perf_counter()
        try:
            # Step RTRL: feed as a user->assistant exchange so the neural
            # coordinator sees a complete key->value pair and updates surprise.
            self._memory.add_turn("user", text)
            self._memory.add_turn("assistant", f"Noted: {text}")

            # Read novelty signal immediately after RTRL step.
            novelty_score, novelty_ema, novelty_ratio = self._read_novelty()

            # Guarantee episodic storage regardless of surprise filter.
            episode_id = self._memory.store_episode(
                text,
                metadata={
                    "eval_fact_id": fact.id,
                    "eval_trial": trial,
                    "eval_repetition": repetition,
                    "use_contradiction": use_contradiction,
                },
                importance=0.6,
                bypass_filter=True,
            )

            elapsed = (time.perf_counter() - t0) * 1000
            return InjectionResult(
                fact_id=fact.id,
                trial=trial,
                repetition=repetition,
                elapsed_ms=elapsed,
                novelty_score=novelty_score,
                novelty_ema=novelty_ema,
                novelty_ratio=novelty_ratio,
                episode_id=episode_id,
                success=True,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return InjectionResult(
                fact_id=fact.id,
                trial=trial, repetition=repetition,
                elapsed_ms=elapsed,
                novelty_score=None, novelty_ema=None, novelty_ratio=None,
                episode_id=None,
                success=False,
                error=str(e),
            )

    async def inject_distractor(self, episode_index: int) -> InjectionResult:
        """Inject a semantically unrelated episode (interference)."""
        text = (
            f"Distractor {episode_index}: The quarterly budget review showed "
            f"a 3.2% variance in line item {episode_index % 50 + 1}. "
            f"The audit committee noted this for follow-up."
        )
        t0 = time.perf_counter()
        try:
            self._memory.add_turn("user", text)
            self._memory.add_turn("assistant", "Understood.")
            novelty_score, novelty_ema, novelty_ratio = self._read_novelty()
            episode_id = self._memory.store_episode(
                text,
                metadata={"eval_distractor": True, "distractor_index": episode_index},
                importance=0.3,
                bypass_filter=True,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            return InjectionResult(
                fact_id=f"distractor_{episode_index}",
                trial=-1, repetition=0,
                elapsed_ms=elapsed,
                novelty_score=novelty_score,
                novelty_ema=novelty_ema,
                novelty_ratio=novelty_ratio,
                episode_id=episode_id,
                success=True,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return InjectionResult(
                fact_id=f"distractor_{episode_index}",
                trial=-1, repetition=0,
                elapsed_ms=elapsed,
                novelty_score=None, novelty_ema=None, novelty_ratio=None,
                episode_id=None, success=False, error=str(e),
            )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        fact: Fact,
        query_type: str,    # "direct" | "paraphrase" | "decoy"
        top_k: int = 5,
    ) -> RetrievalResult:
        query = {
            "direct": fact.direct_query,
            "paraphrase": fact.paraphrase_query,
            "decoy": fact.decoy_query,
        }[query_type]

        t0 = time.perf_counter()
        try:
            context = self._memory.get_context(query=query, episodic_n=top_k)
            chunks = [ep.text for ep in context.episodic]
            elapsed = (time.perf_counter() - t0) * 1000
            return RetrievalResult(
                fact_id=fact.id,
                query=query,
                query_type=query_type,
                top_k=top_k,
                retrieved_chunks=chunks,
                elapsed_ms=elapsed,
                success=True,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return RetrievalResult(
                fact_id=fact.id,
                query=query,
                query_type=query_type,
                top_k=top_k,
                retrieved_chunks=[],
                elapsed_ms=elapsed,
                success=False,
                error=str(e),
            )

    # ── Session management ────────────────────────────────────────────────────

    async def reset_working_memory(self):
        """
        Simulate a session boundary: new session ID, working memory cleared,
        RTRL hidden state reset (weights preserved).
        """
        self._session_counter += 1
        new_session_id = f"eval_session_{self._session_counter}"
        self._memory.new_session(new_session_id)
        print(f"[probe] New session: {new_session_id}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _read_novelty(self) -> tuple:
        """
        Read RTRL surprise signal from NeuralCoordinator.
        Returns (raw_surprise, ema, ratio) or (None, None, None) if unavailable.
        """
        nc = getattr(self._memory, "neural_coord", None)
        if nc is None or not nc.is_warmed_up():
            return None, None, None
        try:
            raw = nc.get_last_surprise()
            ema = nc.get_surprise_ema()
            ratio = (raw / ema) if (ema and ema > 1e-8) else None
            return (
                float(raw) if raw is not None else None,
                float(ema) if ema is not None else None,
                float(ratio) if ratio is not None else None,
            )
        except Exception:
            return None, None, None
