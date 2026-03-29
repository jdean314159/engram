"""Failover LLM engine router.

Provides a single LLMEngine surface that can route requests across multiple
engines in priority order (primary -> local fallback -> optional cloud).

Goals:
  - Local-first reliability under OOM / context overflow / transient errors
  - Deterministic, minimal retries (no infinite loops)
  - Keep portability: no implicit server-side tuning (e.g., GPU layers)

"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Union, AsyncIterator, Dict

from pydantic import BaseModel

from .base import LLMEngine, LogprobResult
from .utils.privacy import sanitize_prompt_for_cloud
from ..telemetry import Telemetry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FailoverPolicy:
    """Controls how the router retries and fails over."""

    # Max total attempts across all engines (including retries).
    max_attempts: int = 4

    # Reduce output tokens on OOM before switching engines.
    reduce_output_on_oom: bool = True
    min_max_tokens: int = 256

    # Compress prompt on context overflow before switching engines.
    compress_on_context_overflow: bool = True

    # If True, allow routing to engines marked is_cloud=True.
    allow_cloud_failover: bool = False

    # One short retry for transient network errors on the same engine.
    transient_retry: bool = True
    transient_retry_backoff_s: float = 0.5

    # Circuit breaker: after N consecutive failures for an engine, skip it for a cooldown.
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_s: float = 30.0

    # Cloud data policy (applies when routing to an engine with is_cloud=True)
    #   - none: disallow cloud usage
    #   - query_only: strip retrieved memory block
    #   - query_plus_summary: replace retrieved memory block with compact summary
    #   - full_context: send full prompt
    cloud_policy: str = "query_plus_summary"


@dataclass
class _EngineHealth:
    failures: int = 0
    cooldown_until: float = 0.0

    def is_healthy(self) -> bool:
        return time.time() >= self.cooldown_until

    def record_failure(self, policy: FailoverPolicy) -> None:
        self.failures += 1
        if self.failures >= policy.circuit_breaker_failures:
            self.cooldown_until = time.time() + policy.circuit_breaker_cooldown_s
            self.failures = 0

    def record_success(self) -> None:
        self.failures = 0
        self.cooldown_until = 0.0


class FailoverEngine(LLMEngine):
    """An LLMEngine wrapper that fails over across multiple engines."""

    def __init__(
        self,
        engines: Sequence[LLMEngine],
        policy: Optional[FailoverPolicy] = None,
        name: str = "failover",
        telemetry: Optional[Telemetry] = None,
    ):
        if not engines:
            raise ValueError("FailoverEngine requires at least one engine")
        # We expose the *primary* model name for logging/compat.
        super().__init__(model_name=getattr(engines[0], "model_name", name))
        self.engines = list(engines)
        self.policy = policy or FailoverPolicy()
        self.name = name
        self._health: Dict[int, _EngineHealth] = {}
        self.telemetry = telemetry or Telemetry(enabled=False)

    # --- helpers ---

    def _is_cloud(self, engine: LLMEngine) -> bool:
        return bool(getattr(engine, "is_cloud", False))

    def _classify_error(self, e: Exception) -> str:
        msg = (str(e) or "").lower()
        # Context / length
        if "context length" in msg or "maximum context" in msg or "too many tokens" in msg:
            return "context"
        # Token budget exceeded (vLLM 400: max_tokens > max_model_len).
        # VLLMEngine already auto-retries this once; if it still surfaces here
        # the engine truly can't serve the request — treat as non-retriable.
        if "max_model_len" in msg and ("max_tokens" in msg or "400" in msg):
            return "token_budget"
        # OOM
        if "out of memory" in msg or "cuda out of memory" in msg or "oom" in msg:
            return "oom"
        # Transient / network
        if "timed out" in msg or "timeout" in msg or "connection" in msg or "unreachable" in msg:
            return "transient"
        return "other"

    def _eligible_engines(self) -> list[LLMEngine]:
        if not self.policy.allow_cloud_failover:
            return [e for e in self.engines if not self._is_cloud(e)]
        if str(self.policy.cloud_policy).lower() == "none":
            return [e for e in self.engines if not self._is_cloud(e)]
        return self.engines

    def _health_for(self, engine: LLMEngine) -> _EngineHealth:
        key = id(engine)
        if key not in self._health:
            self._health[key] = _EngineHealth()
        return self._health[key]

    # --- required interface ---

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[type[BaseModel]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Union[str, BaseModel]:
        attempts = 0
        last_err: Optional[Exception] = None
        cur_prompt = prompt
        cur_max_tokens = max_tokens

        for engine in self._eligible_engines():
            health = self._health_for(engine)
            if not health.is_healthy():
                logger.info(
                    "FailoverEngine(%s): skipping engine=%s (circuit breaker cooldown)",
                    self.name,
                    getattr(engine, "model_name", engine.__class__.__name__),
                )
                self.telemetry.emit(
                    "engine_skip",
                    "engine skipped due to circuit breaker cooldown",
                    engine=getattr(engine, "model_name", engine.__class__.__name__),
                    router=self.name,
                )
                continue
            # For each engine, attempt once, plus an optional transient retry.
            engine_attempts = 0
            while True:
                if attempts >= self.policy.max_attempts:
                    break
                attempts += 1
                engine_attempts += 1

                logger.info(
                    "FailoverEngine(%s): trying engine=%s (attempt %d/%d)",
                    self.name,
                    getattr(engine, "model_name", engine.__class__.__name__),
                    attempts,
                    self.policy.max_attempts,
                )
                self.telemetry.emit(
                    "engine_try",
                    "trying engine",
                    engine=getattr(engine, "model_name", engine.__class__.__name__),
                    attempt=attempts,
                    max_attempts=self.policy.max_attempts,
                    router=self.name,
                )

                try:
                    effective_prompt = cur_prompt
                    if self._is_cloud(engine):
                        effective_prompt = sanitize_prompt_for_cloud(
                            effective_prompt,
                            policy=str(self.policy.cloud_policy),
                        )
                    out = engine.generate(
                        prompt=effective_prompt,
                        system_prompt=system_prompt,
                        response_format=response_format,
                        temperature=temperature,
                        max_tokens=cur_max_tokens,
                        **kwargs,
                    )
                    health.record_success()
                    self.telemetry.emit(
                        "engine_success",
                        "engine succeeded",
                        engine=getattr(engine, "model_name", engine.__class__.__name__),
                        attempt=attempts,
                        router=self.name,
                    )
                    return out
                except Exception as e:
                    last_err = e
                    kind = self._classify_error(e)
                    logger.warning(
                        "FailoverEngine(%s): engine=%s attempt=%d kind=%s error=%s",
                        self.name,
                        getattr(engine, "model_name", engine.__class__.__name__),
                        attempts,
                        kind,
                        str(e)[:300],
                    )
                    try:
                        self.telemetry.emit(
                            "engine_error",
                            "engine error",
                            engine=getattr(engine, "model_name", engine.__class__.__name__),
                            attempt=attempts,
                            error_kind=kind,
                            error=str(e)[:500],
                            router=self.name,
                        )
                    except Exception:
                        pass
                    health.record_failure(self.policy)

                    # Step 1: context overflow -> compress prompt and retry same engine once.
                    if kind == "context" and self.policy.compress_on_context_overflow:
                        try:
                            budget = engine.max_context_length - cur_max_tokens - 256
                            if budget > 256:
                                self.telemetry.emit(
                                    "mitigation",
                                    "context overflow: compress prompt and retry",
                                    engine=getattr(engine, "model_name", engine.__class__.__name__),
                                    target_tokens=budget,
                                    router=self.name,
                                )
                                cur_prompt = engine.compress_prompt(cur_prompt, target_tokens=budget)
                                continue
                        except Exception:
                            # Fall through to failover.
                            pass
                        # Compression failed or budget too small — fall through to failover.

                    # Step 1b: token_budget — server can't serve even with auto-retry.
                    # Break immediately to try the next engine.
                    if kind == "token_budget":
                        logger.warning(
                            "FailoverEngine(%s): engine=%s token budget exhausted; failing over.",
                            self.name,
                            getattr(engine, "model_name", engine.__class__.__name__),
                        )
                        self.telemetry.emit(
                            "engine_switch",
                            "token budget exceeded: failing over",
                            engine=getattr(engine, "model_name", engine.__class__.__name__),
                            router=self.name,
                        )
                        break

                    # Step 2: OOM -> reduce output tokens and retry same engine once.
                    if kind == "oom" and self.policy.reduce_output_on_oom:
                        new_max = max(self.policy.min_max_tokens, int(cur_max_tokens * 0.5))
                        if new_max < cur_max_tokens:
                            logger.info("FailoverEngine(%s): OOM -> reduce max_tokens from %d to %d and retry same engine", self.name, cur_max_tokens, new_max)
                            self.telemetry.emit(
                                "mitigation",
                                "oom: reduce max_tokens and retry",
                                engine=getattr(engine, "model_name", engine.__class__.__name__),
                                from_max_tokens=cur_max_tokens,
                                to_max_tokens=new_max,
                                router=self.name,
                            )
                            cur_max_tokens = new_max
                            continue

                    # Step 3: transient retry once on same engine.
                    if kind == "transient" and self.policy.transient_retry and engine_attempts == 1:
                        logger.info("FailoverEngine(%s): transient error -> retry after %.2fs", self.name, self.policy.transient_retry_backoff_s)
                        self.telemetry.emit(
                            "mitigation",
                            "transient: retry after backoff",
                            engine=getattr(engine, "model_name", engine.__class__.__name__),
                            backoff_s=self.policy.transient_retry_backoff_s,
                            router=self.name,
                        )
                        time.sleep(self.policy.transient_retry_backoff_s)
                        continue

                    logger.info("FailoverEngine(%s): switching to next engine after kind=%s", self.name, kind)
                    self.telemetry.emit(
                        "engine_switch",
                        "switching to next engine",
                        engine=getattr(engine, "model_name", engine.__class__.__name__),
                        switch_kind=kind,
                        router=self.name,
                    )
                    # Otherwise: break to next engine in priority.
                    break


            if attempts >= self.policy.max_attempts:
                break

        # Exhausted
        raise RuntimeError(
            f"FailoverEngine({self.name}) exhausted attempts ({attempts}); last error: {last_err}"
        ) from last_err

    def generate_with_logprobs(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_logprobs: int = 0,
        **kwargs
    ) -> LogprobResult:
        # Only route to engines that support logprobs.
        eligible = []
        for e in self._eligible_engines():
            if getattr(e, "supports_logprobs", True) and hasattr(e, "generate_with_logprobs"):
                eligible.append(e)
        if not eligible:
            raise RuntimeError("No configured engines support logprobs")

        attempts = 0
        last_err: Optional[Exception] = None
        cur_prompt = prompt
        cur_max_tokens = max_tokens

        for engine in eligible:
            health = self._health_for(engine)
            if not health.is_healthy():
                logger.info(
                    "FailoverEngine(%s): skipping engine=%s for logprobs (circuit breaker cooldown)",
                    self.name,
                    getattr(engine, "model_name", engine.__class__.__name__),
                )
                continue
            engine_attempts = 0
            while True:
                if attempts >= self.policy.max_attempts:
                    break
                attempts += 1
                engine_attempts += 1
                logger.info("FailoverEngine(%s): trying engine=%s for logprobs (attempt %d/%d)", self.name, getattr(engine, "model_name", engine.__class__.__name__), attempts, self.policy.max_attempts)

                try:
                    effective_prompt = cur_prompt
                    if self._is_cloud(engine):
                        effective_prompt = sanitize_prompt_for_cloud(
                            effective_prompt,
                            policy=str(self.policy.cloud_policy),
                        )
                    out = engine.generate_with_logprobs(
                        prompt=effective_prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=cur_max_tokens,
                        top_logprobs=top_logprobs,
                        **kwargs,
                    )
                    health.record_success()
                    return out
                except Exception as e:
                    last_err = e
                    kind = self._classify_error(e)
                    health.record_failure(self.policy)

                    # context overflow -> compress and retry
                    if kind == "context" and self.policy.compress_on_context_overflow:
                        try:
                            budget = engine.max_context_length - cur_max_tokens - 256
                            if budget > 256:
                                cur_prompt = engine.compress_prompt(cur_prompt, target_tokens=budget)
                                continue
                        except Exception:
                            pass

                    # oom -> reduce output
                    if kind == "oom" and self.policy.reduce_output_on_oom:
                        new_max = max(self.policy.min_max_tokens, int(cur_max_tokens * 0.5))
                        if new_max < cur_max_tokens:
                            logger.info("FailoverEngine(%s): OOM -> reduce max_tokens from %d to %d and retry same engine", self.name, cur_max_tokens, new_max)
                            cur_max_tokens = new_max
                            continue

                    if kind == "transient" and self.policy.transient_retry and engine_attempts == 1:
                        logger.info("FailoverEngine(%s): transient error -> retry after %.2fs", self.name, self.policy.transient_retry_backoff_s)
                        time.sleep(self.policy.transient_retry_backoff_s)
                        continue

                    break

            if attempts >= self.policy.max_attempts:
                break

        raise RuntimeError(
            f"FailoverEngine({self.name}) logprobs exhausted attempts ({attempts}); last error: {last_err}"
        ) from last_err

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[str]:
        # Streaming failover is tricky; we do a single-engine stream.
        # Use primary eligible engine; if it fails immediately, fall back.
        last_err: Optional[Exception] = None
        for engine in self._eligible_engines():
            try:
                async for chunk in engine.stream(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ):
                    yield chunk
                return
            except Exception as e:
                last_err = e
                logger.warning(
                    "FailoverEngine(%s) stream failed on %s: %s",
                    self.name, getattr(engine, "model_name", engine.__class__.__name__), str(e)[:200]
                )
                continue
        raise RuntimeError(f"FailoverEngine({self.name}) stream failed; last error: {last_err}") from last_err

    def count_tokens(self, text: str) -> int:
        # Use primary engine's counter
        return self.engines[0].count_tokens(text)

    @property
    def max_context_length(self) -> int:
        # Conservative: primary engine's context length
        return self.engines[0].max_context_length
