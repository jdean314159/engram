"""
llama.cpp Engine Implementation

Connects to a llama-server instance (llama.cpp HTTP server).
llama-server exposes an OpenAI-compatible /v1/chat/completions endpoint,
making it compatible with Engram's engine infrastructure.

Key differences from vLLM:
  - Model source is a local GGUF file, not a HuggingFace repo
  - Context length comes from the YAML config (not server probe)
  - GPU offload is controlled by --n-gpu-layers N in the launch command
  - No max_model_len endpoint; context clamping uses the configured value
  - Logprob support is build-dependent (detected at runtime)
  - "wrong_model" failure class doesn't apply (server always serves what it loaded)

GPU offload strategy (--n-gpu-layers):
  0              → full CPU execution (no GPU required)
  N (1..layers)  → split: first N transformer layers on GPU, rest on CPU
  999 or -1      → all layers on GPU (equivalent to vLLM for small models)

Split offload is the primary use case over vLLM: a 32B Q4_K_M model (~18GB)
can be partially offloaded with --n-gpu-layers 40 to keep ~11GB on the RTX 3090
and overflow remaining layers to CPU.  Speed is lower than full-GPU but higher
than full-CPU, and larger models become accessible.

Author: Jeffrey Dean
"""

import logging
import re
from typing import Optional, Any

from .base import LLMEngine, CompressionStrategy, _count_tokens
from .openai_compat_mixin import OpenAICompatMixin

try:
    from pydantic import BaseModel as _BaseModel
except ImportError:  # pragma: no cover
    _BaseModel = None  # type: ignore[assignment,misc]

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:  # pragma: no cover
    OpenAI = AsyncOpenAI = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:  # pragma: no cover
    def retry(*args, **kwargs):
        def _decorator(fn): return fn
        return _decorator
    def stop_after_attempt(*args, **kwargs): return None
    def wait_exponential(*args, **kwargs): return None
    def retry_if_exception_type(*args, **kwargs): return None

logger = logging.getLogger(__name__)

# llama-server default context window when not specified
_LLAMA_DEFAULT_CONTEXT = 4096


class LlamaCppEngine(OpenAICompatMixin, LLMEngine):
    """llama-server (llama.cpp) engine with OpenAI-compatible API.

    Supports full-CPU, full-GPU, and split CPU+GPU execution via
    the --n-gpu-layers launch parameter.

    YAML config example::

        engines:
          qwen7b_gguf:
            type: llama_cpp
            model: qwen7b_q4              # served alias (any string)
            gguf_path: /models/qwen2.5-7b-instruct-q4_k_m.gguf
            base_url: http://127.0.0.1:8080/v1
            max_context: 32768
            n_gpu_layers: 0               # 0=CPU-only; 999=all GPU; N=split
            reasoning_visibility: auto
            launch:
              host: 127.0.0.1
              port: 8080
              extra_args:
                - --threads
                - "8"

    Failover profile example::

        profiles:
          default_local:
            engines:
              - qwen32b_awq          # primary: vLLM full-GPU
              - qwen7b_gguf          # fallback: llama.cpp CPU/split-GPU
            allow_cloud_failover: false
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "dummy",
        system_prompt: Optional[str] = None,
        compression_strategy: CompressionStrategy = CompressionStrategy.COMPRESS,
        max_retries: int = 1,
        timeout: int = 300,          # llama.cpp CPU inference is slower
        max_context: int = _LLAMA_DEFAULT_CONTEXT,
        gguf_path: Optional[str] = None,
        n_gpu_layers: int = 0,
        reasoning_visibility: str = "auto",
    ):
        """
        Args:
            model_name:  Served model alias (arbitrary, used for logging).
            base_url:    llama-server base URL (default: http://127.0.0.1:8080/v1).
            api_key:     Ignored by llama-server; kept for OpenAI client compat.
            system_prompt: Default system prompt.
            compression_strategy: How to handle over-budget prompts.
            max_retries: Generation retries on transient errors.
            timeout:     Request timeout in seconds. CPU inference can be slow;
                         300s default is appropriate for large models on CPU.
            max_context: Context window size (from GGUF metadata / launch args).
                         llama-server does not expose this via API so it must be
                         configured explicitly. Default 4096.
            gguf_path:   Absolute path to the GGUF model file (used for launch
                         command generation; not required for runtime operation).
            n_gpu_layers: Number of transformer layers to offload to GPU.
                         0 = full CPU; 999 or -1 = all layers on GPU;
                         N = split (first N layers GPU, rest CPU).
            reasoning_visibility: "auto" | "hide" | "show" — controls whether
                         chain-of-thought markers are stripped from responses.
        """
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            compression_strategy=compression_strategy,
            max_retries=max_retries,
            timeout=timeout,
        )
        self.base_url = base_url
        self._max_context = max_context
        self.gguf_path = gguf_path
        self.n_gpu_layers = n_gpu_layers
        self.reasoning_visibility = reasoning_visibility

        # llama-server does not expose logprob support via /v1/models; we
        # attempt a probe on first use and cache the result.
        self._supports_logprobs: Optional[bool] = None

        # Diagnostics
        self._last_raw_response: Optional[str] = None
        self._last_cleaned_response: Optional[str] = None
        self._last_reasoning_detected: bool = False

        if OpenAI is None:
            self.client = None
            self.async_client = None
            logger.warning("openai package not installed — LlamaCppEngine will not function")
        else:
            self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
            self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    # ------------------------------------------------------------------
    # Logprob detection
    # ------------------------------------------------------------------

    def _probe_logprob_support(self) -> bool:
        """Attempt a minimal generation with logprobs=True to detect support.

        llama-server added logprob support in mid-2024 builds.  Older builds
        return a 400 or silently ignore the parameter.  We probe once and cache.
        """
        if self.client is None:
            return False
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                logprobs=True,
                top_logprobs=1,
            )
            has_logprobs = (
                resp.choices
                and resp.choices[0].logprobs is not None
                and resp.choices[0].logprobs.content is not None
            )
            return bool(has_logprobs)
        except Exception as e:
            logger.debug("Logprob probe failed for llama-server: %s", e)
            return False

    @property
    def supports_logprobs(self) -> bool:
        if self._supports_logprobs is None:
            self._supports_logprobs = self._probe_logprob_support()
            if self._supports_logprobs:
                logger.info("llama-server at %s supports logprobs", self.base_url)
            else:
                logger.info(
                    "llama-server at %s does not support logprobs; "
                    "surprise filter will store conservatively", self.base_url
                )
        return self._supports_logprobs

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[Any] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Generate a response via llama-server.

        Applies the same reasoning-visibility stripping as VLLMEngine so that
        models like Qwen3 that emit <think>...</think> blocks work correctly.
        """
        if self.client is None:
            raise RuntimeError(
                "LlamaCppEngine: openai package not installed. "
                "Run: pip install engram[engines]"
            )

        sys_prompt = system_prompt or self.system_prompt or ""
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        # Clamp max_tokens to configured context window
        effective_max = min(max_tokens, max(1, self._max_context - 64))
        if effective_max < max_tokens:
            logger.debug(
                "Clamping max_tokens %d → %d (llama-server max_context=%d)",
                max_tokens, effective_max, self._max_context,
            )

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=effective_max,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            err = str(e)
            logger.warning("LlamaCppEngine generation error: %s", err)
            raise

        self._last_raw_response = raw
        cleaned = self._apply_reasoning_visibility(raw)
        self._last_cleaned_response = cleaned
        self._last_reasoning_detected = cleaned != raw

        return cleaned

    # ------------------------------------------------------------------
    # Reasoning filter (identical logic to VLLMEngine)
    # ------------------------------------------------------------------

    def _strip_visible_reasoning(self, text: str) -> str:
        """Remove <think>...</think> blocks and planning markers."""
        # Strip XML-style thinking blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Strip planning section markers
        markers = [
            r"^(Thinking Process|Self-Correction(?:/Verification)?|Final Decision):.*?(?=\n\n|\Z)",
            r"^(Planning|Reasoning|Internal monologue):.*?(?=\n\n|\Z)",
        ]
        for pattern in markers:
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE | re.IGNORECASE).strip()
        return text

    def _apply_reasoning_visibility(self, text: str) -> str:
        vis = (self.reasoning_visibility or "auto").lower()
        if vis == "show":
            return text
        has_reasoning = bool(
            re.search(r"<think>", text, re.IGNORECASE)
            or re.search(r"^(Thinking Process|Self-Correction|Final Decision):", text, re.MULTILINE)
        )
        self._last_reasoning_detected = has_reasoning
        if vis in ("auto", "hide") and has_reasoning:
            return self._strip_visible_reasoning(text)
        return text

    # ------------------------------------------------------------------
    # Token counting and context
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return _count_tokens(text)

    @property
    def max_context_length(self) -> int:
        return self._max_context

    @property
    def is_cloud(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        """Return last-generation diagnostics for sandbox display."""
        return {
            "engine": "llama_cpp",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "gguf_path": self.gguf_path or "(not set)",
            "n_gpu_layers": self.n_gpu_layers,
            "max_context": self._max_context,
            "reasoning_visibility": self.reasoning_visibility,
            "last_reasoning_detected": self._last_reasoning_detected,
            "supports_logprobs": self._supports_logprobs,
        }
