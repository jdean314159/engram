"""
vLLM Engine Implementation

Connects to local vLLM server (OpenAI-compatible API).
Supports your existing Qwen-32B-AWQ setup.

Author: Jeffrey Dean
"""

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:  # pragma: no cover
    OpenAI = AsyncOpenAI = None
from typing import Optional, Union
from pydantic import BaseModel
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:  # pragma: no cover
    def retry(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator
    def stop_after_attempt(*args, **kwargs):
        return None
    def wait_exponential(*args, **kwargs):
        return None
    def retry_if_exception_type(*args, **kwargs):
        return None
import logging

from .base import LLMEngine, CompressionStrategy, _count_tokens
from .openai_compat_mixin import OpenAICompatMixin

logger = logging.getLogger(__name__)


class VLLMEngine(OpenAICompatMixin, LLMEngine):
    """
    Local vLLM server connection with OOM recovery.
    
    Connects to your existing vLLM server.
    Provides logprobs + async streaming via OpenAICompatMixin.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        system_prompt: Optional[str] = None,
        compression_strategy: CompressionStrategy = CompressionStrategy.COMPRESS,
        max_retries: int = 1,
        timeout: int = 120,
        max_context: int = 8192,
        configured_model_name: Optional[str] = None,
        discovery_resolution: Optional[dict] = None,
        reasoning_visibility: str = "auto",
    ):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            compression_strategy=compression_strategy,
            max_retries=max_retries,
            timeout=timeout
        )

        self.base_url = base_url
        self._max_context = max_context
        self.configured_model_name = configured_model_name or model_name
        self.discovery_resolution = discovery_resolution or {
            "requested_model": self.configured_model_name,
            "resolved_model": model_name,
            "source": "configured",
            "match_strategy": "n/a",
            "discovered_ids": [],
        }
        self.reasoning_visibility = reasoning_visibility
        self._last_raw_response = None
        self._last_cleaned_response = None
        self._last_reasoning_detected = False
        # Cached server-side max_model_len (populated on first 400 or explicit probe)
        self._server_max_tokens: Optional[int] = None

        if OpenAI is None or AsyncOpenAI is None:
            self.client = None
            self.async_client = None
        else:
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout
            )
            self.async_client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout
            )

    def _probe_server_max_tokens(self) -> Optional[int]:
        """Query the vLLM server for its actual max_model_len.

        vLLM exposes this via GET /v1/models as max_model_len in the model
        object. Returns None if unavailable or if client is not initialized.
        Result is cached in self._server_max_tokens.
        """
        if self._server_max_tokens is not None:
            return self._server_max_tokens
        if self.client is None:
            return None
        try:
            models = self.client.models.list()
            for model in (models.data or []):
                # vLLM includes max_model_len in the model's extra fields
                raw = getattr(model, "model_extra", None) or {}
                mml = raw.get("max_model_len")
                if mml is None:
                    # Some versions put it directly on the object
                    mml = getattr(model, "max_model_len", None)
                if mml is not None:
                    self._server_max_tokens = int(mml)
                    logger.info(
                        "vLLM server max_model_len=%d (from /v1/models)",
                        self._server_max_tokens,
                    )
                    return self._server_max_tokens
        except Exception as e:
            logger.debug("Could not probe server max_model_len: %s", e)
        return None

    @staticmethod
    def _parse_max_model_len_from_error(error_text: str) -> Optional[int]:
        """Extract max_model_len from a vLLM 400 error message.

        Parses: 'max_tokens=N cannot be greater than max_model_len=M'
        Returns M, or None if not found.
        """
        import re
        m = re.search(r"max_model_len\s*=\s*(\d+)", error_text, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    def _effective_max_tokens(self, requested: int) -> int:
        """Clamp requested output tokens against the known server limit.

        Tries cached _server_max_tokens first; if not available attempts a
        lightweight probe. Falls back gracefully if probe fails.
        """
        server_limit = self._server_max_tokens
        if server_limit is None:
            server_limit = self._probe_server_max_tokens()

        if server_limit is not None and requested > server_limit:
            clamped = max(1, server_limit - 64)  # leave headroom for prompt
            logger.warning(
                "Clamping max_tokens %d → %d (server max_model_len=%d)",
                requested, clamped, server_limit,
            )
            return clamped
        return requested

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            response_format: Optional[type[BaseModel]] = None,
            temperature: float = 0.7,
            max_tokens: int = 2048,
            **kwargs
    ) -> Union[str, BaseModel]:
        """Generate with auto-compression and structured output"""
        

        # Use configured system prompt if not overridden
        sys_prompt = system_prompt or self.system_prompt

        # Clamp output tokens to the server's actual limit before anything else
        max_tokens = self._effective_max_tokens(max_tokens)

        # Check if prompt needs compression
        prompt_tokens = self.count_tokens(prompt)
        context_limit = self.adaptive_context_limit or self.max_context_length
        effective_max_tokens = min(max_tokens, max(1, context_limit - prompt_tokens - 100))

        if prompt_tokens > context_limit - max_tokens - 100:  # Safety margin
            logger.warning("Prompt (%d tokens) exceeds limit, compressing...", prompt_tokens)
            prompt = self.compress_prompt(
                prompt,
                target_tokens=context_limit - max_tokens - 100
            )

        # Add schema instructions if structured output requested
        if response_format:
            from .utils.structured_output import StructuredOutputHandler
            schema_prompt = StructuredOutputHandler.create_schema_prompt(
                response_format,
                include_examples=True
            )
            prompt = f"{prompt}\n\n{schema_prompt}"

        # Build messages
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        # Generate with retry logic
        try:
            raw_response = self._generate_with_retry(
                messages=messages,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                **kwargs
            )

            # Parse structured output if requested
            if response_format:
                from .utils.structured_output import StructuredOutputHandler
                return StructuredOutputHandler.parse(
                    raw_response,
                    response_format
                )

            return raw_response

        except Exception as e:
            error_str = str(e).lower()
            if "cuda out of memory" in error_str or "oom" in error_str:
                if kwargs.pop("_oom_retried", False):
                    raise RuntimeError(f"OOM persists after recovery: {e}")
                logger.error("OOM detected! Attempting recovery...")
                if self.recover_from_oom():
                    logger.info("OOM recovery successful, retrying...")
                    return self.generate(
                        prompt, sys_prompt, response_format,
                        temperature, max_tokens, _oom_retried=True, **kwargs
                    )
                else:
                    raise RuntimeError(f"OOM recovery failed: {e}")
            raise

   
    def _strip_visible_reasoning(self, text: str) -> str:
        stripped = (text or "").strip()
        if not stripped:
            return stripped

        if "</think>" in stripped:
            after = stripped.split("</think>", 1)[1].strip()
            return after or self._FILTERED_MESSAGE

        terminal_markers = [
            "Final answer:",
            "Answer:",
            "Response:",
            "Result:",
            "Output:",
        ]

        lower = stripped.lower()

        # Prefer explicit final-answer markers before any broader reasoning heuristics.
        for marker in terminal_markers:
            idx = lower.rfind(marker.lower())
            if idx != -1:
                candidate = stripped[idx + len(marker):].strip()
                if candidate:
                    return candidate

        looks_like_reasoning = (
            "thinking process:" in lower
            or "let me think" in lower
            or lower.startswith("think:")
            or lower.startswith("reasoning:")
            or lower.startswith("analysis:")
        )

        if looks_like_reasoning:
            paragraphs = [p.strip() for p in stripped.split("\n\n") if p.strip()]
            reasoning_starts = (
                "thinking",
                "reasoning",
                "analysis",
                "scratchpad",
                "thought process",
            )

            for para in reversed(paragraphs):
                para_lower = para.lower()

                for marker in terminal_markers:
                    if para_lower.startswith(marker.lower()):
                        candidate = para[len(marker):].strip()
                        if candidate:
                            return candidate

                if not para_lower.startswith(reasoning_starts):
                    return para

            return self._FILTERED_MESSAGE

        return stripped

    def _apply_reasoning_visibility(self, text: str) -> str:
        mode = getattr(self, "reasoning_visibility", "auto")

        self._last_raw_response = text
        cleaned = self._strip_visible_reasoning(text)
        self._last_cleaned_response = cleaned
        self._last_reasoning_detected = cleaned != text

        if mode == "show":
            return text
        if mode in ("hide", "auto"):
            return cleaned
        return cleaned


    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError))
    )
    def _generate_with_retry(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Internal method with retry decorator"""
        
        if self.client is None:
            raise RuntimeError("OpenAI client is not available. Install the openai package to use VLLMEngine.")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as api_exc:
            err_str = str(api_exc)
            # vLLM returns 400 BadRequestError when max_tokens > max_model_len.
            # Parse the limit, cache it, and retry once with a clamped value.
            if ("400" in err_str or "bad request" in err_str.lower()) and \
               "max_model_len" in err_str.lower() and \
               not kwargs.get("_token_clamped"):
                parsed_limit = self._parse_max_model_len_from_error(err_str)
                if parsed_limit:
                    self._server_max_tokens = parsed_limit
                    clamped = max(1, parsed_limit - 64)
                    logger.warning(
                        "400: max_tokens=%d exceeds server max_model_len=%d; "
                        "retrying with max_tokens=%d",
                        max_tokens, parsed_limit, clamped,
                    )
                    return self._generate_with_retry(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=clamped,
                        _token_clamped=True,
                        **kwargs,
                    )
            raise

        if not getattr(response, "choices", None):
            raise RuntimeError(f"vLLM returned no choices: {response!r}")

        message = getattr(response.choices[0], "message", None)
        if message is None:
            raise RuntimeError(f"vLLM response missing message: {response!r}")

        content = getattr(message, "content", None)

        if isinstance(content, str):
            text = content.strip()
            if text:
                return self._apply_reasoning_visibility(text)

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text"):
                        parts.append(str(item["text"]))
                else:
                    item_text = getattr(item, "text", None)
                    item_type = getattr(item, "type", None)
                    if item_type == "text" and item_text:
                        parts.append(str(item_text))
            text = "\n".join(p.strip() for p in parts if p and str(p).strip())
            if text:
                return self._apply_reasoning_visibility(text)

        text = getattr(response.choices[0], "text", None)
        if isinstance(text, str) and text.strip():
            return self._apply_reasoning_visibility(text.strip())

        refusal = getattr(message, "refusal", None)
        if refusal:
            raise RuntimeError(f"Model returned refusal/no content: {refusal!r}")

        finish_reason = getattr(response.choices[0], "finish_reason", None)
        raise RuntimeError(
            f"vLLM returned empty content. finish_reason={finish_reason!r}, response={response!r}"
        )

    # stream() and generate_with_logprobs() come from OpenAICompatMixin

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (cl100k_base) with len//4 fallback."""
        return _count_tokens(text)

    @property
    def max_context_length(self) -> int:
        """Return configured max context"""
        return self._max_context
