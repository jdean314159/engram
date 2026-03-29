"""
OpenAI-Compatible Engine Mixin

Shared implementation for engines that use an OpenAI-compatible API
(vLLM, Ollama). Provides:
  - generate_with_logprobs()  — used by Library 3 surprise filter
  - async stream()            — proper AsyncOpenAI-based implementation

Both VLLMEngine and OllamaEngine inherit from this mixin, so logprobs
logic is not duplicated.

Author: Jeffrey Dean
"""

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:  # pragma: no cover
    OpenAI = AsyncOpenAI = object
from typing import Optional, AsyncIterator, List
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

from .base import LogprobResult, TokenLogprob

logger = logging.getLogger(__name__)


class OpenAICompatMixin:
    """
    Mixin for OpenAI-compatible local engines (vLLM, Ollama).

    Requires the subclass to have:
        self.client       : OpenAI  (sync)
        self.async_client : AsyncOpenAI
        self.model_name   : str
        self.system_prompt: Optional[str]
        self.count_tokens()
        self.compress_prompt()
        self.adaptive_context_limit
        self.max_context_length
    """

    # ------------------------------------------------------------------
    # Logprobs — shared implementation
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    )
    def generate_with_logprobs(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_logprobs: int = 0,
        **kwargs,
    ) -> LogprobResult:
        """
        Generate text and return per-token log probabilities.

        Used by Library 3 surprise filter to compute perplexity.
        top_logprobs=0  → only the chosen token's logprob (cheapest).
        top_logprobs=N  → N most-likely alternatives per position.
        """
        sys_prompt = system_prompt or self.system_prompt
        context_limit = self.adaptive_context_limit or self.max_context_length

        prompt_tokens = self.count_tokens(prompt)
        if prompt_tokens > context_limit - max_tokens - 100:
            prompt = self.compress_prompt(
                prompt, target_tokens=context_limit - max_tokens - 100
            )

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        # OpenAI API: logprobs must be True (boolean), top_logprobs is the count
        # Default to 1 to get at least the selected token's logprob
        actual_top_logprobs = max(1, top_logprobs)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=actual_top_logprobs,
            **kwargs,
        )

        choice = response.choices[0]
        text = choice.message.content or ""

        token_logprobs: List[TokenLogprob] = []
        if choice.logprobs and choice.logprobs.content:
            for tlp in choice.logprobs.content:
                token_logprobs.append(
                    TokenLogprob(
                        token=tlp.token,
                        logprob=tlp.logprob,
                        bytes=getattr(tlp, 'bytes', None),  # Some implementations don't include bytes
                    )
                )
        else:
            logger.warning("No logprobs returned from %s (choice.logprobs=%s)",
                           self.model_name, choice.logprobs)

        return LogprobResult(text=text, token_logprobs=token_logprobs)

    # ------------------------------------------------------------------
    # Async streaming — proper AsyncOpenAI implementation
    # ------------------------------------------------------------------

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Real-time async streaming via AsyncOpenAI.

        Uses self.async_client (AsyncOpenAI) so the event loop is never
        blocked — critical for voice interface responsiveness.
        """
        sys_prompt = system_prompt or self.system_prompt
        context_limit = self.adaptive_context_limit or self.max_context_length

        prompt_tokens = self.count_tokens(prompt)
        if prompt_tokens > context_limit - max_tokens - 100:
            prompt = self.compress_prompt(
                prompt, target_tokens=context_limit - max_tokens - 100
            )

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
