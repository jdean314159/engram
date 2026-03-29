"""OpenAI-compatible cloud engine.

This engine targets OpenAI's Chat Completions API via the official `openai` python
client. It is intentionally minimal and is meant to be used as an *optional*
last-resort fallback after local engines.

NOTE: logprobs are not implemented here (surprise filtering remains local-first).
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Union, AsyncIterator

from pydantic import BaseModel

from .base import LLMEngine, CompressionStrategy, LogprobResult

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI, AsyncOpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None
    AsyncOpenAI = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class OpenAICloudEngine(LLMEngine):
    """Cloud engine using OpenAI API (or compatible base_url)."""

    is_cloud: bool = True
    supports_logprobs: bool = False

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        system_prompt: Optional[str] = None,
        compression_strategy: CompressionStrategy = CompressionStrategy.COMPRESS,
        max_retries: int = 1,
        timeout: int = 120,
        max_context: int = 8192,
    ):
        if OpenAI is None:
            raise ImportError(
                "openai package is required for OpenAICloudEngine. "
                "Install with: pip install engram[engines]"
            ) from _IMPORT_ERROR

        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            compression_strategy=compression_strategy,
            max_retries=max_retries,
            timeout=timeout,
        )
        self._max_context = max_context

        key = api_key or os.getenv(api_key_env)
        if not key:
            raise ValueError(
                f"Missing API key for OpenAICloudEngine. "
                f"Provide api_key or set {api_key_env}."
            )

        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
        self.async_client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[type[BaseModel]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Union[str, BaseModel]:
        # Cloud: be conservative about overflow; compress if needed.
        sys = system_prompt or self.system_prompt
        messages = []
        if sys:
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": prompt})

        # Approx overflow handling (token counter is heuristic unless tiktoken present).
        prompt_tokens = self.count_tokens((sys or "") + "\n" + prompt)
        if prompt_tokens > self.max_context_length - max_tokens - 256:
            target = self.max_context_length - max_tokens - 256
            messages[-1]["content"] = self.compress_prompt(prompt, target_tokens=max(target, 512))

        # Structured output handled upstream by StructuredOutputHandler for local engines;
        # here we keep it simple: ask for JSON when response_format provided.
        if response_format:
            schema_hint = response_format.model_json_schema()
            messages.append({
                "role": "system",
                "content": (
                    "Return ONLY valid JSON matching this schema. "
                    f"Schema: {schema_hint}"
                )
            })

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = resp.choices[0].message.content or ""
        if response_format:
            return response_format.model_validate_json(text)
        return text

    def generate_with_logprobs(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_logprobs: int = 0,
        **kwargs
    ) -> LogprobResult:
        raise NotImplementedError("OpenAICloudEngine does not implement logprobs in this project.")

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[str]:
        sys = system_prompt or self.system_prompt
        messages = []
        if sys:
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": prompt})

        stream = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for event in stream:
            delta = event.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content

    def count_tokens(self, text: str) -> int:
        # Use base implementation token counter in LLMEngine base
        from .base import _count_tokens
        return _count_tokens(text)

    @property
    def max_context_length(self) -> int:
        return self._max_context
