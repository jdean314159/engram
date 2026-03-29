#!/usr/bin/env python3
"""Template for writing a custom Engram engine adapter.

This file is intentionally simple. Replace the placeholder generation logic with
calls to your own backend.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

from engram.engine.base import LLMEngine, LogprobResult


class SimpleTemplateEngine(LLMEngine):
    """Minimal custom adapter template.

    This class is useful when you want Engram's memory runtime but need to talk
    to a backend that is not already covered by the built-in adapters.
    """

    def __init__(self, model_name: str = "template-backend"):
        super().__init__(model_name=model_name)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format=None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ):
        # Replace this stub with a request to your real backend.
        text = f"[template response from {self.model_name}]\n\n{prompt[:200]}"
        if response_format is not None:
            return response_format.model_validate({"text": text})
        return text

    def generate_with_logprobs(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_logprobs: int = 0,
        **kwargs,
    ) -> LogprobResult:
        # If your backend does not support token logprobs yet, return text only.
        return LogprobResult(text=self.generate(prompt, system_prompt, None, temperature, max_tokens, **kwargs))

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> AsyncIterator[str]:
        yield self.generate(prompt, system_prompt, None, temperature, max_tokens, **kwargs)

    def count_tokens(self, text: str) -> int:
        # Replace with a backend-specific tokenizer if you have one.
        return max(1, len(text) // 4)


if __name__ == "__main__":
    engine = SimpleTemplateEngine()
    print(engine.generate("Hello from a custom Engram adapter."))
