"""
Claude Engine Implementation

For your Claude Pro subscription.
Strategic use for complex reasoning tasks.

Author: Jeffrey Dean
"""

import anthropic
from typing import Optional, Union, AsyncIterator
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

from .base import LLMEngine, CompressionStrategy, LogprobResult, _count_tokens


logger = logging.getLogger(__name__)


class ClaudeEngine(LLMEngine):
    """
    Anthropic Claude API wrapper.
    
    For strategic cloud use on complex tasks.
    Your Pro subscription gives you API access.
    """
    
    def __init__(
        self,
        model_name: str = "claude-sonnet-4.5-20250929",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        compression_strategy: CompressionStrategy = CompressionStrategy.TRUNCATE_END,
        max_retries: int = 1,
        timeout: int = 60,
        max_context: int = 200000
    ):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            compression_strategy=compression_strategy,
            max_retries=max_retries,
            timeout=timeout
        )
        
        self._max_context = max_context
        
        # API key from env if not provided
        if api_key:
            self.client = anthropic.Anthropic(
                api_key=api_key,
                timeout=timeout
            )
            self.async_client = anthropic.AsyncAnthropic(
                api_key=api_key,
                timeout=timeout
            )
        else:
            # Uses ANTHROPIC_API_KEY env var
            self.client = anthropic.Anthropic(timeout=timeout)
            self.async_client = anthropic.AsyncAnthropic(timeout=timeout)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[type[BaseModel]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Union[str, BaseModel]:
        """Generate with Claude"""
        
        sys_prompt = system_prompt or self.system_prompt or "You are Claude, a helpful AI assistant."
        
        # Check if prompt needs compression
        prompt_tokens = self.count_tokens(prompt)
        context_limit = self.max_context_length
        
        if prompt_tokens > context_limit - max_tokens - 1000:
            logger.warning("Prompt (%d tokens) exceeds limit, compressing...", prompt_tokens)
            prompt = self.compress_prompt(
                prompt,
                target_tokens=context_limit - max_tokens - 1000
            )
        
        # Add schema instructions if structured output requested
        if response_format:
            from .utils.structured_output import StructuredOutputHandler
            schema_prompt = StructuredOutputHandler.create_schema_prompt(
                response_format,
                include_examples=True
            )
            prompt = f"{prompt}\n\n{schema_prompt}"
        
        # Generate with retry
        raw_response = self._generate_with_retry(
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
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
    
    def generate_with_logprobs(self, *args, **kwargs) -> LogprobResult:
        """Claude API does not expose logprobs."""
        raise NotImplementedError(
            "Anthropic API does not expose token log probabilities. "
            "Use VLLMEngine or OllamaEngine for logprobs support."
        )
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, anthropic.RateLimitError))
    )
    def _generate_with_retry(
        self,
        system: str,
        messages: list,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Internal method with retry"""
        
        response = self.client.messages.create(
            model=self.model_name,
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.content[0].text
    
    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream Claude response"""
        
        sys_prompt = system_prompt or self.system_prompt or "You are Claude, a helpful AI assistant."
        
        # Check/compress
        prompt_tokens = self.count_tokens(prompt)
        if prompt_tokens > self.max_context_length - max_tokens - 1000:
            prompt = self.compress_prompt(
                prompt,
                target_tokens=self.max_context_length - max_tokens - 1000
            )
        
        # Stream
        async with self.async_client.messages.stream(
            model=self.model_name,
            system=sys_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ) as stream:
            async for text in stream.text_stream:
                yield text
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (cl100k_base) with len//4 fallback."""
        return _count_tokens(text)
    
    @property
    def max_context_length(self) -> int:
        return self._max_context
