"""
Base LLM Engine Interface

Abstract interface that all LLM engines must implement.
Provides unified API for local and cloud models.

Author: Jeffrey Dean
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, AsyncIterator, Dict, Any, List
from pydantic import BaseModel
from enum import Enum
from dataclasses import dataclass, field
import logging
import math

logger = logging.getLogger(__name__)

# --- Shared token counter (tiktoken with fallback) ---
try:
    import tiktoken
    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        """Count tokens using cl100k_base (GPT-4 / Claude-compatible)."""
        return len(_TIKTOKEN_ENCODER.encode(text))

except ImportError:
    logger.warning("tiktoken not installed — using len//4 approximation for token counting")
    _TIKTOKEN_ENCODER = None

    def _count_tokens(text: str) -> int:
        """Fallback: rough estimation at ~4 chars per token."""
        return len(text) // 4


class CompressionStrategy(Enum):
    """How to handle prompts exceeding context window"""
    TRUNCATE_START = "truncate_start"  # Drop oldest content
    TRUNCATE_END = "truncate_end"      # Drop newest content  
    COMPRESS = "compress"               # Summarize/compress content
    ERROR = "error"                     # Raise exception


@dataclass
class TokenLogprob:
    """Log probability for a single token."""
    token: str
    logprob: float
    bytes: Optional[List[int]] = None


@dataclass
class LogprobResult:
    """Logprob data for a full generation."""
    text: str
    token_logprobs: List[TokenLogprob] = field(default_factory=list)

    @property
    def perplexity(self) -> float:
        """Sequence perplexity — primary input to surprise filter."""
        if not self.token_logprobs:
            return 1.0
        avg_nll = -sum(t.logprob for t in self.token_logprobs) / len(self.token_logprobs)
        return math.exp(avg_nll)

    @property
    def mean_logprob(self) -> float:
        """Mean log probability (negative perplexity proxy)."""
        if not self.token_logprobs:
            return 0.0
        return sum(t.logprob for t in self.token_logprobs) / len(self.token_logprobs)
        
    @property
    def token_count(self) -> int:
        """Number of tokens in the generation."""
        return len(self.token_logprobs)


class LLMEngine(ABC):
    """
    Abstract base class for all LLM engines.
    
    Provides:
    - Unified generation interface
    - Automatic prompt compression
    - OOM recovery
    - Streaming support
    - Token counting
    
    Implementations: VLLMEngine, ClaudeEngine, OllamaEngine
    """
    
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        compression_strategy: CompressionStrategy = CompressionStrategy.COMPRESS,
        max_retries: int = 1,
        timeout: int = 120
    ):
        """
        Initialize LLM engine.
        
        Args:
            model_name: Model identifier
            system_prompt: Default system prompt (can be overridden per call)
            compression_strategy: How to handle context overflow
            max_retries: Number of retries on rate limit (your requirement: 1)
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.compression_strategy = compression_strategy
        self.max_retries = max_retries
        self.timeout = timeout
        
        # OOM recovery state
        self.oom_detected = False
        self.adaptive_context_limit = None
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[type[BaseModel]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Union[str, BaseModel]:
        """
        Generate response from LLM.
        
        Auto-compresses if prompt exceeds context window.
        Retries once on rate limit errors.
        Parses to Pydantic model if response_format provided.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override
            response_format: Optional Pydantic model for structured output
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Engine-specific parameters
            
        Returns:
            str if no response_format, BaseModel instance otherwise
        """
        pass
    
    @abstractmethod
    def generate_with_logprobs(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_logprobs: int = 0,
        **kwargs
    ) -> LogprobResult:
        """
        Generate response and return token-level log probabilities.

        Required for TITANS-inspired surprise filter in Library 3.
        top_logprobs: number of alternative tokens per position (0 = selected only).
        Not available on ClaudeEngine (Anthropic API does not expose logprobs).
        """
        pass
    
    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream response tokens in real-time.
        
        Critical for voice interface responsiveness.
        Your requirement: real-time streaming support.
        
        Yields:
            str: Individual tokens or token chunks
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Used for compression decisions.
        """
        pass
    
    def compress_prompt(
        self,
        prompt: str,
        target_tokens: int
    ) -> str:
        """
        Compress prompt to fit within token limit.
        
        Your requirement: auto-compress on context overflow.
        Strategy depends on self.compression_strategy.
        """
        if self.compression_strategy == CompressionStrategy.ERROR:
            raise ValueError(
                f"Prompt ({self.count_tokens(prompt)} tokens) "
                f"exceeds context window ({self.max_context_length} tokens)"
            )
        
        current_tokens = self.count_tokens(prompt)
        if current_tokens <= target_tokens:
            return prompt
        
        ratio = target_tokens / current_tokens
        keep_chars = int(len(prompt) * ratio)
        
        if self.compression_strategy == CompressionStrategy.TRUNCATE_START:
            return prompt[-keep_chars:]
        elif self.compression_strategy == CompressionStrategy.TRUNCATE_END:
            return prompt[:keep_chars]
        elif self.compression_strategy == CompressionStrategy.COMPRESS:
            # Try to compress using the model itself
            try:
                compress_prompt = (
                    f"Compress this text to approximately {target_tokens} tokens "
                    f"while preserving key information:\n\n{prompt}\n\nCompressed version:"
                )
                # Recursive call with compression disabled to avoid infinite loop
                old_strategy = self.compression_strategy
                self.compression_strategy = CompressionStrategy.TRUNCATE_END
                compressed = self.generate(
                    compress_prompt,
                    temperature=0.3,
                    max_tokens=target_tokens
                )
                self.compression_strategy = old_strategy
                return compressed if isinstance(compressed, str) else str(compressed)
            except Exception:
                # Fallback to truncation
                logger.warning("Compression failed, truncating instead")
                return prompt[:keep_chars]
        
        return prompt[:keep_chars]
    
    def recover_from_oom(self) -> bool:
        """
        Attempt to recover from OOM error.
        
        Your requirement: automatic adaptation.
        Reduces adaptive_context_limit by 25% each time.
        
        Returns:
            True if recovery successful, False otherwise
        """
        if self.oom_detected:
            logger.warning("Multiple OOM events detected")
            # Further reduction
            if self.adaptive_context_limit:
                self.adaptive_context_limit = int(self.adaptive_context_limit * 0.75)
                if self.adaptive_context_limit < 1000:
                    logger.error("Context limit too low, cannot recover")
                    return False
                logger.warning("Further reduced context limit to %d tokens", self.adaptive_context_limit)
                return True
            return False
        
        self.oom_detected = True
        
        # First OOM: Reduce to 75% of max
        self.adaptive_context_limit = int(self.max_context_length * 0.75)
        logger.warning("Reduced context limit to %d tokens", self.adaptive_context_limit)
        return True
    
    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """Return maximum context window size"""
        pass
    
    @property
    def current_memory_usage(self) -> Optional[float]:
        """Return current GPU memory usage in GB (if applicable)"""
        return None
