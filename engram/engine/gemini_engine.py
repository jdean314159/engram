"""
Gemini Engine Implementation

Wraps Google Gemini models via the OpenAI-compatible endpoint exposed by
Google AI Studio (generativelanguage.googleapis.com).

This engine is a thin subclass of OpenAICloudEngine — Google's endpoint
accepts the same request/response schema as OpenAI, so no protocol
differences need to be handled here.

Supported models (free tier available at aistudio.google.com):
    gemini-2.0-flash           — fast, capable, recommended default
    gemini-2.0-flash-lite      — lighter variant, higher rate limits
    gemini-1.5-pro             — stronger reasoning, lower free-tier limits
    gemini-1.5-flash           — previous generation flash

Authentication:
    Set GOOGLE_API_KEY environment variable, or pass api_key= directly.
    Free-tier limits (as of 2025): 15 RPM, 1M tokens/day for Flash.

Usage:
    from engram.engine import GeminiEngine

    engine = GeminiEngine()   # uses GOOGLE_API_KEY env var
    response = engine.generate("Plan a 30-minute Spanish lesson.")

Author: Jeffrey Dean
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from .openai_cloud_engine import OpenAICloudEngine
from .base import CompressionStrategy

logger = logging.getLogger(__name__)

# Google AI Studio OpenAI-compatible endpoint
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Context window sizes by model family
_CONTEXT_SIZES: dict[str, int] = {
    "gemini-2.0-flash":       1_048_576,
    "gemini-2.0-flash-lite":  1_048_576,
    "gemini-1.5-pro":         2_097_152,
    "gemini-1.5-flash":       1_048_576,
    "gemini-1.5-flash-8b":    1_048_576,
}
_DEFAULT_CONTEXT = 1_048_576


class GeminiEngine(OpenAICloudEngine):
    backend_label = "gemini"

    """
    Google Gemini via the OpenAI-compatible AI Studio endpoint.

    Inherits all generation, streaming, and context-compression logic from
    OpenAICloudEngine.  The only differences are:
      - base_url points at Google rather than OpenAI
      - api_key is read from GOOGLE_API_KEY by default
      - max_context reflects Gemini's much larger context window

    Args:
        model_name:  Gemini model identifier (default: gemini-2.0-flash).
        api_key:     Google API key.  Falls back to GOOGLE_API_KEY env var.
        system_prompt: Optional default system prompt.
        timeout:     Request timeout in seconds (default 60).
        max_context: Override context window size.  Auto-detected from
                     model_name if not provided.
    """

    is_cloud: bool = True

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        compression_strategy: CompressionStrategy = CompressionStrategy.COMPRESS,
        max_retries: int = 2,
        timeout: int = 60,
        max_context: Optional[int] = None,
    ):
        resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "GeminiEngine requires a Google API key. "
                "Set the GOOGLE_API_KEY environment variable or pass api_key=."
            )

        # Detect context window from model name
        ctx = max_context
        if ctx is None:
            for prefix, size in _CONTEXT_SIZES.items():
                if model_name.startswith(prefix):
                    ctx = size
                    break
            else:
                ctx = _DEFAULT_CONTEXT

        super().__init__(
            model_name=model_name,
            api_key=resolved_key,
            api_key_env="GOOGLE_API_KEY",   # informational — key already resolved
            base_url=_GEMINI_BASE_URL,
            system_prompt=system_prompt,
            compression_strategy=compression_strategy,
            max_retries=max_retries,
            timeout=timeout,
            max_context=ctx,
        )

        logger.info(
            "GeminiEngine initialised: model=%s  context=%d",
            model_name, ctx,
        )

    # generate(), stream(), count_tokens(), max_context_length all
    # inherited from OpenAICloudEngine — no overrides needed.
