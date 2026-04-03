"""Public engine interfaces for Engram.

This package provides reusable adapters for local and cloud LLM backends.
Applications should prefer these public entry points rather than importing
implementation modules directly.
"""

__all__ = [
    "LLMEngine",
    "CompressionStrategy",
    "LogprobResult",
    "TokenLogprob",
    "create_engine",
    "create_failover_engine",
    "load_config",
    "VLLMEngine",
    "OllamaEngine",
    "ClaudeEngine",
    "GeminiEngine",
    "OpenAICloudEngine",
    "StructuredOutputHandler",
    "FailoverEngine",
    "FailoverPolicy",
]


def __getattr__(name):
    if name in ("LLMEngine", "CompressionStrategy", "LogprobResult", "TokenLogprob", "_count_tokens"):
        from . import base
        return getattr(base, name)
    if name in ("create_engine", "load_config", "create_failover_engine"):
        from . import config_loader
        return getattr(config_loader, name)
    if name == "VLLMEngine":
        from .vllm_engine import VLLMEngine
        return VLLMEngine
    if name == "OllamaEngine":
        from .ollama_engine import OllamaEngine
        return OllamaEngine
    if name == "ClaudeEngine":
        from .claude_engine import ClaudeEngine
        return ClaudeEngine
    if name == "GeminiEngine":
        from .gemini_engine import GeminiEngine
        return GeminiEngine
    if name == "OpenAICloudEngine":
        from .openai_cloud_engine import OpenAICloudEngine
        return OpenAICloudEngine
    if name == "StructuredOutputHandler":
        from .utils.structured_output import StructuredOutputHandler
        return StructuredOutputHandler
    if name in ("FailoverEngine", "FailoverPolicy"):
        from .router import FailoverEngine, FailoverPolicy
        return {"FailoverEngine": FailoverEngine, "FailoverPolicy": FailoverPolicy}[name]
    raise AttributeError(f"module 'engram.engine' has no attribute {name!r}")
