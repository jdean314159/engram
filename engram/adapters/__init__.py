"""
engram/adapters/

Adapters connecting Engram to external systems.

EngramLLMAdapter: use any llm_engines backend for embedding/extraction.
EngramRAGAdapter: expose Engram as a RAGPipeline for llm_inspector comparison.
"""
from engram.adapters.llm_adapter import EngramLLMAdapter

__all__ = ["EngramLLMAdapter"]
