"""
engram/adapters/llm_adapter.py

EngramLLMAdapter: allows Engram to use any llm_engines ChatModel/EmbeddingModel
for embedding generation and entity extraction.

This replaces Engram's direct Ollama API calls, enabling:
  - Any llm_engines backend (Ollama, Anthropic, OpenAI, mock)
  - Consistent error handling via LLMEngineError hierarchy
  - Easier testing (pass MockEngine)
  - StructuredOutputHandler for robust entity parsing

Usage:
    from llm_engines import EngineFactory
    from engram.adapters.llm_adapter import EngramLLMAdapter

    engine = EngineFactory.create("ollama", model="qwen3:8b")
    adapter = EngramLLMAdapter(engine)

    memory = EngramMemory(project="myapp", llm_adapter=adapter)

ADR: Engram v0.2.0 — LLM Engine Integration
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Lazy imports to avoid hard-coupling at module load time.
# Engram may be used without llm_engines installed.
try:
    from llm_engines.utils.structured_output import StructuredOutputHandler
    _HAS_STRUCTURED_OUTPUT = True
except ImportError:
    _HAS_STRUCTURED_OUTPUT = False

try:
    from contracts import (
        ChatMessage,
        EmbeddingModel,
        EmbeddingRequest,
        GenerationError,
        GenerationRequest,
    )
    _HAS_CONTRACTS = True
except ImportError:
    _HAS_CONTRACTS = False


# ---------------------------------------------------------------------------
# Entity extraction schema
# ---------------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    type: str
    name: str


class ExtractedRelationship(BaseModel):
    subject: str
    predicate: str
    object: str


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


_EXTRACTION_SYSTEM = (
    "You are a precise information extraction assistant. "
    "Extract named entities and their relationships from text. "
    "Return only valid JSON — no prose, no markdown fences."
)

_EXTRACTION_PROMPT_TEMPLATE = """\
Extract all named entities and relationships from the following text.

Text: {text}

Return a JSON object with this exact structure:
{{
  "entities": [
    {{"type": "Person|Organization|Location|Concept|Other", "name": "entity name"}}
  ],
  "relationships": [
    {{"subject": "entity name", "predicate": "RELATIONSHIP_TYPE", "object": "entity name"}}
  ]
}}

Common relationship types: WORKS_ON, WORKS_AT, KNOWS, LOCATED_IN, PART_OF,
CREATED_BY, DEPENDS_ON, COLLABORATES_WITH, MANAGES, REPORTS_TO.

Return only the JSON object."""


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class EngramLLMAdapter:
    """
    Adapter connecting Engram's memory system to an llm_engines ChatModel.

    Args:
        engine:          A ChatModel (and optionally EmbeddingModel) instance.
        embed_model:     Optional separate EmbeddingModel if the main engine
                         does not support embeddings (e.g. AnthropicEngine).
                         If None, uses engine for embeddings if it supports them,
                         otherwise raises.
        extraction_model: Optional separate engine for entity extraction.
                          Useful for keeping a fast/cheap model for extraction
                          and a larger model for conversation.
        max_embed_tokens: Truncate texts longer than this before embedding
                          (avoids context overflow on embedding models).
    """

    def __init__(
        self,
        engine: Any,
        embed_model: Any | None = None,
        extraction_model: Any | None = None,
        max_embed_tokens: int = 512,
    ) -> None:
        if not _HAS_CONTRACTS:
            raise ImportError(
                "EngramLLMAdapter requires llm_engines. "
                "Install with: pip install llm-engines"
            )
        self.engine = engine
        self._embed_model = embed_model
        self._extraction_engine = extraction_model or engine
        self._max_embed_tokens = max_embed_tokens

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate a single embedding vector for text.

        Tries, in order:
          1. Dedicated embed_model if provided
          2. Main engine if it supports embeddings
          3. Raises RuntimeError if neither supports embeddings

        Args:
            text: Text to embed. Will be truncated if too long.

        Returns:
            list[float] embedding vector.
        """
        # Truncate if needed (rough char-based estimate)
        max_chars = self._max_embed_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.debug("Truncated text to %d chars for embedding", max_chars)

        embed_source = self._embed_model or self.engine

        caps = embed_source.get_capabilities()
        if not caps.embeddings:
            raise RuntimeError(
                f"Engine {type(embed_source).__name__} does not support embeddings. "
                "Pass a dedicated embed_model= that implements EmbeddingModel."
            )

        request = EmbeddingRequest(texts=[text])
        response = embed_source.embed(request)
        if not response.vectors:
            raise RuntimeError("Embedding response returned empty vectors")
        return response.vectors[0]

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a single API call.

        More efficient than calling generate_embedding() in a loop.
        """
        max_chars = self._max_embed_tokens * 4
        truncated = [t[:max_chars] if len(t) > max_chars else t for t in texts]

        embed_source = self._embed_model or self.engine
        caps = embed_source.get_capabilities()
        if not caps.embeddings:
            raise RuntimeError(
                f"Engine {type(embed_source).__name__} does not support embeddings."
            )

        request = EmbeddingRequest(texts=truncated)
        response = embed_source.embed(request)
        return response.vectors

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> dict[str, Any]:
        """
        Extract named entities and relationships from text.

        Returns a dict with keys 'entities' and 'relationships', each a list
        of dicts. Falls back to empty result on any parse failure.

        Uses StructuredOutputHandler for robust JSON parsing (handles markdown
        fences, preamble, and minor JSON formatting errors).
        """
        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(text=text)
        request = GenerationRequest(
            messages=[
                ChatMessage(role="system", content=_EXTRACTION_SYSTEM),
                ChatMessage(role="user", content=prompt),
            ],
            max_tokens=1024,
            temperature=0.0,  # deterministic extraction
        )

        try:
            response = self._extraction_engine.generate(request)
        except Exception as e:
            logger.warning("Entity extraction generation failed: %s", e)
            return {"entities": [], "relationships": []}

        content = response.message.content or ""

        if _HAS_STRUCTURED_OUTPUT:
            result = StructuredOutputHandler.parse_with_details(
                content, ExtractionResult, allow_repair=True
            )
            if result.success and result.data:
                return result.data.model_dump()
            logger.debug(
                "Structured extraction failed (%s), trying fallback", result.error
            )

        # Fallback: manual JSON parse
        return self._fallback_parse(content)

    def _fallback_parse(self, content: str) -> dict[str, Any]:
        """Best-effort JSON extraction without StructuredOutputHandler."""
        import re
        # Try to find JSON object
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return {
                    "entities": data.get("entities", []),
                    "relationships": data.get("relationships", []),
                }
            except json.JSONDecodeError:
                pass
        logger.warning("Entity extraction: could not parse JSON from response")
        return {"entities": [], "relationships": []}

    # ------------------------------------------------------------------
    # Simple generation (for completions Engram triggers internally)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        """
        Simple text generation for internal Engram use (e.g. summarisation).

        Returns the response content as a plain string.
        """
        messages = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=prompt))

        request = GenerationRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        try:
            response = self.engine.generate(request)
            return response.message.content or ""
        except Exception as e:
            logger.error("EngramLLMAdapter.generate failed: %s", e)
            raise
