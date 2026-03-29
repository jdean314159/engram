"""Embedding service for Engram.

Single responsibility: produce a float32 embedding vector for a text string.

The service is lazy — the underlying SentenceTransformer model is not loaded
until the first call to ``embed()``.  It reuses the model already held by
``EpisodicMemory`` when available, so only one copy of the weights lives in
memory at a time.

All results pass through the shared ``EmbeddingCache`` (two-tier LRU + disk),
so repeated texts across sessions are free after the first computation.

Author: Jeffrey Dean
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Lazy, cache-aware text embedding service.

    Args:
        episodic: Optional EpisodicMemory instance.  When provided and its
                  ``embedding_fn`` is a loaded SentenceTransformer wrapper,
                  the service borrows that model rather than loading its own.
        cache:    EmbeddingCache shared with the rest of the stack.
        device:   Device hint for standalone model load ("auto", "cpu", "cuda").
    """

    def __init__(
        self,
        episodic: Optional[Any] = None,   # EpisodicMemory — avoids circular import
        cache: Optional[Any] = None,      # EmbeddingCache
        device: str = "cpu",
    ) -> None:
        self._episodic = episodic
        self._cache = cache
        self._device = device
        self._model: Optional[Any] = None  # loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> Optional[np.ndarray]:
        """Embed text, returning a 1-D float32 array or None on failure.

        Cache is checked first; on a miss the model is invoked and the
        result is written back to the cache.
        """
        model = self._get_model()
        if model is None:
            return None

        if self._cache is not None:
            def _compute(t: str) -> np.ndarray:
                vecs = model([t])
                return np.array(vecs[0], dtype=np.float32)
            try:
                return self._cache.get_or_compute(text, _compute)
            except Exception as e:
                logger.debug("EmbeddingService.embed cache/compute failed: %s", e)
                return None

        # No cache — call directly
        try:
            vecs = model([text])
            return np.array(vecs[0], dtype=np.float32)
        except Exception as e:
            logger.debug("EmbeddingService.embed failed: %s", e)
            return None

    @property
    def model(self) -> Optional[Any]:
        """The underlying embedding callable, or None if unavailable."""
        return self._get_model()

    @property
    def available(self) -> bool:
        """True if the embedding model can be loaded."""
        return self._get_model() is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_model(self) -> Optional[Any]:
        """Return the embedding callable, loading it lazily if needed."""
        if self._model is not None:
            return self._model

        # Prefer borrowing from episodic memory (avoids a duplicate load)
        if self._episodic is not None:
            ef = getattr(self._episodic, "embedding_fn", None)
            if ef is not None:
                self._model = ef
                return self._model

        # Standalone load
        try:
            from .episodic_memory import LocalEmbeddingFunction
            self._model = LocalEmbeddingFunction(
                device=self._device,
                embedding_cache=self._cache,
            )
            return self._model
        except ImportError:
            logger.debug(
                "sentence-transformers not installed; EmbeddingService unavailable"
            )
            return None
