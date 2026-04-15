"""Embedding cache for Engram.

Two-tier caching to avoid redundant embedding computations:

  1. In-memory LRU dict (fast, bounded, lost on restart)
  2. Optional diskcache persistence (slower, unbounded, survives restarts)

For local sentence-transformers: tier 1 alone saves ~2ms per cache hit.
For cloud embedding APIs (OpenAI, Cohere): tier 2 saves real money.

Usage:
    cache = EmbeddingCache(cache_dir=project_dir / "embedding_cache")
    vec = cache.get_or_compute("How do I use asyncio?", embedder_fn)

Author: Jeffrey Dean
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import diskcache (optional dependency)
try:
    from diskcache import Cache as DiskCache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False


class EmbeddingCache:
    """Two-tier embedding cache: in-memory LRU + optional disk persistence.

    Thread-safe. The in-memory tier uses an OrderedDict with a max size.
    The disk tier (if diskcache is installed) provides persistence across
    process restarts.

    Args:
        cache_dir: Directory for disk cache. None = in-memory only.
        max_memory: Max entries in the in-memory LRU tier.
        disk_expire_s: Disk cache expiry in seconds (default: 30 days).
        enabled: Master switch. False = passthrough, no caching.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory: int = 2048,
        disk_expire_s: int = 86400 * 30,  # 30 days
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._max_memory = max_memory
        self._disk_expire_s = disk_expire_s
        self._lock = threading.Lock()

        # Tier 1: in-memory LRU
        self._memory: OrderedDict[str, np.ndarray] = OrderedDict()

        # Tier 2: disk cache (optional)
        self._disk: Optional[Any] = None
        if cache_dir is not None and HAS_DISKCACHE and enabled:
            try:
                disk_path = Path(cache_dir).expanduser().resolve(strict=False)
                disk_path.mkdir(parents=True, exist_ok=True)
                self._disk = DiskCache(str(disk_path))
                logger.info("Disk embedding cache at %s", disk_path)
            except Exception as e:
                logger.warning("Failed to init disk cache: %s", e)

        # Stats
        self._hits_memory = 0
        self._hits_disk = 0
        self._misses = 0

    @staticmethod
    def _key(text: str) -> str:
        """Compute cache key from text."""
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Look up a cached embedding. Returns None on miss."""
        if not self.enabled:
            return None

        key = self._key(text)

        with self._lock:
            # Tier 1: memory
            if key in self._memory:
                self._memory.move_to_end(key)
                self._hits_memory += 1
                return self._memory[key]

        # Tier 2: disk (outside lock — disk I/O can be slow)
        if self._disk is not None:
            try:
                val = self._disk.get(key)
                if val is not None:
                    vec = np.array(val, dtype=np.float32)
                    # Promote to memory tier
                    with self._lock:
                        self._memory[key] = vec
                        self._evict_memory()
                        self._hits_disk += 1
                    return vec
            except Exception:
                pass

        with self._lock:
            self._misses += 1
        return None

    def put(self, text: str, embedding: np.ndarray):
        """Store an embedding in both tiers."""
        if not self.enabled:
            return

        key = self._key(text)

        with self._lock:
            self._memory[key] = embedding
            self._evict_memory()

        # Disk tier (outside lock)
        if self._disk is not None:
            try:
                self._disk.set(key, embedding.tolist(), expire=self._disk_expire_s)
            except Exception as e:
                logger.debug("Disk cache write failed: %s", e)

    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[str], np.ndarray],
    ) -> np.ndarray:
        """Get from cache or compute and store.

        Args:
            text: Text to embed.
            compute_fn: Function that takes text and returns np.ndarray.
        """
        cached = self.get(text)
        if cached is not None:
            return cached

        embedding = compute_fn(text)
        self.put(text, embedding)
        return embedding

    def get_or_compute_batch(
        self,
        texts: List[str],
        compute_fn: Callable[[List[str]], List[np.ndarray]],
    ) -> List[np.ndarray]:
        """Batch version: cache hits skip computation, misses are batched.

        Args:
            texts: List of texts to embed.
            compute_fn: Function that takes a list of texts and returns
                        a list of np.ndarrays.
        """
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        to_compute: List[int] = []  # Indices of texts needing computation

        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached is not None:
                results[i] = cached
            else:
                to_compute.append(i)

        if to_compute:
            batch_texts = [texts[i] for i in to_compute]
            computed = compute_fn(batch_texts)
            for idx, vec in zip(to_compute, computed):
                arr = np.array(vec, dtype=np.float32)
                results[idx] = arr
                self.put(texts[idx], arr)

        return results  # type: ignore

    def _evict_memory(self):
        """Evict oldest entries from in-memory cache. Must hold _lock."""
        while len(self._memory) > self._max_memory:
            self._memory.popitem(last=False)

    def get_stats(self) -> Dict[str, Any]:
        total = self._hits_memory + self._hits_disk + self._misses
        return {
            "enabled": self.enabled,
            "has_disk": self._disk is not None,
            "memory_size": len(self._memory),
            "max_memory": self._max_memory,
            "hits_memory": self._hits_memory,
            "hits_disk": self._hits_disk,
            "misses": self._misses,
            "hit_rate": (self._hits_memory + self._hits_disk) / total if total > 0 else 0.0,
        }

    def clear(self):
        """Clear both tiers."""
        with self._lock:
            self._memory.clear()
        if self._disk is not None:
            try:
                self._disk.clear()
            except Exception:
                pass

    def close(self):
        """Close disk cache."""
        if self._disk is not None:
            try:
                self._disk.close()
            except Exception:
                pass
