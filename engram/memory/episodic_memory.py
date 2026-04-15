"""
Episodic Memory Implementation - FIXED

Author: Jeffrey Dean
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Episode:
    """Single episode in episodic memory."""
    id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    importance: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "text": self.text,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "importance": self.importance,
        }
    
    @classmethod
    def from_chromadb(cls, id: str, document: str, metadata: Dict[str, Any]) -> "Episode":
        return cls(
            id=id,
            timestamp=metadata.get("timestamp", time.time()),
            text=document,
            metadata=json.loads(metadata.get("metadata", "{}")) if isinstance(metadata.get("metadata"), str) else metadata.get("metadata", {}),
            session_id=metadata.get("session_id"),
            project_id=metadata.get("project_id"),
            importance=metadata.get("importance", 0.5),
        )


class LocalEmbeddingFunction:
    """Local embedding function using SentenceTransformers.

    Wraps an optional EmbeddingCache so repeated texts (same query across
    sessions, frequently-retrieved episodes) skip the model entirely.
    """

    # ChromaDB 0.5+ may look for this attribute to detect legacy embedding
    # functions. We provide it to avoid noisy DeprecationWarnings.
    is_legacy = False

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        embedding_cache=None,   # Optional[EmbeddingCache] — avoid circular import
    ):
        self.model_name = model_name
        self._cache = embedding_cache

        # Portability: pick the best available accelerator without requiring it.
        # SentenceTransformers accepts "cpu" / "cuda" (and "mps" via torch).
        from ..utils.device import resolve_device
        resolved = resolve_device(device)
        self.model = SentenceTransformer(model_name, device=resolved)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts, using cache when available."""
        if self._cache is None:
            return self.model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            ).tolist()

        import numpy as np
        results: List[List[float]] = [None] * len(texts)  # type: ignore
        misses: List[int] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results[i] = cached.tolist()
            else:
                misses.append(i)

        if misses:
            miss_texts = [texts[i] for i in misses]
            computed = self.model.encode(
                miss_texts, convert_to_numpy=True, show_progress_bar=False
            )
            for idx, arr in zip(misses, computed):
                vec = np.array(arr, dtype=np.float32)
                self._cache.put(texts[idx], vec)
                results[idx] = vec.tolist()

        return results

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts (for add operations)."""
        return self._encode(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for query texts (for search operations)."""
        return self._encode(input)

    def name(self) -> str:
        """Return embedding function name for ChromaDB compatibility."""
        return f"local_{self.model_name.replace('/', '_')}"


class EpisodicMemory:
    """ChromaDB-backed episodic memory for cross-session retrieval."""
    
    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = "episodes",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "auto",
        embedding_cache=None,   # Optional[EmbeddingCache]
    ):
        """Initialize episodic memory with local embeddings.

        Args:
            persist_dir: Directory for ChromaDB persistence. None = in-memory.
            collection_name: ChromaDB collection name.
            embedding_model: SentenceTransformer model name.
            embedding_device: Device hint ("auto", "cpu", "cuda").
            embedding_cache: Optional EmbeddingCache shared with the rest of
                             the Engram stack. When provided, both add and
                             search calls benefit from the two-tier LRU+disk
                             cache, avoiding redundant model calls.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Initialize ChromaDB client
        if persist_dir:
            persist_dir = Path(persist_dir).expanduser().resolve(strict=False)
            persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )

        # Initialize LOCAL embedding function, wired to the shared cache
        self.embedding_fn = LocalEmbeddingFunction(
            embedding_model,
            device=embedding_device,
            embedding_cache=embedding_cache,
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_episode(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        importance: float = 0.5,
        episode_id: Optional[str] = None,
    ) -> str:
        """Add episode to episodic memory. Returns episode ID."""
        if not text.strip():
            raise ValueError("Episode text cannot be empty")
        
        if episode_id is None:
            episode_id = f"ep_{int(time.time() * 1000000)}"
        
        chroma_metadata = {
            "timestamp": time.time(),
            "session_id": session_id or "unknown",
            "project_id": project_id or "default",
            "importance": importance,
            "metadata": json.dumps(metadata or {}),
        }
        
        self.collection.add(
            documents=[text],
            metadatas=[chroma_metadata],
            ids=[episode_id]
        )
        
        return episode_id
    
    def search(
        self,
        query: str,
        n: int = 5,
        project_id: Optional[str] = None,
        min_importance: float = 0.0,
        days_back: Optional[int] = None,
    ) -> List[Episode]:
        """Semantic search for similar episodes."""
        if not query.strip():
            return []
        
        conditions = []
        
        if project_id:
            conditions.append({"project_id": project_id})
        
        if min_importance > 0.0:
            conditions.append({"importance": {"$gte": min_importance}})
        
        if days_back:
            cutoff_time = time.time() - (days_back * 86400)
            conditions.append({"timestamp": {"$gte": cutoff_time}})
        
        where_clause = None
        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n,
            where=where_clause,
        )
        
        episodes = []
        
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                episode = Episode.from_chromadb(
                    id=doc_id,
                    document=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                )
                episodes.append(episode)
        
        return episodes
    
    def get_recent_episodes(
        self,
        n: int = 10,
        days_back: int = 7,
        project_id: Optional[str] = None,
    ) -> List[Episode]:
        """Get recent episodes by timestamp."""
        cutoff_time = time.time() - (days_back * 86400)
        
        conditions = [{"timestamp": {"$gte": cutoff_time}}]
        if project_id:
            conditions.append({"project_id": project_id})
        
        where_clause = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        
        results = self.collection.get(
            where=where_clause,
            limit=n * 2,
        )
        
        episodes = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                episode = Episode.from_chromadb(
                    id=doc_id,
                    document=results["documents"][i],
                    metadata=results["metadatas"][i],
                )
                episodes.append(episode)
        
        episodes.sort(key=lambda e: e.timestamp, reverse=True)
        
        return episodes[:n]
    
    def get_by_session(self, session_id: str) -> List[Episode]:
        """Retrieve all episodes from a specific session."""
        results = self.collection.get(
            where={"session_id": session_id},
        )
        
        episodes = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                episode = Episode.from_chromadb(
                    id=doc_id,
                    document=results["documents"][i],
                    metadata=results["metadatas"][i],
                )
                episodes.append(episode)
        
        episodes.sort(key=lambda e: e.timestamp)
        
        return episodes
    
    def delete_episode(self, episode_id: str):
        """Delete a specific episode."""
        self.collection.delete(ids=[episode_id])
    
    def delete_old_episodes(self, days: int = 90):
        """Delete episodes older than N days."""
        cutoff_time = time.time() - (days * 86400)
        
        results = self.collection.get(
            where={"timestamp": {"$lt": cutoff_time}},
        )
        
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        
        return 0
    
    def get_metadata_batch(
        self,
        project_id: str,
        limit: int = 500,
        offset: int = 0,
    ) -> List[Dict]:
        """Return a list of {id, timestamp, importance, metadata} dicts.

        Fetches metadata only — no document text or embedding vectors.
        Used by ForgettingPolicy.score_all() to score retention without
        loading full episode content.

        Args:
            project_id: Only return episodes belonging to this project.
            limit:  Max records per call (for pagination).
            offset: Skip this many records (for pagination).

        Returns:
            List of dicts with keys: id, timestamp, importance, metadata.
            Empty list when no more records.
        """
        batch = self.collection.get(
            where={"project_id": project_id},
            limit=limit,
            offset=offset,
            include=["metadatas"],
        )
        ids = batch.get("ids") or []
        metas = batch.get("metadatas") or [{}] * len(ids)
        return [
            {
                "id": eid,
                "timestamp": m.get("timestamp", 0),
                "importance": m.get("importance", 0.5),
                "metadata": m.get("metadata", "{}"),
            }
            for eid, m in zip(ids, metas)
        ]

    def get_by_ids(self, ids: List[str]) -> List[Episode]:
        """Fetch full Episode objects for the given list of ids.

        Used by ForgettingPolicy.run() to retrieve episode text before
        archiving to cold storage.

        Args:
            ids: List of episode ids to fetch.

        Returns:
            List of Episode objects (order matches input ids where possible).
        """
        if not ids:
            return []
        result = self.collection.get(ids=ids)
        episodes = []
        for i, eid in enumerate(result.get("ids") or []):
            meta = (result.get("metadatas") or [{}])[i]
            doc = (result.get("documents") or [""])[i]
            episodes.append(Episode.from_chromadb(eid, doc, meta))
        return episodes

    def delete_episodes(self, ids: List[str]) -> int:
        """Delete multiple episodes by id in a single call.

        Args:
            ids: List of episode ids to delete.

        Returns:
            Number of ids submitted for deletion.
        """
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    def find_similar_episodes(
        self,
        text: str,
        project_id: Optional[str] = None,
        n: int = 5,
        similarity_threshold: float = 0.92,
    ) -> List[tuple]:
        """Find existing episodes with cosine similarity above threshold.

        Returns list of (episode_id, similarity_score) tuples, sorted by
        similarity descending. Used for deduplication before storing a new
        episode.

        Args:
            text: New episode text to compare against.
            project_id: Restrict search to this project.
            n: Max candidates to retrieve before threshold filtering.
            similarity_threshold: Minimum cosine similarity (0-1) to flag
                                  as a near-duplicate. 0.92 is conservative
                                  — catches paraphrases of the same fact
                                  while leaving distinct-but-related facts
                                  alone.
        """
        if not text.strip():
            return []
        try:
            # Embed the new text using the same model as the collection
            new_vec = self.embedding_fn.model.encode(
                [text], normalize_embeddings=True
            )[0]

            # Retrieve top-n candidates via ChromaDB semantic search
            episodes = self.search(query=text, n=n, project_id=project_id)
            if not episodes:
                return []

            # Re-embed candidates and compute exact cosine similarity
            # (ChromaDB distances are approximate; we want exact scores)
            candidate_texts = [ep.text for ep in episodes]
            candidate_vecs = self.embedding_fn.model.encode(
                candidate_texts, normalize_embeddings=True
            )
            similarities = candidate_vecs @ new_vec  # dot product of unit vecs = cosine sim

            results = []
            for ep, sim in zip(episodes, similarities):
                if float(sim) >= similarity_threshold:
                    results.append((ep.id, float(sim)))

            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "find_similar_episodes failed: %s", e
            )
            return []

    def tombstone_episodes(self, ids: List[str]) -> int:
        """Mark episodes as superseded by deleting them.

        Currently a thin wrapper over delete_episodes. In future this will
        set a 'superseded_by' metadata field instead of deleting, supporting
        Option 4 (explicit version tracking) non-destructive semantics.

        Args:
            ids: Episode IDs to tombstone.

        Returns:
            Number of episodes tombstoned.
        """
        if not ids:
            return 0
        import logging
        logging.getLogger(__name__).info(
            "Tombstoning %d superseded episode(s): %s", len(ids), ids
        )
        return self.delete_episodes(ids)

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics.

        Uses ``collection.count()`` for the total, then a bounded metadata
        sample for date-range and project/session diversity — avoids loading
        all documents and embeddings.
        """
        total = self.collection.count()
        if total == 0:
            return {
                "total_episodes": 0,
                "projects": [],
                "sessions": [],
                "oldest_timestamp": None,
                "newest_timestamp": None,
            }

        # Sample up to 2000 records for stats (metadata only, no vectors)
        SAMPLE = min(total, 2000)
        results = self.collection.get(limit=SAMPLE, include=["metadatas"])
        metadatas = results.get("metadatas") or []

        timestamps = [m["timestamp"] for m in metadatas if "timestamp" in m]
        projects = list({m["project_id"] for m in metadatas if m.get("project_id")})
        sessions = list({m["session_id"] for m in metadatas if m.get("session_id")})

        return {
            "total_episodes": total,
            "projects": projects,
            "sessions": len(sessions),
            "oldest_timestamp": min(timestamps) if timestamps else None,
            "newest_timestamp": max(timestamps) if timestamps else None,
            "date_range_days": (
                (max(timestamps) - min(timestamps)) / 86400
                if len(timestamps) > 1 else 0
            ),
        }
    
    def clear_collection(self):
        """Delete all episodes in batches (avoids loading all docs/vectors)."""
        BATCH = 500
        while True:
            results = self.collection.get(limit=BATCH, include=[])
            ids = results.get("ids") or []
            if not ids:
                break
            self.collection.delete(ids=ids)

    def close(self):
        """Release ChromaDB resources."""
        # ChromaDB persistent client doesn't require explicit close,
        # but we clear references for GC.
        self.collection = None
        self.client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
