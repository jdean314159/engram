from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python 3.7 compat (unlikely but safe)
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

logger = logging.getLogger(__name__)


@runtime_checkable
class SemanticLayerProtocol(Protocol):
    """Minimum interface the retrieval layer requires from a semantic backend.

    Any class that implements ``search_generic_memories`` satisfies this
    protocol (structural subtyping — no inheritance required).  Custom
    semantic backends only need to implement this one method; the retrieval
    layer will not probe for ``generic_memory_rows`` or ``list_*`` methods.

    Return value: a list of dicts, each with at minimum:
        ``type``        str  — row category ("fact", "preference", "event", …)
        ``text`` / ``content`` / ``value``  str  — the searchable text
        ``timestamp``   float (seconds since epoch)
        ``match_score`` float  0-1 relevance to the query
    """

    def search_generic_memories(
        self,
        query: str,
        *,
        limit: int = 20,
        per_type_limit: int = 60,
        include_graph: bool = True,
        graph_sentence_limit: int = 6,
    ) -> List[Dict[str, Any]]: ...


@dataclass
class RetrievalCandidate:
    layer: str
    text: str
    payload: Any
    token_count: int
    score: float
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_text(self) -> str:
        return re.sub(r"\s+", " ", self.text.strip().lower())


@dataclass
class RetrievalPolicy:
    """Simple policy surface for retrieval quality tuning."""

    episodic_weight_lexical: float = 0.50
    episodic_weight_importance: float = 0.28
    episodic_weight_recency: float = 0.16
    episodic_weight_density: float = 0.06

    semantic_weight_lexical: float = 0.52
    semantic_weight_quality: float = 0.28
    semantic_weight_recency: float = 0.12
    semantic_weight_density: float = 0.08

    cold_weight_lexical: float = 0.42
    cold_weight_recency: float = 0.10

    min_semantic_overlap: float = 0.08
    near_duplicate_threshold: float = 0.84
    diversity_penalty: float = 0.08
    max_similarity_penalty: float = 0.18
    semantic_prefetch_limit: int = 60
    include_graph_context: bool = True
    graph_sentence_limit: int = 6
    neural_affinity_weight: float = 0.08  # Candidate boost from RTRL predicted value


class UnifiedRetriever:
    """Cross-layer retrieval, deduplication, reranking, and light conflict handling.

    Accepts a ``MemoryContext`` (or any object with the same attribute
    surface) rather than a full ``ProjectMemory`` reference.
    """

    synonym_map = {
        "bug": {"issue", "error", "failure", "problem"},
        "issue": {"bug", "error", "problem"},
        "fix": {"patched", "resolved", "solution", "repair"},
        "python": {"py"},
        "asyncio": {"async", "await", "gather"},
        "llm": {"model", "assistant"},
        "preference": {"prefer", "favorite", "like"},
    }

    def __init__(self, ctx, policy: Optional[RetrievalPolicy] = None):
        # ``project_memory`` alias preserved so existing body references work
        # unchanged during the transition to full MemoryContext adoption.
        self.project_memory = ctx
        self.policy = policy or RetrievalPolicy()

    def retrieve(
        self,
        *,
        query: Optional[str],
        max_tokens: int,
        episodic_n: int = 5,
        semantic_n: int = 5,
        cold_n: int = 5,
        cold_fallback: bool = True,
        cold_min_fill_ratio: float = 0.2,
    ):
        from engram.project_memory import ContextResult  # local import to avoid cycle

        budget = max_tokens
        result = ContextResult()

        working_budget = min(self.project_memory.budget.working, budget)
        result.working = self.project_memory.working.get_context_window(max_tokens=working_budget)
        result.working_tokens = sum(m.token_count for m in result.working)

        if not query:
            return result

        query_terms = self._query_terms(query)
        candidates: List[RetrievalCandidate] = []

        # Compute neural context before candidate scoring so the predicted
        # value vector can contribute to per-candidate affinity scores.
        neural_ctx = None
        if self.project_memory.neural_coord is not None:
            neural_ctx = self.project_memory.neural_coord.query_neural_context(query)
            if neural_ctx is not None:
                result.neural_meta = {k: v for k, v in neural_ctx.items()
                                      if k != "predicted_value"}

        candidates.extend(self._episodic_candidates(query, query_terms, episodic_n, neural_ctx))
        candidates.extend(self._semantic_candidates(query, query_terms, semantic_n, neural_ctx))

        if cold_fallback and self._needs_cold_fallback(candidates, budget, cold_min_fill_ratio):
            self.project_memory.telemetry.emit(
                "cold_fallback",
                "cold storage fallback retrieval triggered",
                project_id=self.project_memory.project_id,
                session_id=self.project_memory.session_id,
                query=query,
                n=cold_n,
            )
            candidates.extend(self._cold_candidates(query, query_terms, cold_n))

        selected = self._select_candidates(candidates, budget - result.working_tokens)

        # Record neural affinity hits for consolidation tracking.
        if (neural_ctx is not None and neural_ctx.get("warmed_up")
                and self.project_memory.neural_coord is not None):
            self.project_memory.neural_coord.record_retrieval_affinities(selected)

        for cand in selected:
            if cand.layer == "episodic":
                result.episodic.append(cand.payload)
                result.episodic_tokens += cand.token_count
            elif cand.layer == "semantic":
                result.semantic.append(cand.payload)
                result.semantic_tokens += cand.token_count
            elif cand.layer == "cold":
                result.cold.append(cand.payload)
                result.cold_tokens += cand.token_count

        return result

    def _episodic_candidates(self, query: str, query_terms: Set[str], n: int,
                             neural_ctx: Optional[Dict] = None) -> List[RetrievalCandidate]:
        # Expand the embedding query with synonyms before sending to ChromaDB.
        # ChromaDB embeds the query string via sentence-transformers; feeding it a
        # richer string that includes domain synonyms improves cosine similarity
        # for short or lexically-specific queries (e.g. "asyncio bug" → also
        # retrieves episodes that use "event loop error" or "async failure").
        #
        # The expansion appends unique synonym tokens as a space-separated suffix.
        # Sentence-transformers average subword token embeddings, so injecting
        # semantically related terms shifts the query vector toward the centroid
        # of the concept cluster.  Long natural-language queries are unaffected
        # because they already contain enough context — the extra tokens add
        # negligible signal.
        embedding_query = self._expand_query_for_embedding(query, query_terms)
        episodes = self.project_memory.search_episodes(embedding_query, n=n)
        out: List[RetrievalCandidate] = []
        now = time.time()

        for rank, ep in enumerate(episodes):
            text = getattr(ep, "text", "")
            token_count = max(1, self.project_memory._token_counter(text))
            importance = float(getattr(ep, "importance", 0.5) or 0.5)
            timestamp = float(getattr(ep, "timestamp", now) or now)
            age_days = max(0.0, (now - timestamp) / 86400.0)
            recency = max(0.0, 1.0 - min(age_days / 30.0, 1.0))
            lexical = self._lexical_overlap_terms(query_terms, text)
            density = self._term_density(query_terms, text)

            neural_affinity = 0.0
            if (neural_ctx is not None and neural_ctx.get("warmed_up")
                    and self.project_memory.neural_coord is not None):
                neural_affinity = self.project_memory.neural_coord.candidate_affinity(
                    neural_ctx["predicted_value"], text)

            score = (
                self.policy.episodic_weight_lexical * lexical
                + self.policy.episodic_weight_importance * importance
                + self.policy.episodic_weight_recency * recency
                + self.policy.episodic_weight_density * density
                + self.policy.neural_affinity_weight * max(0.0, neural_affinity)
                - rank * 0.02
            )
            out.append(
                RetrievalCandidate(
                    layer="episodic",
                    text=text,
                    payload=ep,
                    token_count=token_count,
                    score=score,
                    source_id=getattr(ep, "id", None),
                    metadata={"importance": importance, "recency": recency,
                              "lexical": lexical, "density": density,
                              "neural_affinity": neural_affinity},
                )
            )
        return out

    def _semantic_candidates(self, query: str, query_terms: Set[str], n: int,
                             neural_ctx: Optional[Dict] = None) -> List[RetrievalCandidate]:
        semantic = getattr(self.project_memory, "semantic", None)
        if semantic is None:
            return []

        generic_rows = self._fetch_semantic_rows(semantic, query, n)

        rows: List[RetrievalCandidate] = []
        seen_categories: Dict[str, float] = {}
        now = time.time()

        for row in generic_rows:
            row_type = str(row.get("type", "fact"))
            text = self._semantic_row_text(row)
            lexical = max(self._lexical_overlap_terms(query_terms, text), float(row.get("match_score", 0.0) or 0.0))
            if row_type != "preference" and lexical < self.policy.min_semantic_overlap:
                continue

            quality = self._semantic_quality(row)
            timestamp = float(row.get("timestamp", now) or now)
            recency = max(0.0, 1.0 - min(max(0.0, now - timestamp) / (180.0 * 86400.0), 1.0))
            density = self._term_density(query_terms, text)

            neural_affinity = 0.0
            if (neural_ctx is not None and neural_ctx.get("warmed_up")
                    and self.project_memory.neural_coord is not None):
                neural_affinity = self.project_memory.neural_coord.candidate_affinity(
                    neural_ctx["predicted_value"], text)

            score = (
                self.policy.semantic_weight_lexical * lexical
                + self.policy.semantic_weight_quality * quality
                + self.policy.semantic_weight_recency * recency
                + self.policy.semantic_weight_density * density
                + self.policy.neural_affinity_weight * max(0.0, neural_affinity)
            )

            if row_type == "preference":
                category = str(row.get("category", "general")).lower()
                if lexical < self.policy.min_semantic_overlap and not self._mentions_category(query_terms, category):
                    continue
                score += 0.04 if self._mentions_category(query_terms, category) else 0.0
                best = seen_categories.get(category)
                if best is not None and best >= score:
                    continue
                seen_categories[category] = score
                rows = [
                    item for item in rows
                    if not (item.layer == "semantic" and item.payload.get("type") == "preference" and str(item.payload.get("category", "")).lower() == category)
                ]

            rows.append(
                RetrievalCandidate(
                    layer="semantic",
                    text=text,
                    payload=row,
                    token_count=max(1, self.project_memory._token_counter(text)),
                    score=score,
                    source_id=str(row.get("id", "")),
                    metadata={"quality": quality, "lexical": lexical, "density": density, "recency": recency, "type": row_type, "neural_affinity": neural_affinity},
                )
            )

        rows.sort(key=lambda item: item.score, reverse=True)
        return rows[: max(n, 1)]

    def _fetch_semantic_rows(
        self,
        semantic: Any,
        query: str,
        n: int,
    ) -> List[Dict[str, Any]]:
        """Fetch generic rows from a semantic backend.

        Preferred path: ``search_generic_memories`` (satisfies
        ``SemanticLayerProtocol``).  Falls back to ``generic_memory_rows``
        for simple custom backends, and finally to individual ``list_*``
        methods for legacy or minimal implementations.

        Custom backends should implement ``search_generic_memories`` to
        avoid the fallback paths.
        """
        limit = max(n * 3, 12)
        if isinstance(semantic, SemanticLayerProtocol):
            return semantic.search_generic_memories(
                query,
                limit=limit,
                per_type_limit=self.policy.semantic_prefetch_limit,
                include_graph=self.policy.include_graph_context,
                graph_sentence_limit=self.policy.graph_sentence_limit,
            )
        if hasattr(semantic, "generic_memory_rows"):
            return semantic.generic_memory_rows(limit_per_type=limit)
        # Minimal fallback for custom backends with only list_* methods
        rows: List[Dict[str, Any]] = []
        for tag, method in (
            ("preference", "list_preferences"),
            ("event", "list_events"),
            ("fact", "list_facts"),
        ):
            fn = getattr(semantic, method, None)
            if fn is not None:
                rows.extend({"type": tag, **row} for row in fn(limit=limit))
        if not rows:
            logger.warning(
                "Semantic backend %s does not implement SemanticLayerProtocol "
                "(search_generic_memories). No semantic candidates retrieved.",
                semantic.__class__.__name__,
            )
        return rows

    def _semantic_row_text(self, row: Dict[str, Any]) -> str:
        row_type = str(row.get("type", "fact"))
        if row_type == "preference":
            return f"Preference ({row.get('category', 'general')}): {row.get('value', '')}".strip()
        if row_type == "event":
            return str(row.get("summary") or row.get("detail") or "").strip()
        if row_type == "graph_context":
            return str(row.get("content") or row.get("text") or "").strip()
        return str(row.get("content", "")).strip()

    @staticmethod
    def _semantic_quality(row: Dict[str, Any]) -> float:
        row_type = str(row.get("type", "fact"))
        if row_type == "preference":
            return float(row.get("strength", 0.5) or 0.5)
        if row_type == "event":
            return float(row.get("importance", 0.6) or 0.6)
        return float(row.get("confidence", 0.62) or 0.62)

    def _cold_candidates(self, query: str, query_terms: Set[str], n: int) -> List[RetrievalCandidate]:
        out: List[RetrievalCandidate] = []
        try:
            cold_rows = self.project_memory.cold.retrieve(
                query=query,
                n=n,
                project_id=self.project_memory.project_id,
            )
        except Exception:
            return out
        now = time.time()
        for row in cold_rows:
            text = row.get("text", "") or ""
            lexical = self._lexical_overlap_terms(query_terms, text)
            density = self._term_density(query_terms, text)
            timestamp = float(row.get("timestamp", now) or now)
            recency = max(0.0, 1.0 - min(max(0.0, now - timestamp) / (365.0 * 86400.0), 1.0))
            score = self.policy.cold_weight_lexical * lexical + self.policy.cold_weight_recency * recency + 0.08 * density + 0.10
            out.append(
                RetrievalCandidate(
                    layer="cold",
                    text=text,
                    payload=row,
                    token_count=max(1, self.project_memory._token_counter(text)),
                    score=score,
                    source_id=str(row.get("id", "")),
                    metadata={"lexical": lexical, "density": density, "recency": recency},
                )
            )
        return out

    def _select_candidates(self, candidates: Iterable[RetrievalCandidate], remaining_budget: int) -> List[RetrievalCandidate]:
        budget_left = max(0, remaining_budget)
        selected: List[RetrievalCandidate] = []
        seen = set()
        per_layer_tokens = {"episodic": 0, "semantic": 0, "cold": 0}
        layer_caps = {
            "episodic": self.project_memory.budget.episodic,
            "semantic": self.project_memory.budget.semantic,
            "cold": self.project_memory.budget.cold,
        }

        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        for cand in ranked:
            if budget_left <= 0:
                break
            norm = cand.normalized_text()
            if not norm or norm in seen:
                continue
            if self._is_near_duplicate(cand, selected):
                continue
            layer_cap = layer_caps.get(cand.layer, budget_left)
            if per_layer_tokens.get(cand.layer, 0) + cand.token_count > layer_cap:
                continue
            if cand.token_count > budget_left:
                continue
            diversity_penalty = self._max_similarity_penalty(cand, selected)
            adjusted_score = cand.score - diversity_penalty
            if adjusted_score <= 0:
                continue
            cand.metadata["adjusted_score"] = adjusted_score
            selected.append(cand)
            seen.add(norm)
            budget_left -= cand.token_count
            per_layer_tokens[cand.layer] = per_layer_tokens.get(cand.layer, 0) + cand.token_count
        return selected

    def _is_near_duplicate(self, candidate: RetrievalCandidate, selected: Sequence[RetrievalCandidate]) -> bool:
        for existing in selected:
            if self._text_similarity(candidate.text, existing.text) >= self.policy.near_duplicate_threshold:
                return True
        return False

    def _max_similarity_penalty(self, candidate: RetrievalCandidate, selected: Sequence[RetrievalCandidate]) -> float:
        if not selected:
            return 0.0
        best = max(self._text_similarity(candidate.text, item.text) for item in selected)
        if best < 0.45:
            return 0.0
        return min(self.policy.max_similarity_penalty, best * self.policy.diversity_penalty)

    def _needs_cold_fallback(self, candidates: List[RetrievalCandidate], budget: int, cold_min_fill_ratio: float) -> bool:
        episodic_tokens = sum(c.token_count for c in candidates if c.layer == "episodic")
        semantic_count = sum(1 for c in candidates if c.layer == "semantic")
        if semantic_count:
            return False
        episodic_budget_est = min(self.project_memory.budget.episodic, max(0, budget - self.project_memory.budget.working))
        if episodic_tokens == 0:
            return True
        return episodic_budget_est > 0 and episodic_tokens < int(episodic_budget_est * cold_min_fill_ratio)

    def _query_terms(self, query: str) -> Set[str]:
        terms = self._tokenize(query)
        expanded = set(terms)
        for term in list(terms):
            expanded.update(self.synonym_map.get(term, set()))
        return expanded

    def _expand_query_for_embedding(self, query: str, query_terms: Set[str]) -> str:
        """Return a synonym-expanded version of ``query`` for ChromaDB embedding.

        For each token in the query that has a synonym entry, appends the
        synonyms as extra words.  The expanded string is only used for the
        embedding call — scoring still uses the original ``query_terms`` so
        that lexical overlap scores remain accurate.

        Returns the original query unchanged when no synonyms apply (the common
        case for long natural-language queries).

        Example::

            "asyncio bug" → "asyncio bug async await gather issue error failure problem"
        """
        # Re-tokenize to get original terms only (query_terms is already expanded
        # by _query_terms and would filter out all synonyms as "already present").
        original_tokens = self._tokenize(query)
        extras: Set[str] = set()
        for term in original_tokens:
            for syn in self.synonym_map.get(term, set()):
                if syn not in original_tokens:
                    extras.add(syn)

        if not extras:
            return query

        # Append synonyms in sorted order for determinism
        return query + " " + " ".join(sorted(extras))

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        raw = re.findall(r"[A-Za-z0-9_./-]+", (text or "").lower())
        out: Set[str] = set()
        for tok in raw:
            if len(tok) > 2:
                out.add(tok)
            for piece in re.split(r"[._/-]+", tok):
                if len(piece) > 2:
                    out.add(piece)
        return out

    def _lexical_overlap_terms(self, query_terms: Set[str], text: str) -> float:
        text_terms = self._tokenize(text)
        if not query_terms or not text_terms:
            return 0.0
        return len(query_terms & text_terms) / max(1, len(query_terms))

    def _term_density(self, query_terms: Set[str], text: str) -> float:
        tokens = re.findall(r"[A-Za-z0-9_./-]+", (text or "").lower())
        if not tokens or not query_terms:
            return 0.0
        hits = sum(1 for tok in tokens if tok in query_terms)
        return min(1.0, hits / max(1, len(tokens)))

    def _text_similarity(self, a: str, b: str) -> float:
        aa = self._tokenize(a)
        bb = self._tokenize(b)
        if not aa or not bb:
            return 0.0
        return len(aa & bb) / max(1, len(aa | bb))

    def _mentions_category(self, query_terms: Set[str], category: str) -> bool:
        category_terms = self._tokenize(category)
        return bool(query_terms & category_terms)
