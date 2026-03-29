"""Search and scoring logic for Engram semantic memory.

``SemanticSearchMixin`` provides all retrieval methods for ``SemanticMemory``.
It is separated from the storage/schema concerns so each responsibility can
be read, tested, and extended independently.

Requires the host class to provide:
  - ``self.query(cypher, params) -> List[Dict]``
  - ``self.node_tables: set`` — populated table names
  - ``self._lock`` — threading.RLock (not used directly here, but relied on
    by ``self.query()``)

Author: Jeffrey Dean
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TERM_RE = re.compile(r"[A-Za-z0-9_./-]+")


class SemanticSearchMixin:
    """Mixin providing search and scoring methods for SemanticMemory.

    All methods assume ``self.query()``, ``self.node_tables``, and
    ``self._lock`` are provided by the host class.
    """

    # ------------------------------------------------------------------
    # Public search surface
    # ------------------------------------------------------------------

    def search_preferences(self, query: str, *, limit: int = 20) -> List[Dict[str, Any]]:
        """Search preference nodes by query overlap."""
        terms = self._search_terms(query)
        if not terms:
            return []
        where, params = self._contains_where_clause(
            ["lower(p.category)", "lower(p.value)"], terms
        )
        try:
            rows = self.query(
                f"""
                MATCH (p:Preference)
                WHERE {where}
                RETURN p.id as id, p.category as category, p.value as value,
                       p.strength as strength, p.timestamp as timestamp
                ORDER BY p.strength DESC, p.timestamp DESC
                LIMIT {int(limit * 3)}
                """,
                params,
            )
        except Exception:
            rows = []
        return self._score_semantic_rows(rows, row_type="preference", query=query, limit=limit)

    def search_facts(self, query: str, *, limit: int = 20) -> List[Dict[str, Any]]:
        """Search fact nodes by query overlap."""
        terms = self._search_terms(query)
        if not terms:
            return []
        where, params = self._contains_where_clause(
            ["lower(f.content)", "lower(f.source)", "lower(f.metadata)"], terms
        )
        try:
            rows = self.query(
                f"""
                MATCH (f:Fact)
                WHERE {where}
                RETURN f.id as id, f.content as content, f.timestamp as timestamp,
                       f.confidence as confidence, f.source as source, f.metadata as metadata
                ORDER BY f.confidence DESC, f.timestamp DESC
                LIMIT {int(limit * 3)}
                """,
                params,
            )
        except Exception:
            rows = []
        return self._score_semantic_rows(rows, row_type="fact", query=query, limit=limit)

    def search_events(self, query: str, *, limit: int = 20) -> List[Dict[str, Any]]:
        """Search event nodes by query overlap."""
        terms = self._search_terms(query)
        if not terms:
            return []
        fields = [
            "lower(coalesce(e.summary, ''))",
            "lower(coalesce(e.detail, ''))",
            "lower(coalesce(e.name, ''))",
            "lower(coalesce(e.description, ''))",
            "lower(coalesce(e.source, ''))",
            "lower(coalesce(e.metadata, ''))",
        ]
        where, params = self._contains_where_clause(fields, terms)
        try:
            rows = self.query(
                f"""
                MATCH (e:Event)
                WHERE {where}
                RETURN e.id as id,
                       coalesce(e.summary, e.name) as summary,
                       coalesce(e.detail, e.description) as detail,
                       coalesce(e.timestamp, e.start_time, 0.0) as timestamp,
                       coalesce(e.importance, 0.55) as importance,
                       coalesce(e.source, 'event') as source,
                       coalesce(e.metadata, '') as metadata
                ORDER BY importance DESC, timestamp DESC
                LIMIT {int(limit * 3)}
                """,
                params,
            )
        except Exception:
            rows = []
        return self._score_semantic_rows(rows, row_type="event", query=query, limit=limit)

    def search_graph_context(
        self,
        query: str,
        *,
        limit: int = 6,
        max_hops: int = 1,
        direct_sentence_limit: int = 4,
    ) -> List[Dict[str, Any]]:
        """Search the extraction graph using entity matching and co-occurrence.

        Finds query-matching Entity nodes, pulls their source Sentence nodes,
        then walks CO_OCCURS edges up to ``max_hops`` to gather supporting
        context from related entities.

        Returns rows with ``type="graph_context"`` suitable for inclusion in
        ``search_generic_memories``.
        """
        if "Entity" not in self.node_tables or "Sentence" not in self.node_tables:
            return []
        terms = self._search_terms(query)
        if not terms:
            return []

        # Step 1: find matching entities
        entity_rows: List[Dict[str, Any]] = []
        for term in terms:
            try:
                rows = self.query(
                    f"""
                    MATCH (e:Entity)
                    WHERE lower(e.normalized) CONTAINS $term OR lower(e.text) CONTAINS $term
                    RETURN e.id as id, e.text as text, e.normalized as normalized,
                           e.score as score, e.mention_count as mention_count
                    ORDER BY e.score DESC, e.mention_count DESC
                    LIMIT {max(3, int(limit))}
                    """,
                    {"term": term.lower()},
                )
            except Exception:
                rows = []
            entity_rows.extend(rows)

        ranked_entities = self._score_graph_entities(entity_rows, query)
        if not ranked_entities:
            return []

        seen_sentences: set = set()
        out: List[Dict[str, Any]] = []

        for entity in ranked_entities[: max(3, limit)]:
            entity_id = entity.get("id")
            entity_text = str(entity.get("text") or "")
            base_boost = float(entity.get("match_score", 0.0) or 0.0)
            if not entity_id:
                continue

            # Step 2: direct evidence sentences
            try:
                direct_rows = self.query(
                    f"""
                    MATCH (s:Sentence)-[:MENTIONS]->(e:Entity)
                    WHERE e.id = $eid
                    RETURN s.id as id, s.text as text, s.timestamp as timestamp
                    ORDER BY s.timestamp DESC
                    LIMIT {int(direct_sentence_limit)}
                    """,
                    {"eid": entity_id},
                )
            except Exception:
                direct_rows = []

            for row in direct_rows:
                sentence = str(row.get("text") or "").strip()
                if not sentence or sentence in seen_sentences:
                    continue
                seen_sentences.add(sentence)
                out.append({
                    "type": "graph_context",
                    "id": row.get("id"),
                    "content": sentence,
                    "timestamp": row.get("timestamp"),
                    "confidence": min(0.95, 0.52 + base_boost * 0.4),
                    "source": f"entity:{entity_text}",
                    "match_score": min(
                        1.0,
                        self._query_overlap_score(terms, sentence) + base_boost * 0.35,
                    ),
                    "entity": entity_text,
                    "graph_distance": 0,
                })

            # Step 3: co-occurring entity sentences (multi-hop)
            if max_hops >= 1:
                try:
                    related_rows = self.query(
                        f"""
                        MATCH (e:Entity)-[r:CO_OCCURS*1..{int(max_hops)}]-(other:Entity)
                        WHERE e.id = $eid AND e.id <> other.id
                        RETURN DISTINCT other.id as id, other.text as text,
                               other.score as score, other.mention_count as mention_count
                        LIMIT {max(4, int(limit))}
                        """,
                        {"eid": entity_id},
                    )
                except Exception:
                    related_rows = []

                for related in self._score_graph_entities(related_rows, query)[
                    : max(2, limit // 2 or 1)
                ]:
                    rid = related.get("id")
                    if not rid:
                        continue
                    try:
                        related_sentences = self.query(
                            """
                            MATCH (s:Sentence)-[:MENTIONS]->(e:Entity)
                            WHERE e.id = $eid
                            RETURN s.id as id, s.text as text, s.timestamp as timestamp
                            ORDER BY s.timestamp DESC
                            LIMIT 2
                            """,
                            {"eid": rid},
                        )
                    except Exception:
                        related_sentences = []

                    for row in related_sentences:
                        sentence = str(row.get("text") or "").strip()
                        if not sentence or sentence in seen_sentences:
                            continue
                        seen_sentences.add(sentence)
                        out.append({
                            "type": "graph_context",
                            "id": row.get("id"),
                            "content": sentence,
                            "timestamp": row.get("timestamp"),
                            "confidence": min(
                                0.9,
                                0.46 + float(related.get("match_score", 0.0)) * 0.3,
                            ),
                            "source": f"related_entity:{related.get('text')}",
                            "match_score": min(
                                1.0,
                                self._query_overlap_score(terms, sentence)
                                + float(related.get("match_score", 0.0)) * 0.2,
                            ),
                            "entity": related.get("text"),
                            "graph_distance": 1,
                        })

        out.sort(
            key=lambda item: (
                float(item.get("match_score", 0.0)),
                float(item.get("confidence", 0.0)),
            ),
            reverse=True,
        )
        return out[:limit]

    def search_generic_memories(
        self,
        query: str,
        *,
        limit: int = 20,
        per_type_limit: int = 60,
        include_graph: bool = True,
        graph_sentence_limit: int = 6,
    ) -> List[Dict[str, Any]]:
        """Return generic semantic rows relevant to a text query.

        This is the converged semantic retrieval surface used by
        ``UnifiedRetriever``. It runs targeted graph-native queries for each
        memory type, then merges and deduplicates the results.

        Satisfies ``SemanticLayerProtocol``.
        """
        terms = self._search_terms(query)
        if not terms:
            return []

        rows: List[Dict[str, Any]] = []
        rows.extend(self.search_preferences(query, limit=min(per_type_limit, max(limit, 8))))
        rows.extend(self.search_events(query, limit=min(per_type_limit, max(limit, 8))))
        rows.extend(self.search_facts(query, limit=min(per_type_limit, max(limit, 8))))
        rows.extend(self.search_typed_relations(query, limit=min(per_type_limit, max(limit, 8))))

        if include_graph:
            rows.extend(self.search_graph_context(
                query,
                limit=min(graph_sentence_limit, max(2, limit // 2 or 1)),
            ))

        deduped = self._dedupe_ranked_rows(rows)
        deduped.sort(key=lambda item: float(item.get("match_score", 0.0)), reverse=True)
        return deduped[:limit]

    def search_typed_relations(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search PREFERS, USES, and KNOWS_ABOUT typed edges by query overlap.

        These edges connect Preference/Fact nodes to Entity nodes via typed
        relations extracted from conversation text.  Results are formatted as
        standard semantic rows so they merge cleanly with other retrieval types.

        Returns rows with type "typed_relation" and fields:
            relation_type, subject_text, entity_text, confidence,
            text (human-readable), match_score, timestamp.
        """
        terms = self._search_terms(query)
        if not terms:
            return []

        rows: List[Dict[str, Any]] = []

        # PREFERS: Preference → Entity
        try:
            if "Preference" in self.node_tables and "Entity" in self.node_tables:
                pref_rows = self.query(
                    f"""
                    MATCH (p:Preference)-[r:PREFERS]->(e:Entity)
                    RETURN p.category AS category, p.value AS pref_value,
                           e.text AS entity, r.confidence AS confidence,
                           r.source_text AS source_text, r.timestamp AS timestamp
                    LIMIT {int(limit * 3)}
                    """
                )
                for row in (pref_rows or []):
                    entity = row.get("entity", "")
                    pref_val = row.get("pref_value", "")
                    text = f"Prefers {pref_val} (related to {entity})"
                    score = self._query_overlap_score(
                        terms, f"{pref_val} {entity} {row.get('source_text', '')}"
                    )
                    if score > 0:
                        rows.append({
                            "type": "typed_relation",
                            "relation_type": "PREFERS",
                            "subject_text": pref_val,
                            "entity_text": entity,
                            "confidence": row.get("confidence", 0.0),
                            "text": text,
                            "value": text,
                            "match_score": score * float(row.get("confidence", 0.75)),
                            "timestamp": row.get("timestamp", 0.0),
                        })
        except Exception:
            pass

        # USES and KNOWS_ABOUT: Fact → Entity
        for rel_type in ("USES", "KNOWS_ABOUT"):
            try:
                if "Fact" in self.node_tables and "Entity" in self.node_tables:
                    rel_rows = self.query(
                        f"""
                        MATCH (f:Fact)-[r:{rel_type}]->(e:Entity)
                        RETURN f.content AS fact_content, e.text AS entity,
                               r.confidence AS confidence,
                               r.source_text AS source_text, r.timestamp AS timestamp
                        LIMIT {int(limit * 3)}
                        """
                    )
                    for row in (rel_rows or []):
                        entity = row.get("entity", "")
                        fact = row.get("fact_content", "")
                        verb = "Uses" if rel_type == "USES" else "Knows about"
                        text = f"{verb}: {entity}"
                        score = self._query_overlap_score(
                            terms, f"{entity} {fact} {row.get('source_text', '')}"
                        )
                        if score > 0:
                            rows.append({
                                "type": "typed_relation",
                                "relation_type": rel_type,
                                "subject_text": fact[:120],
                                "entity_text": entity,
                                "confidence": row.get("confidence", 0.0),
                                "text": text,
                                "value": text,
                                "match_score": score * float(row.get("confidence", 0.70)),
                                "timestamp": row.get("timestamp", 0.0),
                            })
            except Exception:
                pass

        rows.sort(key=lambda r: float(r.get("match_score", 0.0)), reverse=True)
        return rows[:limit]

    # ------------------------------------------------------------------
    # Internal scoring and text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _search_terms(query: str) -> List[str]:
        """Extract meaningful tokens from a query string."""
        return [
            tok for tok in _TERM_RE.findall((query or "").lower())
            if len(tok) > 2
        ]

    @staticmethod
    def _query_overlap_score(terms: List[str], text: str) -> float:
        """Overlap score: fraction of query terms present in text, weighted by density."""
        text_terms = {
            tok for tok in _TERM_RE.findall((text or "").lower())
            if len(tok) > 2
        }
        if not terms or not text_terms:
            return 0.0
        overlap = len(set(terms) & text_terms)
        if overlap == 0:
            return 0.0
        density = overlap / max(1, len(text_terms))
        coverage = overlap / max(1, len(set(terms)))
        return min(1.0, 0.7 * coverage + 0.3 * density)

    @staticmethod
    def _semantic_row_text(row: Dict[str, Any]) -> str:
        """Extract the best available text field from a semantic row."""
        return str(
            row.get("content")
            or row.get("summary")
            or row.get("detail")
            or row.get("value")
            or row.get("text")
            or row.get("description")
            or ""
        ).strip()

    def _score_semantic_rows(
        self,
        rows: List[Dict[str, Any]],
        *,
        row_type: str,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Score and filter a list of raw Kuzu rows by query overlap."""
        terms = self._search_terms(query)
        scored: List[Dict[str, Any]] = []
        for row in rows:
            text = self._semantic_row_text(row)
            if row_type == "preference":
                text = f"Preference ({row.get('category', 'general')}): {row.get('value', '')}".strip()
            match_score = self._query_overlap_score(terms, text)
            if match_score <= 0:
                continue
            item = {"type": row_type, "match_score": match_score, **row}
            item["semantic_text"] = text
            scored.append(item)
        scored.sort(key=lambda item: float(item.get("match_score", 0.0)), reverse=True)
        return scored[:limit]

    def _score_graph_entities(
        self,
        rows: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """Score and deduplicate entity rows from a graph query."""
        terms = self._search_terms(query)
        scored: List[Dict[str, Any]] = []
        seen: set = set()
        for row in rows:
            entity_id = row.get("id")
            if entity_id in seen:
                continue
            seen.add(entity_id)
            text = str(row.get("text") or row.get("normalized") or "").strip()
            if not text:
                continue
            overlap = self._query_overlap_score(terms, text)
            quality = min(1.0, float(row.get("score", 0.0) or 0.0))
            mentions = min(1.0, float(row.get("mention_count", 0.0) or 0.0) / 8.0)
            row = dict(row)
            row["match_score"] = min(1.0, overlap * 0.7 + quality * 0.2 + mentions * 0.1)
            scored.append(row)
        scored.sort(key=lambda item: float(item.get("match_score", 0.0)), reverse=True)
        return scored

    @staticmethod
    def _contains_where_clause(
        fields: List[str],
        terms: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """Build a Kuzu WHERE clause that matches any term in any of the fields."""
        clauses = []
        params: Dict[str, Any] = {}
        for idx, term in enumerate(terms):
            p = f"term_{idx}"
            params[p] = term.lower()
            field_clause = " OR ".join(f"{field} CONTAINS ${p}" for field in fields)
            clauses.append(f"({field_clause})")
        if not clauses:
            return "TRUE", {}
        return " OR ".join(clauses), params

    def _dedupe_ranked_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Deduplicate rows by (type, id-or-text), keeping the highest-scoring copy."""
        seen: set = set()
        out: List[Dict[str, Any]] = []
        for row in sorted(
            rows,
            key=lambda item: float(item.get("match_score", 0.0)),
            reverse=True,
        ):
            key = (
                row.get("type"),
                row.get("id") or self._semantic_row_text(row).lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out
