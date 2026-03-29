"""Graph extraction pipeline for Engram semantic memory.

Populates Kuzu graph from text WITHOUT LLM calls.

Architecture (from LinearRAG/E²GraphRAG papers):

    Text → Sentences → Entities → Co-occurrence Graph

Extraction methods:
    1. TF-IDF (default) — statistical, zero dependencies beyond scikit-learn
    2. SpaCy (optional) — NLP-based, gives entity types (PERSON, ORG, etc.)

Relation method:
    Co-occurrence only — entities appearing in the same sentence are linked.
    No typed relations (WORKS_AT, FOUNDED, etc.). The LLM figures out
    relationship semantics from the source text at query time.

Graph structure (tri-graph):
    Entity nodes    — extracted terms/named entities
    Sentence nodes  — source text fragments
    Edges:
        CO_OCCURS   — entity↔entity (same sentence)
        MENTIONS    — sentence→entity (provenance)

Cost: $0 for indexing. One LLM call at query time.

Usage:
    from engram.memory.extraction import GraphExtractor, ExtractionConfig

    extractor = GraphExtractor(semantic_memory, config=ExtractionConfig())
    stats = extractor.index_text(text)
    # stats = {"entities": 47, "sentences": 23, "relations": 156, "llm_calls": 0}

    # Or index multiple documents
    stats = extractor.index_documents(["doc1...", "doc2..."])

Author: Jeffrey Dean
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractedRelation:
    """A typed relation extracted from conversation text."""
    subject_id: str       # Kuzu node id of the subject (Preference or Fact)
    subject_table: str    # "Preference" or "Fact"
    relation_type: str    # "PREFERS", "USES", "KNOWS_ABOUT"
    object_text: str      # Raw entity text (will be normalised → Entity node)
    object_id: str        # Entity node id (sha256[:16] of normalised text)
    source_text: str      # The sentence that triggered this extraction
    confidence: float = 0.75


# Compiled pattern sets for typed relation extraction.
# These mirror the patterns in IngestionPolicy but produce graph edges
# rather than IngestionDecision payloads.
# _TERM_CAP captures 1-4 words, stopping before prepositions/conjunctions.
# At most 2 words; each word is alphanumeric+symbols (no spaces within the separator words)
_TERM_CAP = r"(?P<value>[A-Za-z][A-Za-z0-9._-]*(?:\s+[A-Za-z][A-Za-z0-9._-]*){0,1})"

_PREFER_PATTERNS = [
    # "I prefer/like/love/choose <TERM>"
    re.compile(
        r"\bI (?:strongly |really |always )?(?:prefer|like|love|choose)\s+" + _TERM_CAP,
        re.IGNORECASE,
    ),
    # "<TERM> is my preferred/favorite/go-to"
    re.compile(
        r"\b" + _TERM_CAP + r"\s+is (?:my (?:preferred|favorite|go-to)|better|best)\b",
        re.IGNORECASE,
    ),
]

_USES_PATTERNS = [
    # "I use/am using/work with <TERM>"
    re.compile(
        r"\bI (?:use|am using|work with|rely on|depend on)\s+" + _TERM_CAP,
        re.IGNORECASE,
    ),
    # "we use/adopted/switched to <TERM>"
    re.compile(
        r"\bwe (?:use|are using|adopted|switched to)\s+" + _TERM_CAP,
        re.IGNORECASE,
    ),
]

_KNOWS_PATTERNS = [
    # "I work at/for <TERM>"
    re.compile(
        r"\bI (?:work|worked) (?:at|for)\s+" + _TERM_CAP,
        re.IGNORECASE,
    ),
    # "<TERM> is a tool/framework/library/language/database/platform"
    re.compile(
        r"\b" + _TERM_CAP + r"\s+is (?:a|an) (?:tool|framework|library|language|database|platform)\b",
        re.IGNORECASE,
    ),
]

# Generic stop-words that produce useless entity edges
_RELATION_STOP_VALUES = {
    "it", "this", "that", "them", "they", "one", "something", "anything",
    "everything", "nothing", "some", "any", "all", "both", "each",
    "many", "much", "more", "most", "other", "another", "such",
}


def extract_typed_relations(
    sentences: List[str],
    semantic_memory,
    config: "ExtractionConfig",
) -> List["ExtractedRelation"]:
    """Extract typed PREFERS / USES / KNOWS_ABOUT relations from sentences.

    Each matched relation is linked to an existing Preference or Fact node
    (if one exists for this project) and an Entity node (created on demand).

    This is a pure heuristic pass — no LLM calls.  False positives are
    acceptable; the graph is used as a soft signal for retrieval ranking,
    not as ground truth.

    Args:
        sentences:       List of sentence strings to process.
        semantic_memory: SemanticMemory instance (for id lookup and node creation).
        config:          ExtractionConfig (for entity filtering settings).

    Returns:
        List of ExtractedRelation objects ready to write to Kuzu.
    """
    relations: List[ExtractedRelation] = []
    seen: Set[Tuple[str, str, str]] = set()  # (subject_id, rel_type, object_id)

    def _entity_id(text: str) -> str:
        norm = _normalize_entity(text)
        return hashlib.sha256(norm.encode()).hexdigest()[:16]

    # Prepositions and articles that commonly trail a captured term
    _TRAILING_STOP = {
        "over", "as", "on", "at", "in", "for", "with", "to", "of",
        "the", "a", "an", "and", "or", "but", "my", "our", "your",
        "via", "by", "from", "into", "onto", "than",
    }

    def _add(subject_id: str, subject_table: str, rel_type: str,
             value_text: str, source: str, confidence: float) -> None:
        value_text = value_text.strip(" .,;:!?\"'")
        # Strip a single trailing stop-word/preposition (e.g. "pytest over" → "pytest")
        words = value_text.split()
        if len(words) > 1 and words[-1].lower() in _TRAILING_STOP:
            value_text = " ".join(words[:-1])
        norm = _normalize_entity(value_text)
        if not _is_valid_entity(norm, config):
            return
        if norm in _RELATION_STOP_VALUES:
            return
        obj_id = _entity_id(norm)
        key = (subject_id, rel_type, obj_id)
        if key in seen:
            return
        seen.add(key)
        relations.append(ExtractedRelation(
            subject_id=subject_id,
            subject_table=subject_table,
            relation_type=rel_type,
            object_text=value_text,
            object_id=obj_id,
            source_text=source[:200],
            confidence=confidence,
        ))

    # Retrieve existing Preference and Fact nodes from Kuzu so we can link to them.
    # Fall back to a synthetic "root_fact" node if no specific match found.
    try:
        prefs = semantic_memory.query(
            "MATCH (p:Preference) RETURN p.id as id, p.category as category, p.value as value"
        )
    except Exception:
        prefs = []

    try:
        facts = semantic_memory.query(
            "MATCH (f:Fact) RETURN f.id as id, f.content as content"
        )
    except Exception:
        facts = []

    # Build quick lookup: normalised token → node id
    pref_lookup: Dict[str, str] = {
        _normalize_entity(p.get("value", "")): p["id"]
        for p in prefs if p.get("id") and p.get("value")
    }
    fact_lookup: Dict[str, str] = {
        _normalize_entity(f.get("content", "")[:60]): f["id"]
        for f in facts if f.get("id") and f.get("content")
    }

    # Fallback synthetic fact id used when no specific fact node matches
    _ROOT_FACT = "fact_root_user_context"

    for sentence in sentences:
        # PREFERS relations
        for pattern in _PREFER_PATTERNS:
            m = pattern.search(sentence)
            if not m:
                continue
            value = m.group("value")
            # Find the closest matching preference node; use root fact if none
            norm_v = _normalize_entity(value)
            subject_id = pref_lookup.get(norm_v)
            if subject_id:
                _add(subject_id, "Preference", "PREFERS", value, sentence, 0.80)
            else:
                # Emit a KNOWS_ABOUT from root fact instead
                _add(_ROOT_FACT, "Fact", "KNOWS_ABOUT", value, sentence, 0.65)
            break  # one per sentence per pass

        # USES relations
        for pattern in _USES_PATTERNS:
            m = pattern.search(sentence)
            if not m:
                continue
            value = m.group("value")
            norm_v = _normalize_entity(value)
            subject_id = fact_lookup.get(norm_v, _ROOT_FACT)
            _add(subject_id, "Fact", "USES", value, sentence, 0.75)
            break

        # KNOWS_ABOUT relations
        for pattern in _KNOWS_PATTERNS:
            m = pattern.search(sentence)
            if not m:
                continue
            value = m.group("value")
            norm_v = _normalize_entity(value)
            subject_id = fact_lookup.get(norm_v, _ROOT_FACT)
            _add(subject_id, "Fact", "KNOWS_ABOUT", value, sentence, 0.70)
            break

    return relations


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntity:
    """An entity extracted from text."""
    text: str
    normalized: str          # Lowercased, whitespace-collapsed
    entity_type: str = "TERM"  # TERM (TF-IDF) or PERSON/ORG/GPE/etc (SpaCy)
    score: float = 0.0       # TF-IDF score or SpaCy confidence
    source_sentences: List[int] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Deterministic ID from normalized text."""
        return hashlib.sha256(self.normalized.encode()).hexdigest()[:16]


@dataclass
class ExtractedSentence:
    """A sentence extracted from text."""
    index: int
    text: str
    document_index: int = 0

    @property
    def id(self) -> str:
        return f"sent_{self.document_index}_{self.index}"


@dataclass
class CooccurrenceEdge:
    """A co-occurrence relationship between two entities."""
    entity_a_id: str
    entity_b_id: str
    sentence_ids: List[str] = field(default_factory=list)
    count: int = 1  # Number of sentences where they co-occur

    @property
    def id(self) -> str:
        pair = tuple(sorted([self.entity_a_id, self.entity_b_id]))
        return f"cooc_{pair[0]}_{pair[1]}"


@dataclass
class ExtractionConfig:
    """Configuration for graph extraction."""
    # Method: "tfidf" or "spacy"
    method: str = "tfidf"

    # TF-IDF settings
    tfidf_min_score: float = 0.20      # Minimum TF-IDF score to keep
    tfidf_max_entities_per_chunk: int = 15  # Cap per chunk
    tfidf_ngram_range: Tuple[int, int] = (1, 2)  # Uni/bigrams (trigrams too noisy)
    tfidf_min_df: int = 1              # Min document frequency
    tfidf_max_df: float = 0.85         # Max document frequency (fraction)

    # SpaCy settings
    spacy_model: str = "en_core_web_sm"
    spacy_entity_types: Optional[List[str]] = None  # None = all types

    # Entity filtering
    min_entity_length: int = 2         # Skip single characters
    max_entity_length: int = 80        # Skip absurdly long strings
    stop_entities: Set[str] = field(default_factory=lambda: {
        "example", "etc", "also", "however", "therefore",
        "using", "used", "use", "may", "can", "will",
    })

    # Sentence splitting
    min_sentence_length: int = 10      # Skip very short sentences
    max_sentence_length: int = 1000    # Truncate very long ones

    # Co-occurrence
    min_cooccurrence_count: int = 1    # Min co-occurrences to create edge

    # Graph population
    store_sentences: bool = True       # Also store Sentence nodes
    update_existing: bool = True       # Update scores for existing entities

    # Typed relation extraction (PREFERS / USES / KNOWS_ABOUT)
    extract_typed: bool = True         # Run TypedRelationExtractor after co-occurrence pass


@dataclass
class ExtractionStats:
    """Statistics from an extraction run."""
    entities: int = 0
    sentences: int = 0
    relations: int = 0
    typed_relations: int = 0   # PREFERS / USES / KNOWS_ABOUT edges added
    documents: int = 0
    llm_calls: int = 0
    elapsed_s: float = 0.0
    method: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": self.entities,
            "sentences": self.sentences,
            "relations": self.relations,
            "typed_relations": self.typed_relations,
            "documents": self.documents,
            "llm_calls": self.llm_calls,
            "elapsed_s": round(self.elapsed_s, 3),
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Regex: split on .!? followed by space+uppercase, or newlines
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?:\n\s*\n)')


def split_sentences(text: str, config: ExtractionConfig) -> List[str]:
    """Split text into sentences.

    Uses regex heuristic. Not perfect, but fast and dependency-free.
    For better results, SpaCy's sentence splitter is used when method="spacy".
    """
    # First split on double newlines (paragraphs)
    raw = _SENT_SPLIT.split(text)

    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) < config.min_sentence_length:
            continue
        if len(s) > config.max_sentence_length:
            s = s[:config.max_sentence_length]
        sentences.append(s)

    return sentences


# ---------------------------------------------------------------------------
# TF-IDF entity extraction
# ---------------------------------------------------------------------------

def extract_entities_tfidf(
    sentences: List[str],
    config: ExtractionConfig,
) -> List[ExtractedEntity]:
    """Extract entities using TF-IDF scoring.

    Entities = statistically distinctive terms. High TF-IDF score means
    the term is frequent in its source chunk but rare across the corpus.
    Zero hallucination (can't hallucinate a statistic).

    Requires: scikit-learn
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        raise ImportError(
            "TF-IDF extraction requires scikit-learn. "
            "Install with: pip install scikit-learn"
        )

    if not sentences:
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=config.tfidf_ngram_range,
        stop_words="english",
        min_df=config.tfidf_min_df,
        max_df=config.tfidf_max_df,
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9_.-]*[A-Za-z0-9]\b",
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # Empty vocabulary (e.g., all stop words)
        return []

    feature_names = vectorizer.get_feature_names_out()

    # Collect entities per sentence
    entity_map: Dict[str, ExtractedEntity] = {}  # normalized → entity

    for sent_idx in range(len(sentences)):
        scores = tfidf_matrix[sent_idx].toarray()[0]
        # Get top-scoring terms for this sentence
        top_indices = np.argsort(scores)[::-1]

        count = 0
        for idx in top_indices:
            if scores[idx] < config.tfidf_min_score:
                break
            if count >= config.tfidf_max_entities_per_chunk:
                break

            term = feature_names[idx]
            normalized = _normalize_entity(term)

            if not _is_valid_entity(normalized, config):
                continue

            if normalized in entity_map:
                entity_map[normalized].score = max(entity_map[normalized].score, scores[idx])
                entity_map[normalized].source_sentences.append(sent_idx)
            else:
                entity_map[normalized] = ExtractedEntity(
                    text=term,
                    normalized=normalized,
                    entity_type="TERM",
                    score=float(scores[idx]),
                    source_sentences=[sent_idx],
                )
            count += 1

    return list(entity_map.values())


# ---------------------------------------------------------------------------
# SpaCy entity extraction (optional)
# ---------------------------------------------------------------------------

def extract_entities_spacy(
    sentences: List[str],
    config: ExtractionConfig,
) -> List[ExtractedEntity]:
    """Extract typed entities using SpaCy NER.

    Gives entity types (PERSON, ORG, GPE, DATE, etc.) but requires
    the spacy package and a language model.

    Requires: spacy, en_core_web_sm (or other model)
    """
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "SpaCy extraction requires spacy. "
            "Install with: pip install spacy && python -m spacy download en_core_web_sm"
        )

    nlp = spacy.load(config.spacy_model)
    allowed_types = set(config.spacy_entity_types) if config.spacy_entity_types else None

    entity_map: Dict[str, ExtractedEntity] = {}

    for sent_idx, sentence in enumerate(sentences):
        doc = nlp(sentence)
        for ent in doc.ents:
            if allowed_types and ent.label_ not in allowed_types:
                continue

            normalized = _normalize_entity(ent.text)
            if not _is_valid_entity(normalized, config):
                continue

            if normalized in entity_map:
                entity_map[normalized].source_sentences.append(sent_idx)
                entity_map[normalized].score += 1.0  # Frequency as score
            else:
                entity_map[normalized] = ExtractedEntity(
                    text=ent.text,
                    normalized=normalized,
                    entity_type=ent.label_,
                    score=1.0,
                    source_sentences=[sent_idx],
                )

    return list(entity_map.values())


# ---------------------------------------------------------------------------
# N-gram subsumption filter
# ---------------------------------------------------------------------------

def _filter_subsumed_entities(entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
    """Remove n-gram entities that are substrings of higher-scoring entities.

    If "asyncio event loop" (score=0.4) exists, remove "asyncio event"
    and "event loop" since the longer form is more specific.
    Keeps entity counts manageable and reduces quadratic co-occurrence costs.
    """
    if len(entities) <= 1:
        return entities

    # Sort by length descending; longer entities subsume shorter
    by_length = sorted(entities, key=lambda e: len(e.normalized), reverse=True)
    kept: List[ExtractedEntity] = []
    kept_texts: Set[str] = set()

    for entity in by_length:
        subsumed = any(
            entity.normalized in kt and entity.normalized != kt
            for kt in kept_texts
        )
        if not subsumed:
            kept.append(entity)
            kept_texts.add(entity.normalized)

    return kept


# ---------------------------------------------------------------------------
# Co-occurrence relation extraction
# ---------------------------------------------------------------------------

def extract_cooccurrences(
    entities: List[ExtractedEntity],
    sentences: List[ExtractedSentence],
    config: ExtractionConfig,
) -> List[CooccurrenceEdge]:
    """Build co-occurrence edges between entities in the same sentence.

    No LLM needed. No typed relations. The downstream LLM figures out
    the nature of the relationship from the source text.
    """
    # Build sentence → entities index
    sent_to_entities: Dict[int, List[ExtractedEntity]] = {}
    for entity in entities:
        for sent_idx in entity.source_sentences:
            sent_to_entities.setdefault(sent_idx, []).append(entity)

    # Build pairwise co-occurrences
    edge_map: Dict[str, CooccurrenceEdge] = {}  # pair_key → edge

    for sent_idx, ents in sent_to_entities.items():
        sent_id = sentences[sent_idx].id if sent_idx < len(sentences) else f"sent_{sent_idx}"

        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                a, b = ents[i], ents[j]
                if a.id == b.id:
                    continue

                pair_key = tuple(sorted([a.id, b.id]))
                key = f"{pair_key[0]}_{pair_key[1]}"

                if key in edge_map:
                    edge_map[key].count += 1
                    edge_map[key].sentence_ids.append(sent_id)
                else:
                    edge_map[key] = CooccurrenceEdge(
                        entity_a_id=a.id,
                        entity_b_id=b.id,
                        sentence_ids=[sent_id],
                        count=1,
                    )

    # Filter by minimum co-occurrence count
    return [
        e for e in edge_map.values()
        if e.count >= config.min_cooccurrence_count
    ]


# ---------------------------------------------------------------------------
# Entity normalization and filtering
# ---------------------------------------------------------------------------

def _normalize_entity(text: str) -> str:
    """Normalize entity text for deduplication."""
    # Lowercase, collapse whitespace, strip punctuation edges
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(".,;:!?\"'()-")
    return normalized


def _is_valid_entity(normalized: str, config: ExtractionConfig) -> bool:
    """Check if entity passes quality filters."""
    if len(normalized) < config.min_entity_length:
        return False
    if len(normalized) > config.max_entity_length:
        return False
    if normalized in config.stop_entities:
        return False
    # Skip pure numbers
    if normalized.replace(".", "").replace(",", "").isdigit():
        return False
    return True


# ---------------------------------------------------------------------------
# Graph population
# ---------------------------------------------------------------------------

class GraphExtractor:
    """Populates Engram's semantic memory graph from text.

    Wraps entity extraction + co-occurrence + Kuzu insertion.

    Usage:
        extractor = GraphExtractor(semantic_memory)
        stats = extractor.index_text("Python's asyncio module provides...")
        stats = extractor.index_documents(["doc1...", "doc2..."])
    """

    def __init__(
        self,
        semantic_memory,
        config: Optional[ExtractionConfig] = None,
    ):
        """
        Args:
            semantic_memory: SemanticMemory instance (must have add_node, add_relationship, query)
            config: Extraction configuration. None = defaults (TF-IDF).
        """
        self.memory = semantic_memory
        self.config = config or ExtractionConfig()
        self._ensure_schema()

    def _ensure_schema(self):
        """Create Entity/Sentence/CO_OCCURS/MENTIONS/typed relation tables if needed."""
        mem = self.memory

        mem._create_node_table_safe("Entity", [
            ("id", "STRING"),
            ("text", "STRING"),
            ("normalized", "STRING"),
            ("entity_type", "STRING"),
            ("score", "DOUBLE"),
            ("mention_count", "INT64"),
        ], "id")

        if self.config.store_sentences:
            mem._create_node_table_safe("Sentence", [
                ("id", "STRING"),
                ("text", "STRING"),
                ("doc_index", "INT64"),
                ("sent_index", "INT64"),
                ("timestamp", "DOUBLE"),
            ], "id")

            mem._create_rel_table_safe("MENTIONS", "Sentence", "Entity", [
                ("position", "INT64"),
            ])

        mem._create_rel_table_safe("CO_OCCURS", "Entity", "Entity", [
            ("count", "INT64"),
            ("sentence_ids", "STRING"),  # JSON array
        ])

        # Typed relation tables — created AFTER Entity node table above.
        # SemanticMemory._init_core_schema also attempts these, but may fail
        # when Entity doesn't yet exist.  Force recreation here by removing
        # stale cache entries so _create_rel_table_safe actually executes.
        for rel, src, dst in [
            ("PREFERS",     "Preference", "Entity"),
            ("USES",        "Fact",       "Entity"),
            ("KNOWS_ABOUT", "Fact",       "Entity"),
        ]:
            rel_key = f"{rel}_{src}_{dst}"
            # Remove from cache so the CREATE is always attempted here
            mem.rel_tables.discard(rel_key)
            mem._create_rel_table_safe(rel, src, dst, [
                ("confidence",   "DOUBLE"),
                ("source_text",  "STRING"),
                ("timestamp",    "DOUBLE"),
            ])

    def index_text(self, text: str, document_index: int = 0) -> ExtractionStats:
        """Extract entities and relations from a single text block.

        Returns ExtractionStats.
        """
        t0 = time.time()
        stats = ExtractionStats(documents=1, method=self.config.method)

        # 1. Split into sentences
        if self.config.method == "spacy":
            try:
                import spacy
                nlp = spacy.load(self.config.spacy_model)
                doc = nlp(text)
                raw_sents = [sent.text.strip() for sent in doc.sents
                             if len(sent.text.strip()) >= self.config.min_sentence_length]
            except ImportError:
                raw_sents = split_sentences(text, self.config)
        else:
            raw_sents = split_sentences(text, self.config)

        sentences = [
            ExtractedSentence(index=i, text=s, document_index=document_index)
            for i, s in enumerate(raw_sents)
        ]
        stats.sentences = len(sentences)

        if not sentences:
            stats.elapsed_s = time.time() - t0
            return stats

        # 2. Extract entities
        sent_texts = [s.text for s in sentences]
        if self.config.method == "spacy":
            entities = extract_entities_spacy(sent_texts, self.config)
        else:
            entities = extract_entities_tfidf(sent_texts, self.config)

        # Remove redundant n-gram substrings
        entities = _filter_subsumed_entities(entities)
        stats.entities = len(entities)

        # 3. Extract co-occurrences
        cooccurrences = extract_cooccurrences(entities, sentences, self.config)
        stats.relations = len(cooccurrences)

        # 4. Populate graph
        self._write_to_graph(entities, sentences, cooccurrences)

        # 5. Typed relation extraction (PREFERS / USES / KNOWS_ABOUT)
        if self.config.extract_typed:
            try:
                typed_rels = extract_typed_relations(
                    sent_texts, self.memory, self.config
                )
                stats.typed_relations = self._write_typed_relations(typed_rels)
            except Exception as e:
                logger.debug("Typed relation extraction failed: %s", e)

        stats.elapsed_s = time.time() - t0
        logger.info(
            "Extracted %d entities, %d relations from %d sentences (%.2fs, method=%s)",
            stats.entities, stats.relations, stats.sentences,
            stats.elapsed_s, stats.method,
        )
        return stats

    def index_documents(self, documents: List[str]) -> ExtractionStats:
        """Extract from multiple documents. Returns aggregate stats."""
        t0 = time.time()
        total = ExtractionStats(method=self.config.method)

        for doc_idx, doc_text in enumerate(documents):
            doc_stats = self.index_text(doc_text, document_index=doc_idx)
            total.entities += doc_stats.entities
            total.sentences += doc_stats.sentences
            total.relations += doc_stats.relations
            total.documents += 1

        total.elapsed_s = time.time() - t0
        return total

    def _write_to_graph(
        self,
        entities: List[ExtractedEntity],
        sentences: List[ExtractedSentence],
        cooccurrences: List[CooccurrenceEdge],
    ):
        """Write extracted data to Kuzu graph."""
        import json

        # Write entity nodes
        for entity in entities:
            try:
                self.memory.add_node("Entity", entity.id, {
                    "text": entity.text,
                    "normalized": entity.normalized,
                    "entity_type": entity.entity_type,
                    "score": entity.score,
                    "mention_count": len(entity.source_sentences),
                })
            except Exception as e:
                logger.debug("Entity write failed for %s: %s", entity.normalized, e)

        # Write sentence nodes
        if self.config.store_sentences:
            for sent in sentences:
                try:
                    self.memory.add_node("Sentence", sent.id, {
                        "text": sent.text,
                        "doc_index": sent.document_index,
                        "sent_index": sent.index,
                        "timestamp": time.time(),
                    })
                except Exception as e:
                    logger.debug("Sentence write failed for %s: %s", sent.id, e)

            # Write MENTIONS edges (sentence → entity)
            for entity in entities:
                for sent_idx in entity.source_sentences:
                    if sent_idx < len(sentences):
                        try:
                            self.memory.add_relationship(
                                "Sentence", sentences[sent_idx].id,
                                "Entity", entity.id,
                                "MENTIONS",
                                {"position": sent_idx},
                            )
                        except Exception as e:
                            logger.debug("MENTIONS edge failed: %s", e)

        # Write CO_OCCURS edges
        for edge in cooccurrences:
            try:
                self.memory.add_relationship(
                    "Entity", edge.entity_a_id,
                    "Entity", edge.entity_b_id,
                    "CO_OCCURS",
                    {
                        "count": edge.count,
                        "sentence_ids": json.dumps(edge.sentence_ids[:20]),  # Cap storage
                    },
                )
            except Exception as e:
                logger.debug("CO_OCCURS edge failed: %s", e)

    def _write_typed_relations(self, relations: List["ExtractedRelation"]) -> int:
        """Write typed PREFERS/USES/KNOWS_ABOUT edges to Kuzu.

        Ensures the target Entity node exists before creating the edge.
        Returns the count of edges successfully written.
        """
        written = 0
        for rel in relations:
            entity_ok = False
            subject_ok = False
            edge_ok = False
            entity_err = subject_err = edge_err = None

            # Step 1: Ensure target Entity node exists
            try:
                norm = _normalize_entity(rel.object_text)
                self.memory.add_node("Entity", rel.object_id, {
                    "text": rel.object_text,
                    "normalized": norm,
                    "entity_type": "TERM",
                    "score": rel.confidence,
                    "mention_count": 1,
                })
                entity_ok = True
            except Exception as e:
                entity_err = e
                logger.warning("Typed relation: Entity write failed for %r: %s", rel.object_text, e)
                continue

            # Step 2: Ensure subject node exists (for synthetic Fact subjects)
            if rel.subject_table == "Fact":
                try:
                    self.memory.add_node("Fact", rel.subject_id, {
                        "content": f"user_context:{rel.subject_id}",
                        "timestamp": time.time(),
                        "confidence": 0.5,
                        "source": "typed_extraction",
                        "metadata": "{}",
                    })
                    subject_ok = True
                except Exception as e:
                    subject_err = e
                    logger.warning(
                        "Typed relation: Fact subject write failed for %r: %s — skipping",
                        rel.subject_id, e,
                    )
                    continue
            else:
                subject_ok = True  # Preference nodes already exist

            # Step 3: Write the typed edge
            try:
                self.memory.add_relationship(
                    rel.subject_table, rel.subject_id,
                    "Entity", rel.object_id,
                    rel.relation_type,
                    {
                        "confidence": rel.confidence,
                        "source_text": rel.source_text[:200],
                        "timestamp": time.time(),
                    },
                )
                written += 1
                edge_ok = True
            except Exception as e:
                edge_err = e
                logger.warning(
                    "Typed relation: edge write failed (%s %s(%s) → Entity(%s)): %s",
                    rel.relation_type, rel.subject_table, rel.subject_id,
                    rel.object_id, e,
                )

            logger.debug(
                "Typed rel %s: entity_ok=%s subject_ok=%s edge_ok=%s | err: entity=%s subject=%s edge=%s",
                rel.relation_type, entity_ok, subject_ok, edge_ok,
                entity_err, subject_err, edge_err,
            )

        return written

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def find_related_entities(
        self,
        entity_text: str,
        max_hops: int = 2,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find entities related to a given entity via co-occurrence.

        Uses Kuzu's variable-length path queries for multi-hop traversal.
        """
        normalized = _normalize_entity(entity_text)
        eid = hashlib.sha256(normalized.encode()).hexdigest()[:16]

        cypher = f"""
            MATCH (a:Entity)-[:CO_OCCURS*1..{max_hops}]-(b:Entity)
            WHERE a.id = $eid AND a.id <> b.id
            RETURN DISTINCT b.text AS entity, b.entity_type AS type,
                   b.score AS score, b.mention_count AS mentions
            ORDER BY b.score DESC
            LIMIT {limit}
        """
        return self.memory.query(cypher, {"eid": eid})

    def get_entity_context(
        self,
        entity_text: str,
        max_sentences: int = 5,
    ) -> List[str]:
        """Get source sentences that mention an entity.

        This is what gets injected into the LLM prompt — the original
        text that the entity was extracted from, providing full context
        for the LLM to reason about relationships.
        """
        if not self.config.store_sentences:
            return []

        normalized = _normalize_entity(entity_text)
        eid = hashlib.sha256(normalized.encode()).hexdigest()[:16]

        cypher = f"""
            MATCH (s:Sentence)-[:MENTIONS]->(e:Entity)
            WHERE e.id = $eid
            RETURN s.text AS sentence
            ORDER BY s.timestamp DESC
            LIMIT {max_sentences}
        """
        results = self.memory.query(cypher, {"eid": eid})
        return [r["sentence"] for r in results if r.get("sentence")]

    def get_subgraph_context(
        self,
        query_terms: List[str],
        max_hops: int = 2,
        max_sentences: int = 10,
    ) -> str:
        """Build a text context block from the graph for LLM consumption.

        Given query terms, finds related entities via co-occurrence,
        then retrieves source sentences for those entities. Returns
        a formatted text block ready for prompt injection.

        This is the key method for integrating graph knowledge into
        the LLM prompt.
        """
        all_sentences: List[str] = []
        seen: Set[str] = set()

        for term in query_terms:
            # Get source sentences for the term itself
            for sent in self.get_entity_context(term, max_sentences=3):
                if sent not in seen:
                    all_sentences.append(sent)
                    seen.add(sent)

            # Get related entities and their contexts
            related = self.find_related_entities(term, max_hops=max_hops, limit=5)
            for r in related:
                for sent in self.get_entity_context(r["entity"], max_sentences=2):
                    if sent not in seen:
                        all_sentences.append(sent)
                        seen.add(sent)

            if len(all_sentences) >= max_sentences:
                break

        return "\n---\n".join(all_sentences[:max_sentences])

    def get_stats(self) -> Dict[str, Any]:
        """Get graph extraction statistics."""
        try:
            entity_count = self.memory.query(
                "MATCH (e:Entity) RETURN count(e) AS c"
            )
            edge_count = self.memory.query(
                "MATCH ()-[r:CO_OCCURS]->() RETURN count(r) AS c"
            )
            # Typed relation counts (may not exist in older graphs)
            typed_counts: Dict[str, int] = {}
            for rel in ("PREFERS", "USES", "KNOWS_ABOUT"):
                try:
                    rows = self.memory.query(
                        f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c"
                    )
                    typed_counts[rel.lower()] = rows[0]["c"] if rows else 0
                except Exception:
                    typed_counts[rel.lower()] = 0

            return {
                "entities": entity_count[0]["c"] if entity_count else 0,
                "co_occurs_edges": edge_count[0]["c"] if edge_count else 0,
                "typed_relations": typed_counts,
                "method": self.config.method,
            }
        except Exception:
            return {
                "entities": 0,
                "co_occurs_edges": 0,
                "typed_relations": {},
                "method": self.config.method,
            }

    def find_typed_relations(
        self,
        relation_type: str = "PREFERS",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Query all typed relations of a given type from the graph.

        Args:
            relation_type: One of PREFERS, USES, KNOWS_ABOUT.
            limit:         Max results.

        Returns:
            List of dicts with entity text, confidence, and source text.
        """
        try:
            if relation_type == "PREFERS":
                cypher = f"""
                    MATCH (p:Preference)-[r:PREFERS]->(e:Entity)
                    RETURN p.category AS category, p.value AS pref_value,
                           e.text AS entity, r.confidence AS confidence,
                           r.source_text AS source_text
                    ORDER BY r.confidence DESC
                    LIMIT {int(limit)}
                """
            else:
                cypher = f"""
                    MATCH (f:Fact)-[r:{relation_type}]->(e:Entity)
                    RETURN f.content AS fact, e.text AS entity,
                           r.confidence AS confidence,
                           r.source_text AS source_text
                    ORDER BY r.confidence DESC
                    LIMIT {int(limit)}
                """
            return self.memory.query(cypher) or []
        except Exception as e:
            logger.debug("find_typed_relations(%s) failed: %s", relation_type, e)
            return []
