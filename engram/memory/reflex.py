"""Tier 1 Reflex layer for Engram's three-tier extraction pipeline.

Zero-cost regex gate that detects high-signal turns and extracts
typed relations for immediate Kuzu graph updates.

Runs synchronously inside the MemoryDaemon before process_turn(),
adding structured semantic writes for relations the ingestion
pipeline's generic patterns would miss.

Author: Jeffrey Dean
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
_MAX_FACT_LEN = 320


# ---------------------------------------------------------------------------
# Signal patterns — ordered by specificity, most specific first
# ---------------------------------------------------------------------------

_IMPLEMENTED = re.compile(
    r"\b(?:we\s+)?(?:implemented|added|built|created|wrote)\s+(?P<value>.+?)(?:\.|$)",
    re.IGNORECASE,
)
_FIXED = re.compile(
    r"\b(?:we\s+)?(?:fixed|resolved|patched|corrected)\s+(?:the\s+)?(?P<value>.+?)(?:\.|$)",
    re.IGNORECASE,
)
_DEPENDS = re.compile(
    r"\b(?P<from>\w[\w\s]{1,40}?)\s+(?:requires|depends on|needs)\s+(?P<to>\w[\w\s]{1,40}?)(?:\.|$)",
    re.IGNORECASE,
)
_DECIDED = re.compile(
    r"\b(?:we\s+)?decided\s+(?:to\s+)?(?P<value>.+?)(?:\.|$)",
    re.IGNORECASE,
)
_USES_TOOL = re.compile(
    r"\b(?:we\s+(?:use|are\s+using|switched\s+to)|switched\s+to)\s+(?P<value>[\w][\w\s\.\-]{1,60}?)(?:\s+for|\s+to|\.|$)",
    re.IGNORECASE,
)
_NEXT_STEP = re.compile(
    r"\b(?:next\s+step|next\s+we|todo)\s+(?:is\s+)?(?:to\s+)?(?P<value>.+?)(?:\.|$)",
    re.IGNORECASE,
)

# Minimum signal word count to bother extracting
_MIN_WORDS = 8

# High-signal keywords that trigger reflex extraction attempt
_TRIGGER_TOKENS = frozenset([
    "implemented", "added", "built", "fixed", "resolved", "patched",
    "decided", "requires", "depends", "using", "switched", "next",
    "todo", "phase", "corrected", "created", "wrote",
])


@dataclass
class ReflexRelation:
    """A typed relation extracted by the Reflex layer."""
    relation_type: str          # IMPLEMENTED, FIXED, DECIDED, USES, DEPENDS_ON, NEXT_STEP
    subject: str                # what the relation is about
    object: Optional[str]       # target (for binary relations like DEPENDS_ON)
    confidence: float = 0.65
    source_text: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_fact_payload(self) -> Dict[str, Any]:
        """Convert to a fact payload for SemanticMemory.add_fact()."""
        content = f"[{self.relation_type}] {self.subject}"
        if self.object:
            content += f" → {self.object}"
        return {
            "content": content[:_MAX_FACT_LEN],
            "confidence": self.confidence,
            "source": "reflex_extraction",
            "metadata": {
                "relation_type": self.relation_type,
                "subject": self.subject,
                "object": self.object,
                "surface_text": self.source_text[:200],
            },
        }


class ReflexExtractor:
    """Tier 1 regex-based relation extractor.

    Runs inside the MemoryDaemon before process_turn(). Zero LLM calls.
    Extracts typed relations from high-signal turns and writes them
    directly to SemanticMemory as Fact nodes.

    Args:
        semantic: SemanticMemory instance (may be None — no-op if so).
        min_confidence: Relations below this threshold are discarded.
    """

    def __init__(self, semantic: Any, min_confidence: float = 0.55):
        self._semantic = semantic
        self._min_confidence = min_confidence
        self._extractions = 0
        self._writes = 0
        self._skipped = 0

    def should_trigger(self, text: str) -> bool:
        """Fast check: does this text contain any high-signal tokens?"""
        if len(text.split()) < _MIN_WORDS:
            return False
        lower = text.lower()
        return any(tok in lower for tok in _TRIGGER_TOKENS)

    def extract(self, text: str) -> List[ReflexRelation]:
        """Extract typed relations from text. Returns empty list on no signal."""
        if not self.should_trigger(text):
            return []

        relations: List[ReflexRelation] = []

        # IMPLEMENTED / BUILT / ADDED
        for m in _IMPLEMENTED.finditer(text):
            val = m.group("value").strip()
            if len(val.split()) <= 15:  # ignore runaway matches
                relations.append(ReflexRelation(
                    relation_type="IMPLEMENTED",
                    subject=val[:160],
                    object=None,
                    confidence=0.70,
                    source_text=text[:200],
                ))

        # FIXED / RESOLVED
        for m in _FIXED.finditer(text):
            val = m.group("value").strip()
            if len(val.split()) <= 15:
                relations.append(ReflexRelation(
                    relation_type="FIXED",
                    subject=val[:160],
                    object=None,
                    confidence=0.72,
                    source_text=text[:200],
                ))

        # DEPENDS_ON (binary)
        for m in _DEPENDS.finditer(text):
            frm = m.group("from").strip()
            to = m.group("to").strip()
            if frm and to and len(frm.split()) <= 8 and len(to.split()) <= 8:
                relations.append(ReflexRelation(
                    relation_type="DEPENDS_ON",
                    subject=frm[:80],
                    object=to[:80],
                    confidence=0.65,
                    source_text=text[:200],
                ))

        # DECIDED
        for m in _DECIDED.finditer(text):
            val = m.group("value").strip()
            if len(val.split()) <= 20:
                relations.append(ReflexRelation(
                    relation_type="DECIDED",
                    subject=val[:160],
                    object=None,
                    confidence=0.68,
                    source_text=text[:200],
                ))

        # USES (tool/technology)
        for m in _USES_TOOL.finditer(text):
            val = m.group("value").strip()
            if len(val.split()) <= 8:
                relations.append(ReflexRelation(
                    relation_type="USES",
                    subject=val[:120],
                    object=None,
                    confidence=0.65,
                    source_text=text[:200],
                ))

        # NEXT_STEP / TODO
        for m in _NEXT_STEP.finditer(text):
            val = m.group("value").strip()
            if len(val.split()) <= 20:
                relations.append(ReflexRelation(
                    relation_type="NEXT_STEP",
                    subject=val[:160],
                    object=None,
                    confidence=0.60,
                    source_text=text[:200],
                ))

        # Filter by confidence
        relations = [r for r in relations if r.confidence >= self._min_confidence]
        self._extractions += len(relations)
        return relations

    def write_to_semantic(self, relations: List[ReflexRelation]) -> int:
        """Write extracted relations to SemanticMemory as Fact nodes.

        Returns number of facts written.
        """
        if self._semantic is None or not relations:
            return 0

        written = 0
        seen = set()
        for rel in relations:
            key = (rel.relation_type, rel.subject[:40], rel.object)
            if key in seen:
                continue
            seen.add(key)
            try:
                payload = rel.to_fact_payload()
                self._semantic.add_fact(**payload)
                written += 1
                logger.debug(
                    "Reflex: wrote %s relation: %s",
                    rel.relation_type, rel.subject[:60],
                )
            except Exception as exc:
                logger.warning("Reflex: failed to write relation: %s", exc)

        self._writes += written
        if written > 0:
            logger.info("Reflex: wrote %d relations from turn", written)
        return written

    def process(self, text: str) -> int:
        """Extract and write relations in one call. Returns facts written."""
        if self._semantic is None:
            self._skipped += 1
            return 0
        relations = self.extract(text)
        if not relations:
            return 0
        return self.write_to_semantic(relations)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "extractions": self._extractions,
            "writes": self._writes,
            "skipped": self._skipped,
        }