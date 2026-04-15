"""Tier 2 Cognitive layer for Engram's three-tier extraction pipeline.

LLM-based relation extractor that runs asynchronously during the idle
window between assistant response and next user input.

Examines the last 3-5 turns with a CPU-side 8B model, refines and
supplements Tier 1 Reflex extractions with higher-confidence typed
relations, and writes them to the Kuzu semantic graph.

Runs in a daemon thread inside MemoryDaemon — never blocks the main
inference loop. Engine is configured via llm_engines.yaml under the
key 'tier2_cognitive'.

Relation types supported:
    IMPLEMENTED, FIXED, DECIDED, USES, DEPENDS_ON, NEXT_STEP,
    STRUGGLES_WITH, LEARNED, CORRECTED

Additional relation types can be added by extending VALID_RELATION_TYPES
and updating the prompt. Document new types in the project README so
that downstream consumers of the Kuzu graph know what to expect.

Author: Jeffrey Dean
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_FACT_LEN = 320
_MAX_TURNS = 5
_INFERENCE_TIMEOUT = 30.0  # seconds — drop if model takes too long

VALID_RELATION_TYPES = frozenset([
    "IMPLEMENTED",
    "FIXED",
    "DECIDED",
    "USES",
    "DEPENDS_ON",
    "NEXT_STEP",
    "STRUGGLES_WITH",
    "LEARNED",
    "CORRECTED",
])

_EXTRACTION_PROMPT = """\
Extract typed semantic relations from this conversation excerpt.

Relation types:
- IMPLEMENTED: something was built or added
- FIXED: a bug or problem was resolved
- DECIDED: a decision was made
- USES: a tool, library, or technology is being used
- DEPENDS_ON: one component requires another
- NEXT_STEP: a planned future action
- STRUGGLES_WITH: the user is having difficulty with something
- LEARNED: the user has understood or mastered something
- CORRECTED: the user was corrected on a misconception

Rules:
- Only extract relations that are clearly stated, not inferred
- subject: what the relation is about (keep concise, max 10 words)
- object: target of the relation for binary types like DEPENDS_ON (null otherwise)
- confidence: 0.0-1.0 reflecting how clearly stated the relation is
- Return ONLY a JSON array with keys: subject, object, relation, confidence. No preamble, no markdown fences.

Conversation:
{turns}

Relations (JSON array):"""


@dataclass
class CognitiveRelation:
    """A typed relation extracted by the Cognitive layer."""
    relation_type: str
    subject: str
    object: Optional[str]
    confidence: float = 0.75
    source: str = "cognitive_extraction"
    timestamp: float = field(default_factory=time.time)

    def to_fact_payload(self) -> Dict[str, Any]:
        content = f"[{self.relation_type}] {self.subject}"
        if self.object:
            content += f" → {self.object}"
        return {
            "content": content[:_MAX_FACT_LEN],
            "confidence": self.confidence,
            "source": self.source,
            "metadata": {
                "relation_type": self.relation_type,
                "subject": self.subject,
                "object": self.object,
            },
        }


class CognitiveExtractor:
    """Tier 2 LLM-based relation extractor.

    Runs inside the MemoryDaemon in a background thread after each
    assistant turn. Uses a CPU-side 8B model (configured in
    llm_engines.yaml as 'tier2_cognitive') to extract typed relations
    from the last 3-5 turns.

    No-op when:
    - semantic layer is None
    - tier2_cognitive engine is not configured in llm_engines.yaml
    - model path does not exist

    Args:
        semantic:       SemanticMemory instance (may be None).
        engine_config:  Dict from llm_engines.yaml 'tier2_cognitive' entry.
                        None = disabled.
        min_confidence: Relations below this threshold are discarded.
    """

    def __init__(
        self,
        semantic: Any,
        engine_config: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.60,
    ):
        self._semantic = semantic
        self._engine_config = engine_config
        self._min_confidence = min_confidence
        self._engine = None
        self._lock = threading.Lock()
        self._extractions = 0
        self._writes = 0
        self._skipped = 0
        self._errors = 0

        if engine_config is not None:
            self._engine = self._load_engine(engine_config)

    def _load_engine(self, config: Dict[str, Any]) -> Any:
        """Load the llama.cpp engine from config. Returns None on failure."""
        try:
            from pathlib import Path
            from ..engine.llama_cpp_engine import LlamaCppEngine

            model_path = Path(config["gguf_path"]).expanduser()
            if not model_path.exists():
                logger.warning(
                    "CognitiveExtractor: model not found at %s — disabled",
                    model_path,
                )
                return None

            engine = LlamaCppEngine(
                base_url=config.get("base_url", "http://127.0.0.1:8080/v1"),
                api_key=config.get("api_key", "dummy"),
                gguf_path=str(model_path),
                n_gpu_layers=int(config.get("n_gpu_layers", 0)),
                model_name=config.get("model", ""),
                max_context=int(config.get("context_size", 2048)),
            )
            logger.info(
                "CognitiveExtractor: loaded engine from %s", model_path
            )
            return engine
        except ImportError:
            logger.warning(
                "CognitiveExtractor: LlamaCppEngine not available — disabled"
            )
            return None
        except Exception as exc:
            logger.warning(
                "CognitiveExtractor: engine load failed: %s — disabled", exc
            )
            return None

    @property
    def enabled(self) -> bool:
        return self._semantic is not None and self._engine is not None

    def _format_turns(self, turns: List[Any]) -> str:
        """Format a list of Message objects into a prompt-ready string."""
        lines = []
        for msg in turns:
            role = getattr(msg, "role", "unknown").capitalize()
            content = getattr(msg, "content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def extract(self, turns: List[Any]) -> List[CognitiveRelation]:
        """Run LLM inference to extract typed relations from turns.

        Returns empty list on any failure — Tier 2 is best-effort.
        """
        if not self.enabled or not turns:
            return []

        turns_text = self._format_turns(turns[-_MAX_TURNS:])
        if not turns_text.strip():
            return []

        prompt = _EXTRACTION_PROMPT.format(turns=turns_text)

        try:
            response = self._engine.generate(
                prompt,
                max_tokens=self._engine_config.get("max_tokens", 512),
                temperature=self._engine_config.get("temperature", 0.1),
            )
        except Exception as exc:
            self._errors += 1
            logger.warning("CognitiveExtractor: inference failed: %s", exc)
            return []

        return self._parse_response(response)

    def _parse_response(self, response: str) -> List[CognitiveRelation]:
        """Parse JSON array from model response. Tolerant of minor formatting issues."""
        text = response.strip()

        # Strip markdown fences if model ignores instructions
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            ).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find a JSON array anywhere in the response
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    logger.debug(
                        "CognitiveExtractor: could not parse response: %s",
                        text[:200],
                    )
                    return []
            else:
                logger.debug(
                    "CognitiveExtractor: no JSON array found in response: %s",
                    text[:200],
                )
                return []

        if not isinstance(data, list):
            return []

        relations = []
        for item in data:
            if not isinstance(item, dict):
                continue
            relation_type = str(item.get("relation", item.get("type", ""))).upper().strip()
            if relation_type not in VALID_RELATION_TYPES:
                continue
            subject = str(item.get("subject", "")).strip()
            if not subject:
                continue
            obj = item.get("object")
            obj = str(obj).strip() if obj else None
            confidence = float(item.get("confidence", 0.75))
            if confidence < self._min_confidence:
                continue
            relations.append(CognitiveRelation(
                relation_type=relation_type,
                subject=subject[:160],
                object=obj[:80] if obj else None,
                confidence=confidence,
            ))

        self._extractions += len(relations)
        return relations

    def write_to_semantic(self, relations: List[CognitiveRelation]) -> int:
        """Write extracted relations to SemanticMemory as Fact nodes.

        Returns number of facts written.
        """
        if self._semantic is None or not relations:
            return 0

        written = 0
        seen: set = set()
        for rel in relations:
            key = (rel.relation_type, rel.subject[:40], rel.object)
            if key in seen:
                continue
            seen.add(key)
            try:
                payload = rel.to_fact_payload()
                if getattr(self._semantic, '_write_conn', None) is None:
                    logger.debug("Cognitive: semantic connection closed, skipping write")
                    break
                self._semantic.add_fact(**payload)
                written += 1
                logger.debug(
                    "Cognitive: wrote %s relation: %s",
                    rel.relation_type, rel.subject[:60],
                )
            except Exception as exc:
                logger.warning(
                    "Cognitive: failed to write relation: %s", exc
                )

        self._writes += written
        if written > 0:
            logger.info(
                "Cognitive: wrote %d relations from turn window", written
            )
        return written

    def process(self, turns: List[Any]) -> int:
        """Extract and write relations in one call. Returns facts written.

        Designed to be called in a daemon thread — all errors are caught
        and logged, never propagated.
        """
        if not self.enabled:
            self._skipped += 1
            return 0
        try:
            relations = self.extract(turns)
            if not relations:
                return 0
            return self.write_to_semantic(relations)
        except Exception as exc:
            self._errors += 1
            logger.warning("CognitiveExtractor.process failed: %s", exc)
            return 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "extractions": self._extractions,
            "writes": self._writes,
            "skipped": self._skipped,
            "errors": self._errors,
        }