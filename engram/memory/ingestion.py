from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Pattern, Sequence

logger = logging.getLogger(__name__)


@dataclass
class SemanticMemoryWrite:
    kind: str
    payload: Dict[str, Any]


@dataclass
class IngestionDecision:
    should_store_episode: bool = False
    episode_text: Optional[str] = None
    episode_importance: float = 0.0
    semantic_writes: List[SemanticMemoryWrite] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    source_text: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_store_episode": self.should_store_episode,
            "episode_text": self.episode_text,
            "episode_importance": round(self.episode_importance, 3),
            "semantic_writes": [
                {"kind": item.kind, "payload": dict(item.payload)} for item in self.semantic_writes
            ],
            "reasons": list(self.reasons),
            "source_text": self.source_text,
            "timestamp": self.timestamp,
        }


@dataclass
class IngestionPolicy:
    """Configurable write-policy surface for conversational memory formation."""

    episode_threshold: float = 0.35
    max_episode_chars: int = 1200
    dedup_search_n: int = 3
    ephemeral_patterns: Sequence[Pattern[str]] = field(default_factory=tuple)
    preference_patterns: Sequence[Pattern[str]] = field(default_factory=tuple)
    fact_patterns: Sequence[Pattern[str]] = field(default_factory=tuple)
    user_profile_terms: Sequence[str] = field(default_factory=tuple)
    task_state_terms: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def for_project_type(cls, project_type: Any) -> "IngestionPolicy":
        pt = str(getattr(project_type, "value", project_type) or "").lower()

        base = cls(
            episode_threshold=0.35,
            max_episode_chars=1200,
            dedup_search_n=3,
            ephemeral_patterns=(
                re.compile(r"\bignore this\b", re.IGNORECASE),
                re.compile(r"\bdo not remember\b", re.IGNORECASE),
                re.compile(r"\bfor this message only\b", re.IGNORECASE),
                re.compile(r"\btemporary note\b", re.IGNORECASE),
                re.compile(r"\bnot for long[- ]term memory\b", re.IGNORECASE),
                re.compile(r"\bephemeral\b", re.IGNORECASE),
                re.compile(r"\bmensaje temporal\b", re.IGNORECASE),
                re.compile(r"\bno recuerdes esto\b", re.IGNORECASE),
            ),
            preference_patterns=(
                re.compile(r"\bI prefer\s+(?P<value>.+)", re.IGNORECASE),
                re.compile(r"\bI like\s+(?P<value>.+)", re.IGNORECASE),
                re.compile(r"\bI (?:do not|don't) like\s+(?P<value>.+)", re.IGNORECASE),
                re.compile(r"\bmy favorite\s+(?P<category>\w+)\s+is\s+(?P<value>.+)", re.IGNORECASE),
                re.compile(r"\bprefiero\s+(?P<value>.+)", re.IGNORECASE),
                re.compile(r"\bme gusta\s+(?P<value>.+)", re.IGNORECASE),
            ),
            fact_patterns=(
                re.compile(r"\bI am\s+(?P<value>.+)", re.IGNORECASE),
                re.compile(r"\bI work on\s+(?P<value>.+)", re.IGNORECASE),
                re.compile(r"\bI use\s+(?P<value>.+)", re.IGNORECASE),
                re.compile(r"\bwe decided to\s+(?P<value>.+)", re.IGNORECASE),
                re.compile(r"\bthe plan is to\s+(?P<value>.+)", re.IGNORECASE),
                re.compile(r"\bestoy\s+(?P<value>.+)", re.IGNORECASE),
                re.compile(r"\buso\s+(?P<value>.+)", re.IGNORECASE),
                re.compile(r"\bdecidimos\s+(?P<value>.+)", re.IGNORECASE),
            ),
            user_profile_terms=(
                "prefer", "favorite", "i like", "i don't like", "i am", "i use", "i work on",
                "prefiero", "me gusta", "uso", "trabajo en",
            ),
            task_state_terms=(
                "decided", "plan", "implemented", "fixed", "bug", "issue", "todo", "next step",
                "resolved", "decision", "milestone", "deadline", "decidimos", "plan", "error",
            ),
        )

        if "language_tutor" in pt:
            return cls(
                episode_threshold=0.56,
                max_episode_chars=1200,
                dedup_search_n=3,
                ephemeral_patterns=base.ephemeral_patterns,
                preference_patterns=base.preference_patterns,
                fact_patterns=base.fact_patterns,
                user_profile_terms=tuple(list(base.user_profile_terms) + ["studying", "learning", "practicing", "aprendo", "practico"]),
                task_state_terms=tuple(list(base.task_state_terms) + ["mistake", "grammar", "vocabulary", "pronunciation", "gramática", "vocabulario"]),
            )

        if "programming" in pt:
            return cls(
                episode_threshold=0.6,
                max_episode_chars=1400,
                dedup_search_n=4,
                ephemeral_patterns=base.ephemeral_patterns,
                preference_patterns=base.preference_patterns,
                fact_patterns=base.fact_patterns,
                user_profile_terms=tuple(list(base.user_profile_terms) + ["stack", "framework", "language", "tooling"]),
                task_state_terms=tuple(list(base.task_state_terms) + ["stacktrace", "incident", "deploy", "regression", "refactor", "asyncio"]),
            )

        return base


class MemoryIngestor:
    """Heuristic memory formation pipeline with configurable policy packs."""

    def __init__(
        self,
        ctx,
        policy: Optional[IngestionPolicy] = None,
        episode_threshold: Optional[float] = None,
        max_episode_chars: Optional[int] = None,
    ):
        # ``project_memory`` alias preserved for body compatibility.
        self.project_memory = ctx
        self.policy = policy or IngestionPolicy.for_project_type(getattr(ctx, "project_type", None))
        if episode_threshold is not None:
            self.policy.episode_threshold = episode_threshold
        if max_episode_chars is not None:
            self.policy.max_episode_chars = max_episode_chars

    def process_turn(
        self,
        *,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        paired_text: Optional[str] = None,
    ) -> IngestionDecision:
        metadata = metadata or {}
        text = (content or "").strip()
        decision = IngestionDecision(source_text=text)
        if not text:
            decision.reasons.append("empty_content")
            return decision

        if self._is_ephemeral(text, metadata):
            decision.reasons.append("ephemeral_signal")
            return decision

        signals = self._score_text(text=text, role=role, metadata=metadata)
        score = signals["score"]
        decision.reasons.extend(signals["reasons"])

        semantic_writes = self._dedupe_semantic_writes(
            self._extract_semantic_writes(text, role=role)
        )
        decision.semantic_writes.extend(semantic_writes)
        if semantic_writes:
            decision.reasons.append("semantic_candidates")
            score += min(0.18, 0.06 * len(semantic_writes))

        logger.info("ingestion: role=%s score=%.2f threshold=%.2f will_store=%s",
                    role, score, self.policy.episode_threshold,
                    score >= self.policy.episode_threshold)
        episode_text = paired_text if paired_text else text
        episode_text = episode_text[: self.policy.max_episode_chars].strip()
        decision.episode_text = episode_text
        decision.episode_importance = max(0.0, min(1.0, score))
        decision.should_store_episode = score >= self.policy.episode_threshold and len(episode_text) >= 24

        if decision.should_store_episode and self.project_memory.episodic is not None:
            recent = self.project_memory.search_episodes(episode_text[:200], n=self.policy.dedup_search_n)
            normalized = self._normalize(episode_text)
            for ep in recent:
                if self._normalize(getattr(ep, "text", "")) == normalized:
                    decision.should_store_episode = False
                    decision.reasons.append("duplicate_episode")
                    break

        return decision

    def apply(self, decision: IngestionDecision) -> Dict[str, Any]:
        outcome: Dict[str, Any] = {
            "episode_id": None,
            "semantic_writes": 0,
            "decision": decision.to_dict(),
        }

        if decision.should_store_episode and decision.episode_text:
            try:
                episode_id = self.project_memory.store_episode(
                    decision.episode_text,
                    metadata={
                        "ingested": True,
                        "reasons": decision.reasons,
                        "source_text": decision.source_text[:400],
                    },
                    importance=decision.episode_importance,
                    bypass_filter=True,
                )
                outcome["episode_id"] = episode_id
            except Exception as exc:
                logger.warning("Episode ingestion failed: %s", exc)

        if self.project_memory.semantic is not None:
            for write in decision.semantic_writes:
                try:
                    if write.kind == "preference":
                        if not self._semantic_preference_exists(write.payload):
                            node = self.project_memory.semantic.add_preference(**write.payload)
                            outcome["semantic_writes"] += 1
                            # Also extract typed graph edges from the source text
                            self._index_semantic_write_to_graph(
                                write, node, decision.source_text
                            )
                    elif write.kind == "fact":
                        if not self._semantic_fact_exists(write.payload):
                            node = self.project_memory.semantic.add_fact(**write.payload)
                            outcome["semantic_writes"] += 1
                            self._index_semantic_write_to_graph(
                                write, node, decision.source_text
                            )
                except Exception as exc:
                    logger.debug("Semantic ingestion failed: %s", exc)

        return outcome

    def _is_ephemeral(self, text: str, metadata: Dict[str, Any]) -> bool:
        if metadata.get("ephemeral"):
            return True
        lowered = text.lower()
        if "test" in lowered and "message" in lowered and len(lowered.split()) <= 8:
            return True
        return any(pattern.search(text) for pattern in self.policy.ephemeral_patterns)

    def _score_text(self, *, text: str, role: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        score = 0.05
        reasons: List[str] = []
        lower = text.lower()

        if metadata.get("remember") or metadata.get("importance", 0) >= 0.8:
            score += 0.55
            reasons.append("explicit_memory_signal")

        if "remember" in lower or "important" in lower or "recuerda" in lower:
            score += 0.28
            reasons.append("memory_language")

        if any(token in lower for token in self.policy.user_profile_terms):
            score += 0.22
            reasons.append("user_profile_signal")

        if any(token in lower for token in self.policy.task_state_terms):
            score += 0.16
            reasons.append("task_state_signal")

        if role == "assistant":
            score += 0.20
            reasons.append("assistant_turn_summary")

        words = len(text.split())
        score += min(0.12, words / 200.0)
        if words >= 20:
            reasons.append("substantive_turn")

        return {"score": min(1.0, score), "reasons": reasons}

    _LLM_BOILERPLATE = re.compile(
        r"^I (?:can |will |would |could |am happy to|am ready|understand|noticed|"
        r"have analyzed|see that|think|believe|suggest|recommend)",
        re.IGNORECASE,
    )

    def _extract_semantic_writes(self, text: str, role: str = "user") -> List[SemanticMemoryWrite]:
        # Assistant turns: no fact extraction — LLM first-person statements
        # are not facts about the user or project.
        # Preference extraction also skipped: only meaningful from user speech.
        if role == "assistant":
            return []
        writes: List[SemanticMemoryWrite] = []

        for pattern in self.policy.preference_patterns:
            match = pattern.search(text)
            if not match:
                continue
            gd = {k: (v.strip(" .,!\n\t") if isinstance(v, str) else v) for k, v in match.groupdict().items()}
            category = gd.get("category") or self._infer_preference_category(gd.get("value") or "")
            value = gd.get("value")
            if value:
                strength = 0.68 if any(tok in text.lower() for tok in ("don't like", "do not like", "no me gusta")) else 0.75
                writes.append(
                    SemanticMemoryWrite(
                        kind="preference",
                        payload={
                            "category": category[:80],
                            "value": value[:240],
                            "strength": strength,
                            "source": "conversation",
                        },
                    )
                )
                break

        for pattern in self.policy.fact_patterns:
            match = pattern.search(text)
            if not match:
                continue
            value = match.groupdict().get("value", "").strip(" .,!\n\t")
            if value:
                writes.append(
                    SemanticMemoryWrite(
                        kind="fact",
                        payload={
                            "content": value[:320],
                            "confidence": 0.72,
                            "source": "conversation",
                            "metadata": {"surface_text": text[:400]},
                        },
                    )
                )
                break
        return writes

    def _dedupe_semantic_writes(self, writes: List[SemanticMemoryWrite]) -> List[SemanticMemoryWrite]:
        seen = set()
        out: List[SemanticMemoryWrite] = []
        for write in writes:
            if write.kind == "preference":
                key = (write.kind, self._normalize(write.payload.get("category", "")), self._normalize(write.payload.get("value", "")))
            else:
                key = (write.kind, self._normalize(write.payload.get("content", "")))
            if key in seen:
                continue
            seen.add(key)
            out.append(write)
        return out

    def _index_semantic_write_to_graph(
        self,
        write: "SemanticMemoryWrite",
        node,
        source_text: str,
    ) -> None:
        """Feed a confirmed semantic write into the GraphExtractor typed relation pass.

        Called after a Preference or Fact node has been successfully written to
        Kuzu. Runs extract_typed_relations on the source sentence so that PREFERS /
        USES / KNOWS_ABOUT edges connect the new node to Entity nodes.

        Silently no-ops when the extractor is not available or extraction fails.
        """
        extractor = getattr(self.project_memory, "extractor", None)
        if extractor is None or not source_text:
            return
        try:
            from .extraction import extract_typed_relations
            sentences = [source_text[:500]]
            typed_rels = extract_typed_relations(
                sentences,
                self.project_memory.semantic,
                extractor.config,
            )
            if typed_rels:
                extractor._write_typed_relations(typed_rels)
        except Exception as e:
            logger.debug("_index_semantic_write_to_graph failed: %s", e)

    def _semantic_preference_exists(self, payload: Dict[str, Any]) -> bool:
        semantic = getattr(self.project_memory, "semantic", None)
        if semantic is None:
            return False
        category = self._normalize(payload.get("category", ""))
        value = self._normalize(payload.get("value", ""))
        rows = None
        if hasattr(semantic, "preferences") and isinstance(getattr(semantic, "preferences"), list):
            rows = getattr(semantic, "preferences")
        elif hasattr(semantic, "list_preferences"):
            try:
                rows = semantic.list_preferences(limit=200)
            except Exception:
                rows = None
        if not rows:
            return False
        for pref in rows:
            if self._normalize(pref.get("category", "")) == category and self._normalize(pref.get("value", "")) == value:
                return True
        return False

    def _semantic_fact_exists(self, payload: Dict[str, Any]) -> bool:
        semantic = getattr(self.project_memory, "semantic", None)
        if semantic is None:
            return False
        content = self._normalize(payload.get("content", ""))
        rows = None
        if hasattr(semantic, "facts") and isinstance(getattr(semantic, "facts"), list):
            rows = getattr(semantic, "facts")
        elif hasattr(semantic, "list_facts"):
            try:
                rows = semantic.list_facts(limit=200)
            except Exception:
                rows = None
        if not rows:
            return False
        for fact in rows:
            if self._normalize(fact.get("content", "")) == content:
                return True
            if self._text_similarity(fact.get("content", ""), payload.get("content", "")) >= 0.9:
                return True
        return False

    @staticmethod
    def _infer_preference_category(value: str) -> str:
        lowered = (value or "").lower()
        if any(tok in lowered for tok in ("python", "haskell", "java", "javascript", "rust", "go")):
            return "language"
        if any(tok in lowered for tok in ("vim", "emacs", "vscode", "cursor", "zed")):
            return "tooling"
        return "general"

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        ta = set(re.findall(r"[A-Za-z0-9_./-]+", (a or "").lower()))
        tb = set(re.findall(r"[A-Za-z0-9_./-]+", (b or "").lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(1, len(ta | tb))

    @staticmethod
    def stable_id(prefix: str, text: str) -> str:
        return f"{prefix}_{hashlib.sha256(text.encode()).hexdigest()[:16]}"
