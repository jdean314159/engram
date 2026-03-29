from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LifecycleConfig:
    auto_promote_importance: float = 0.8
    promote_recent_limit: int = 20
    archive_after_days: int = 45
    archive_importance_max: float = 0.55
    delete_after_archive: bool = False


@dataclass
class LifecycleReport:
    promoted_events: int = 0
    promoted_facts: int = 0
    archived_episodes: int = 0
    deleted_episodes: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "promoted_events": self.promoted_events,
            "promoted_facts": self.promoted_facts,
            "archived_episodes": self.archived_episodes,
            "deleted_episodes": self.deleted_episodes,
            "details": list(self.details),
        }


class MemoryLifecycleManager:
    """Cross-layer lifecycle rules for Engram's explicit memories.

    This is intentionally conservative: it adds generic promotion + archival
    policies without requiring model-assisted extraction or brittle schemas.
    """

    def __init__(self, ctx, config: Optional[LifecycleConfig] = None):
        # ``project_memory`` alias preserved for body compatibility.
        self.project_memory = ctx
        self.config = config or LifecycleConfig()

    def promote_episode(
        self,
        *,
        episode_id: Optional[str],
        text: str,
        importance: float,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "episode_promotion",
    ) -> Dict[str, int]:
        if self.project_memory.semantic is None or not text.strip():
            return {"event": 0, "fact": 0}

        metadata = dict(metadata or {})
        digest = self._digest(text)
        event_meta = {**metadata, "episode_id": episode_id, "digest": digest}
        promoted = {"event": 0, "fact": 0}

        try:
            existing_events = self.project_memory.semantic.list_events(limit=500)
        except Exception:
            existing_events = []
        if not self._contains_digest(existing_events, digest):
            summary = self._summarize_text(text)
            self.project_memory.semantic.add_event(
                summary=summary,
                detail=text[:4000],
                importance=max(0.5, float(importance or 0.5)),
                source=source,
                metadata=event_meta,
                event_id=f"event_{digest[:16]}",
            )
            promoted["event"] = 1

        if self._looks_like_fact(text):
            try:
                existing_facts = self.project_memory.semantic.list_facts(limit=500)
            except Exception:
                existing_facts = []
            if not self._contains_digest(existing_facts, digest):
                self.project_memory.semantic.add_fact(
                    content=self._summarize_text(text, max_words=28),
                    confidence=min(0.95, max(0.55, float(importance or 0.5))),
                    source=source,
                    metadata=event_meta,
                    fact_id=f"fact_{digest[:16]}",
                )
                promoted["fact"] = 1

        return promoted

    def run_maintenance(self) -> LifecycleReport:
        report = LifecycleReport()
        episodic = getattr(self.project_memory, "episodic", None)
        if episodic is None:
            return report

        try:
            recent = episodic.get_recent_episodes(
                n=self.config.promote_recent_limit,
                days_back=max(7, self.config.archive_after_days),
                project_id=self.project_memory.project_id,
            )
        except Exception:
            recent = []

        now = time.time()
        to_archive: List[Dict[str, Any]] = []
        for ep in recent:
            importance = float(getattr(ep, "importance", 0.5) or 0.5)
            text = getattr(ep, "text", "") or ""
            metadata = dict(getattr(ep, "metadata", {}) or {})
            age_days = max(0.0, (now - float(getattr(ep, "timestamp", now) or now)) / 86400.0)

            if importance >= self.config.auto_promote_importance:
                result = self.promote_episode(
                    episode_id=getattr(ep, "id", None),
                    text=text,
                    importance=importance,
                    metadata=metadata,
                    source="maintenance",
                )
                report.promoted_events += result["event"]
                report.promoted_facts += result["fact"]
                if result["event"] or result["fact"]:
                    report.details.append({"episode_id": getattr(ep, "id", None), "action": "promote", **result})

            if age_days >= self.config.archive_after_days and importance <= self.config.archive_importance_max:
                to_archive.append({
                    "id": getattr(ep, "id", None),
                    "timestamp": getattr(ep, "timestamp", now),
                    "project_id": self.project_memory.project_id,
                    "session_id": getattr(ep, "session_id", None) or self.project_memory.session_id,
                    "text": text,
                    "metadata": {**metadata, "archived_from": "episodic", "importance": importance},
                })

        if to_archive and getattr(self.project_memory, "cold", None) is not None:
            report.archived_episodes = int(self.project_memory.cold.archive(to_archive) or 0)
            if self.config.delete_after_archive:
                for row in to_archive[:report.archived_episodes]:
                    ep_id = row.get("id")
                    if ep_id:
                        try:
                            episodic.delete_episode(ep_id)
                            report.deleted_episodes += 1
                        except Exception:
                            pass

        if getattr(self.project_memory, "telemetry", None) is not None:
            try:
                self.project_memory.telemetry.emit(
                    "memory_lifecycle",
                    "memory lifecycle maintenance completed",
                    project_id=self.project_memory.project_id,
                    session_id=self.project_memory.session_id,
                    **report.to_dict(),
                )
            except Exception:
                pass

        return report

    @staticmethod
    def _summarize_text(text: str, max_words: int = 36) -> str:
        words = (text or "").split()
        if len(words) <= max_words:
            return " ".join(words).strip()
        return " ".join(words[:max_words]).strip() + " ..."

    @staticmethod
    def _looks_like_fact(text: str) -> bool:
        lowered = (text or "").lower()
        return any(phrase in lowered for phrase in ("i use ", "i prefer ", "we decided", "the plan is", "remember that"))

    @staticmethod
    def _digest(text: str) -> str:
        normalized = " ".join((text or "").split()).strip().lower()
        return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _contains_digest(rows: List[Dict[str, Any]], digest: str) -> bool:
        for row in rows:
            raw = row.get("metadata")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}
            if isinstance(raw, dict) and raw.get("digest") == digest:
                return True
        return False
