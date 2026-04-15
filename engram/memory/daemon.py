"""Memory Daemon for Engram's async ingestion pipeline.

Single-writer background thread that consumes TurnEvents from the
EventBus and runs the ingestion pipeline without blocking add_turn().

Author: Jeffrey Dean
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

from .event_bus import EventBus, TurnEvent
from .reflex import ReflexExtractor
from .cognitive import CognitiveExtractor

if TYPE_CHECKING:
    from ..project_memory import ProjectMemory

logger = logging.getLogger(__name__)
_MAX_COGNITIVE_TURNS = 5


class MemoryDaemon:
    """Background thread that owns the memory write pipeline.

    Consumes TurnEvents from the EventBus and runs:
    - Episode ingestion (ChromaDB write)
    - Semantic extraction (Kuzu write) via existing ingestor

    The main thread never waits for these operations — it fires
    a TurnEvent and returns immediately.

    Args:
        project_memory: The ProjectMemory instance to write into.
        event_bus:      Shared EventBus to consume from.
    """

    def __init__(self, project_memory: "ProjectMemory", event_bus: EventBus):
        self._pm = project_memory
        self._bus = event_bus
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"engram-daemon-{project_memory.project_id}",
            daemon=True,
        )
        self._processed = 0
        self._errors = 0
        self._reflex = ReflexExtractor(
            semantic=getattr(project_memory, "semantic", None),
        )
        cognitive_config = self._load_cognitive_config()
        self._cognitive = CognitiveExtractor(
            semantic=getattr(project_memory, "semantic", None),
            engine_config=cognitive_config,
        )

    @staticmethod
    def _load_cognitive_config() -> Optional[dict]:
        """Load tier2_cognitive engine config from llm_engines.yaml if present."""
        try:
            from ..engine.config_loader import load_config
            config = load_config()
            return (config.get("engines") or {}).get("tier2_cognitive")
        except Exception as exc:
            logger.debug("CognitiveExtractor config not found: %s", exc)
            return None

    def start(self):
        """Start the daemon thread."""
        self._thread.start()
        logger.info("MemoryDaemon started for project=%s", self._pm.project_id)

    def stop(self, timeout: float = 5.0):
        """Signal the daemon to stop and wait for it to finish."""
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        logger.info(
            "MemoryDaemon stopped for project=%s (processed=%d errors=%d)",
            self._pm.project_id, self._processed, self._errors,
        )

    def _run(self):
        """Main daemon loop. Runs until stop() is called."""
        while not self._stop_event.is_set():
            event = self._bus.get(timeout=1.0)
            if event is None:
                continue
            try:
                self._process(event)
                self._processed += 1
            except Exception as exc:
                self._errors += 1
                logger.warning("MemoryDaemon: error processing event: %s", exc)
            finally:
                self._bus.task_done()

        # Drain remaining events on shutdown
        while True:
            event = self._bus.get(timeout=0.1)
            if event is None:
                break
            try:
                self._process(event)
                self._processed += 1
            except Exception as exc:
                self._errors += 1
                logger.warning("MemoryDaemon: drain error: %s", exc)
            finally:
                self._bus.task_done()

    def _process(self, event: TurnEvent):
        logger.info("daemon._process: role=%s chars=%d", event.role, len(event.text or ""))
        logger.info("daemon._process: reflex_enabled=%s cognitive_enabled=%s",
                    self._reflex._semantic is not None,
                    self._cognitive.enabled)
        """Run ingestion pipeline for one TurnEvent."""
        if event.session_id != self._pm.session_id:
            logger.debug(
                "MemoryDaemon: event session_id=%s differs from current=%s; using event session",
                event.session_id, self._pm.session_id,
            )

        # Tier 1: Reflex extraction — typed relations before ingestion
        if event.role == "assistant" and event.text:
            if self._reflex._semantic is None:
                self._reflex._semantic = getattr(self._pm, "semantic", None)
            if self._cognitive._semantic is None:
                self._cognitive._semantic = getattr(self._pm, "semantic", None)
            reflex_writes = self._reflex.process(event.text)
            if reflex_writes:
                logger.debug(
                    "MemoryDaemon: reflex extracted %d relations from assistant turn",
                    reflex_writes,
                )

        decision = self._pm.ingestor.process_turn(
            role=event.role,
            content=event.text,
            metadata=event.metadata,
            paired_text=event.paired_text,
        )
        outcome = self._pm.ingestor.apply(decision)

        # Tier 2: Cognitive extraction — async, after assistant turns only
        if event.role == "assistant":
            recent = self._pm.working.get_recent(_MAX_COGNITIVE_TURNS)
            threading.Thread(
                target=self._cognitive.process,
                args=(recent,),
                name=f"engram-cognitive-{self._pm.project_id}",
                daemon=True,
            ).start()
        logger.debug(
            "MemoryDaemon: processed role=%s episode_stored=%s semantic_writes=%d",
            event.role,
            bool(outcome.get("episode_id")),
            outcome.get("semantic_writes", 0),
        )

    def get_stats(self) -> dict:
        return {
            "running": self._thread.is_alive(),
            "processed": self._processed,
            "errors": self._errors,
            "reflex": self._reflex.get_stats(),
            "cognitive": self._cognitive.get_stats(),
            **self._bus.get_stats(),
        }