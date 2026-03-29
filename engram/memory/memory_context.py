"""MemoryContext — minimal dependency surface for memory orchestrators.

Replaces the wide ``project_memory`` reference that
``UnifiedRetriever``, ``MemoryLifecycleManager``, and ``MemoryIngestor``
previously held.  Each orchestrator now receives only what it actually uses.

Design notes
------------
* ``session_id`` is mutable (changes on ``new_session()``), so it is stored
  as a single-element list used as a mutable cell.  ``MemoryContext.session_id``
  is a property that reads from the cell.  ``ProjectMemory.new_session()``
  updates ``_session_cell[0]`` so all contexts see the change automatically.
* ``search_episodes`` and ``store_episode`` are callables injected by
  ``ProjectMemory``.  They carry the forgetting-policy and surprise-filter
  side-effects that make those operations more than simple layer calls.
* There is no back-reference to ``ProjectMemory``; orchestrators cannot
  escape this boundary.

Author: Jeffrey Dean
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class MemoryContext:
    """Explicit dependency bundle for memory orchestrators.

    Attributes
    ----------
    working:          Layer 1 — WorkingMemory instance.
    episodic:         Layer 2 — EpisodicMemory or None.
    semantic:         Layer 3 — SemanticMemory or None.
    cold:             Layer 4 — ColdStorage instance.
    neural_coord:     NeuralCoordinator or None.
    embedding_service: EmbeddingService.
    budget:           Token allocation across layers.
    token_counter:    Callable ``(str) -> int``.
    project_id:       Stable project identifier string.
    telemetry:        Telemetry sink for structured events.
    search_episodes:  Bound method from ProjectMemory that includes forgetting
                      policy access tracking.
    store_episode:    Bound method from ProjectMemory that includes surprise
                      filter gating.
    _session_cell:    Mutable cell ``[session_id]``; updated by new_session().
    """

    working: Any
    episodic: Optional[Any]
    semantic: Optional[Any]
    cold: Any
    neural_coord: Optional[Any]
    embedding_service: Any
    budget: Any
    token_counter: Callable
    project_id: str
    project_type: Any
    telemetry: Any
    search_episodes: Callable
    store_episode: Callable
    _session_cell: List[str] = field(default_factory=lambda: ["default"])

    @property
    def session_id(self) -> str:
        return self._session_cell[0]

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_cell[0] = value

    @property
    def _token_counter(self):
        """Alias so retrieval body references project_memory._token_counter work."""
        return self.token_counter
