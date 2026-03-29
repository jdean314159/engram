"""Simple profiling helpers for conversational memory operations."""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _measure(fn, *args, **kwargs) -> Tuple[float, Any]:
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, result


def profile_project_memory(
    project_memory: Any,
    turns: Sequence[Tuple[str, str]],
    queries: Iterable[str],
) -> Dict[str, Any]:
    add_turn_ms: List[float] = []
    get_context_ms: List[float] = []

    for role, content in turns:
        elapsed_ms, _ = _measure(project_memory.add_turn, role, content)
        add_turn_ms.append(elapsed_ms)

    last_context = None
    for query in queries:
        elapsed_ms, last_context = _measure(project_memory.get_context, query=query, max_tokens=project_memory.budget.total)
        get_context_ms.append(elapsed_ms)

    return {
        "add_turn": {
            "count": len(add_turn_ms),
            "avg_ms": round(statistics.fmean(add_turn_ms), 3) if add_turn_ms else 0.0,
            "p95_ms": round(sorted(add_turn_ms)[max(0, int(len(add_turn_ms) * 0.95) - 1)], 3) if add_turn_ms else 0.0,
        },
        "get_context": {
            "count": len(get_context_ms),
            "avg_ms": round(statistics.fmean(get_context_ms), 3) if get_context_ms else 0.0,
            "p95_ms": round(sorted(get_context_ms)[max(0, int(len(get_context_ms) * 0.95) - 1)], 3) if get_context_ms else 0.0,
        },
        "diagnostics": project_memory.get_diagnostics_snapshot() if hasattr(project_memory, "get_diagnostics_snapshot") else {},
        "last_context": last_context.to_dict() if hasattr(last_context, "to_dict") else None,
    }
