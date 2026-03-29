"""Retrieval evaluation fixtures and scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence


@dataclass
class RetrievalFixture:
    name: str
    query: str
    expected_substrings: Sequence[str]
    forbidden_substrings: Sequence[str] = ()
    min_recall: float = 0.6
    notes: str = ""


DEFAULT_FIXTURES: List[RetrievalFixture] = [
    RetrievalFixture(
        name="asyncio_bug_recall",
        query="python asyncio bug",
        expected_substrings=("asyncio", "python", "error"),
        forbidden_substrings=("gardening",),
        min_recall=0.66,
        notes="Programming incident recall should surface explicit error context.",
    ),
    RetrievalFixture(
        name="preference_recall",
        query="preferred language",
        expected_substrings=("preference", "python"),
        min_recall=1.0,
        notes="User preferences should be easy to recall.",
    ),
    RetrievalFixture(
        name="task_state_recall",
        query="what did we decide for the retry loop",
        expected_substrings=("decided", "retry"),
        notes="Recent decisions should survive into semantic or episodic memory.",
    ),
    RetrievalFixture(
        name="graph_context_recall",
        query="worker failure retry",
        expected_substrings=("worker", "retry"),
        notes="Extraction-graph context should contribute supporting sentences.",
    ),
    RetrievalFixture(
        name="cold_archive_recall",
        query="archived python incidents",
        expected_substrings=("archived", "python"),
        notes="Cold storage should still help when explicit memory is sparse.",
    ),
]


def flatten_context_text(context: Any) -> str:
    parts: List[str] = []
    for msg in getattr(context, "working", []) or []:
        parts.append(getattr(msg, "content", ""))
    for ep in getattr(context, "episodic", []) or []:
        parts.append(getattr(ep, "text", ""))
    for row in getattr(context, "semantic", []) or []:
        if isinstance(row, dict):
            parts.append(str(row.get("content") or row.get("summary") or row.get("value") or row.get("text") or row.get("detail") or ""))
    for row in getattr(context, "cold", []) or []:
        if isinstance(row, dict):
            parts.append(str(row.get("text") or row.get("content") or ""))
    return "\n".join(part for part in parts if part).lower()


def evaluate_context(context: Any, fixture: RetrievalFixture) -> Dict[str, Any]:
    text = flatten_context_text(context)
    expected_hits = [needle for needle in fixture.expected_substrings if needle.lower() in text]
    forbidden_hits = [needle for needle in fixture.forbidden_substrings if needle.lower() in text]
    recall = len(expected_hits) / max(1, len(fixture.expected_substrings))
    precision_penalty = len(forbidden_hits) / max(1, len(fixture.forbidden_substrings)) if fixture.forbidden_substrings else 0.0
    score = max(0.0, recall - precision_penalty)
    passed = recall >= fixture.min_recall and not forbidden_hits
    return {
        "fixture": fixture.name,
        "query": fixture.query,
        "recall": round(recall, 3),
        "score": round(score, 3),
        "forbidden_hits": forbidden_hits,
        "expected_hits": expected_hits,
        "expected_total": len(fixture.expected_substrings),
        "min_recall": fixture.min_recall,
        "passed": passed,
        "notes": fixture.notes,
    }


def summarize_fixture_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"ok": True, "avg_recall": 0.0, "pass_rate": 1.0, "failed": []}
    avg_recall = sum(float(item.get("recall", 0.0) or 0.0) for item in results) / len(results)
    pass_rate = sum(1 for item in results if item.get("passed")) / len(results)
    failed = [item["fixture"] for item in results if not item.get("passed")]
    return {
        "ok": not failed,
        "avg_recall": round(avg_recall, 3),
        "pass_rate": round(pass_rate, 3),
        "failed": failed,
    }


def run_retrieval_fixtures(project_memory: Any, fixtures: Iterable[RetrievalFixture] = DEFAULT_FIXTURES) -> Dict[str, Any]:
    fixtures = list(fixtures)
    results = []
    for fixture in fixtures:
        context = project_memory.get_context(query=fixture.query, max_tokens=project_memory.budget.total)
        results.append(evaluate_context(context, fixture))
    summary = summarize_fixture_results(results)
    return {
        "ok": summary["ok"],
        "summary": summary,
        "fixtures": [asdict(fixture) for fixture in fixtures],
        "results": results,
    }
