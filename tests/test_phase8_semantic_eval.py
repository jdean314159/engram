from dataclasses import dataclass

from engram.eval.retrieval_eval import (
    DEFAULT_FIXTURES,
    RetrievalFixture,
    evaluate_context,
    run_retrieval_fixtures,
    summarize_fixture_results,
)
from engram.project_memory import ContextResult
from engram.memory.working_memory import Message


@dataclass
class FakeEpisode:
    id: str
    text: str
    importance: float = 0.7
    timestamp: float = 1_700_000_000.0


class FixtureAwareProjectMemory:
    def __init__(self):
        self.budget = type("Budget", (), {"total": 120})()

    def get_context(self, query=None, max_tokens=120):
        q = (query or "").lower()
        context = ContextResult(
            working=[Message(role="user", content="We discussed Python and retry handling.", token_count=6)],
            episodic=[],
            semantic=[],
            cold=[],
        )
        if "asyncio" in q or "bug" in q:
            context.semantic.append({"type": "fact", "content": "The Python asyncio worker hit an error during gather()."})
        if "preferred language" in q:
            context.semantic.append({"type": "preference", "content": "Preference: language", "value": "Python"})
        if "retry loop" in q or "decide" in q:
            context.episodic.append(FakeEpisode(id="ep1", text="We decided to keep the retry loop and cap retries at three."))
        if "worker failure" in q or "retry" in q:
            context.semantic.append({"type": "graph_context", "content": "Related sentence: worker failure triggered the retry path."})
        if "archived" in q:
            context.cold.append({"text": "Archived Python incidents from earlier sessions."})
        return context


def test_evaluate_context_uses_fixture_thresholds():
    context = ContextResult(
        semantic=[{"type": "fact", "content": "Python asyncio error with retry loop."}],
        cold=[],
    )
    fixture = RetrievalFixture(
        name="threshold",
        query="python asyncio bug",
        expected_substrings=("python", "asyncio", "error"),
        min_recall=1.0,
    )
    result = evaluate_context(context, fixture)
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_run_retrieval_fixtures_returns_summary_for_default_set():
    pm = FixtureAwareProjectMemory()
    report = run_retrieval_fixtures(pm, DEFAULT_FIXTURES)
    assert report["ok"] is True
    assert report["summary"]["avg_recall"] >= 0.8
    assert report["summary"]["pass_rate"] == 1.0
    assert not report["summary"]["failed"]


def test_summarize_fixture_results_reports_failures():
    summary = summarize_fixture_results(
        [
            {"fixture": "a", "recall": 1.0, "passed": True},
            {"fixture": "b", "recall": 0.2, "passed": False},
        ]
    )
    assert summary["ok"] is False
    assert summary["failed"] == ["b"]
    assert summary["pass_rate"] == 0.5
