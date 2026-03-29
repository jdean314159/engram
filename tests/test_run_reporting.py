from pathlib import Path

from engram import ProjectMemory, ProjectType
from engram.verifiers import ExactMatchVerifier


class CyclingFakeEngine:
    backend_label = "fake"
    model_name = "cycling-fake-model"

    def __init__(self):
        self._responses = [
            "wrong answer",
            "correct answer",
            "",
            "fallback answer",
        ]
        self._idx = 0

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str) -> str:
        response = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return response


def test_recent_run_and_failure_summaries(tmp_path: Path):
    pm = ProjectMemory(
        project_id="proj-reporting-1",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=tmp_path,
        llm_engine=CyclingFakeEngine(),
    )

    pm.run_strategy(
        "direct_answer",
        "Say something",
        query="Say something",
    )

    pm.run_strategy(
        "multi_candidate",
        "Generate multiple answers",
        query="Generate multiple answers",
        n_candidates=3,
        selection_policy="first_non_empty",
    )

    recent = pm.recent_run_summaries(limit=10)
    assert len(recent) >= 2
    assert "strategy" in recent[0]
    assert "status" in recent[0]

    failures = pm.recent_failure_summaries(limit=10)
    assert isinstance(failures, list)

    pm.close()


def test_strategy_summary(tmp_path: Path):
    pm = ProjectMemory(
        project_id="proj-reporting-2",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=tmp_path,
        llm_engine=CyclingFakeEngine(),
    )

    pm.run_strategy(
        "direct_answer",
        "One direct answer",
        query="One direct answer",
    )

    pm.run_strategy(
        "multi_candidate",
        "Multi candidate answer",
        query="Multi candidate answer",
        n_candidates=3,
        selection_policy="first_non_empty",
    )

    pm.run_strategy(
        "propose_then_verify",
        "Verify candidates",
        query="Verify candidates",
        n_candidates=3,
        verifier=ExactMatchVerifier("correct answer"),
    )

    summary = pm.strategy_summary(limit=20)
    assert len(summary) >= 3

    strategies = {row["strategy"]: row for row in summary}
    assert "direct_answer" in strategies
    assert "multi_candidate" in strategies
    assert "propose_then_verify" in strategies

    assert strategies["direct_answer"]["run_count"] >= 1
    assert strategies["multi_candidate"]["run_count"] >= 1
    assert strategies["propose_then_verify"]["run_count"] >= 1
    assert strategies["propose_then_verify"]["verified_selected_count"] >= 1

    pm.close()
