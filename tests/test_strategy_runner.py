from pathlib import Path

from engram import ProjectMemory, ProjectType


class FakeEngine:
    backend_label = "fake"
    model_name = "fake-model"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str) -> str:
        return "strategy reply"


def test_run_strategy_direct_answer(tmp_path: Path):
    pm = ProjectMemory(
        project_id="proj-strategy",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=tmp_path,
        llm_engine=FakeEngine(),
    )

    result = pm.run_strategy(
        "direct_answer",
        "Hello from strategy",
        query="Hello from strategy",
    )

    assert result["reply"] == "strategy reply"
    assert result["strategy"] == "direct_answer"
    assert "run_id" in result

    row = pm.get_run(result["run_id"])
    assert row is not None
    assert row["status"] == "succeeded"
    assert row["strategy"] == "direct_answer"

    pm.close()


def test_available_strategies_contains_direct_answer(tmp_path: Path):
    pm = ProjectMemory(
        project_id="proj-strategy-list",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=tmp_path,
        llm_engine=FakeEngine(),
    )

    strategies = pm.available_strategies()
    assert "direct_answer" in strategies

    pm.close()
