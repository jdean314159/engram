from pathlib import Path

from engram import ProjectMemory, ProjectType


class FakeEngine:
    backend_label = "fake"
    model_name = "fake-model"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str) -> str:
        return "test reply"


def test_respond_writes_experiment_run(tmp_path: Path):
    pm = ProjectMemory(
        project_id="proj-test",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=tmp_path,
        llm_engine=FakeEngine(),
    )

    result = pm.respond("Hello there", query="Hello there")

    assert result["reply"] == "test reply"
    assert "run_id" in result

    row = pm.get_run(result["run_id"])
    assert row is not None
    assert row["status"] == "succeeded"
    assert row["backend_label"] == "fake"
    assert row["model_name"] == "fake-model"

    pm.close()
