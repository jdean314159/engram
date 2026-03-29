import json
from pathlib import Path

from engram import ProjectMemory, ProjectType


class CyclingFakeEngine:
    backend_label = "fake"
    model_name = "cycling-fake-model"

    def __init__(self):
        self._responses = [
            "",
            "second candidate",
            "third candidate is longer",
        ]
        self._idx = 0

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str) -> str:
        response = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return response


def test_multi_candidate_first_non_empty(tmp_path: Path):
    pm = ProjectMemory(
        project_id="proj-multi-candidate",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=tmp_path,
        llm_engine=CyclingFakeEngine(),
    )

    result = pm.run_strategy(
        "multi_candidate",
        "Hello from multi-candidate",
        query="Hello from multi-candidate",
        n_candidates=3,
        selection_policy="first_non_empty",
    )

    assert result["strategy"] == "multi_candidate"
    assert result["reply"] == "second candidate"
    assert result["selected_index"] == 1
    assert len(result["candidates"]) == 3

    row = pm.get_run(result["run_id"])
    assert row is not None
    assert row["status"] == "succeeded"
    assert row["strategy"] == "multi_candidate"

    params = json.loads(row["parameters_json"])
    assert params["n_candidates"] == 3
    assert params["selection_policy"] == "first_non_empty"

    pm.close()


def test_multi_candidate_longest_non_empty(tmp_path: Path):
    pm = ProjectMemory(
        project_id="proj-multi-candidate-longest",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=tmp_path,
        llm_engine=CyclingFakeEngine(),
    )

    result = pm.run_strategy(
        "multi_candidate",
        "Pick the longest candidate",
        query="Pick the longest candidate",
        n_candidates=3,
        selection_policy="longest_non_empty",
    )

    assert result["strategy"] == "multi_candidate"
    assert result["reply"] == "third candidate is longer"
    assert result["selected_index"] == 2
    assert len(result["candidates"]) == 3

    pm.close()


def test_available_strategies_contains_multi_candidate(tmp_path: Path):
    pm = ProjectMemory(
        project_id="proj-strategy-list-2",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=tmp_path,
        llm_engine=CyclingFakeEngine(),
    )

    strategies = pm.available_strategies()
    assert "multi_candidate" in strategies

    pm.close()
