import json
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
            "another wrong answer",
        ]
        self._idx = 0

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str) -> str:
        response = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return response


def test_propose_then_verify_selects_verified_candidate(tmp_path: Path):
    pm = ProjectMemory(
        project_id="proj-propose-verify",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=tmp_path,
        llm_engine=CyclingFakeEngine(),
    )

    result = pm.run_strategy(
        "propose_then_verify",
        "Choose the correct answer",
        query="Choose the correct answer",
        n_candidates=3,
        verifier=ExactMatchVerifier("correct answer"),
    )

    assert result["strategy"] == "propose_then_verify"
    assert result["reply"] == "correct answer"
    assert result["selected_index"] == 1
    assert result["selected_verification"]["passed"] is True
    assert result["verifier_name"] == "exact_match"

    row = pm.get_run(result["run_id"])
    assert row is not None
    assert row["status"] == "succeeded"
    assert row["strategy"] == "propose_then_verify"

    params = json.loads(row["parameters_json"])
    assert params["n_candidates"] == 3
    assert params["verifier_name"] == "exact_match"

    metrics = json.loads(row["metrics_json"])
    assert metrics["candidate_count"] == 3
    assert metrics["verification_pass_count"] == 1
    assert metrics["selected_passed"] is True

    pm.close()


def test_available_strategies_contains_propose_then_verify(tmp_path: Path):
    pm = ProjectMemory(
        project_id="proj-propose-verify-list",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=tmp_path,
        llm_engine=CyclingFakeEngine(),
    )

    strategies = pm.available_strategies()
    assert "propose_then_verify" in strategies

    pm.close()
