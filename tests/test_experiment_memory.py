from pathlib import Path

from engram.memory.experiment_memory import ExperimentMemory


def test_start_and_finish_run(tmp_path: Path):
    db = ExperimentMemory(tmp_path / "experiments.db")

    run_id = db.start_run(
        project_id="proj-1",
        session_id="sess-1",
        goal="Test a response path",
        task_type="chat",
        strategy="direct_answer",
        parameters={"temperature": 0.2},
    )

    db.finish_run(
        run_id,
        status="succeeded",
        backend_label="ollama",
        model_name="qwen",
        metrics={"duration_ms": 12.3, "prompt_tokens": 100},
        retrieval={"query": "hello"},
        outcome_summary="hello back",
        failure_mode=None,
        lessons_learned=["worked"],
        artifacts=[],
    )

    row = db.get_run(run_id)
    assert row is not None
    assert row["run_id"] == run_id
    assert row["status"] == "succeeded"
    assert row["backend_label"] == "ollama"
    assert row["model_name"] == "qwen"

    db.close()


def test_search_runs_filters(tmp_path: Path):
    db = ExperimentMemory(tmp_path / "experiments.db")

    run1 = db.start_run(
        project_id="proj-1",
        session_id="sess-1",
        goal="Chat run",
        task_type="chat",
        strategy="direct_answer",
    )
    db.finish_run(run1, status="succeeded")

    run2 = db.start_run(
        project_id="proj-1",
        session_id="sess-1",
        goal="Eval run",
        task_type="eval_run",
        strategy="parallel_candidates",
    )
    db.finish_run(run2, status="failed", failure_mode="TimeoutError")

    rows = db.search_runs(task_type="eval_run")
    assert len(rows) == 1
    assert rows[0]["run_id"] == run2

    rows = db.search_runs(strategy="direct_answer")
    assert len(rows) == 1
    assert rows[0]["run_id"] == run1

    rows = db.search_runs(failure_mode="TimeoutError")
    assert len(rows) == 1
    assert rows[0]["run_id"] == run2

    db.close()
    
