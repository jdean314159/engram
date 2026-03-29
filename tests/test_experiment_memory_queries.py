from pathlib import Path

from engram.memory.experiment_memory import ExperimentMemory


def test_recent_failures_and_recent_runs(tmp_path: Path):
    db = ExperimentMemory(tmp_path / "experiments.db")

    run1 = db.start_run(
        project_id="proj",
        session_id="sess",
        goal="ok run",
        task_type="chat",
    )
    db.finish_run(run1, status="succeeded")

    run2 = db.start_run(
        project_id="proj",
        session_id="sess",
        goal="bad run",
        task_type="chat",
    )
    db.finish_run(run2, status="failed", failure_mode="TimeoutError")

    failures = db.recent_failures(limit=10)
    assert len(failures) == 1
    assert failures[0]["run_id"] == run2

    runs = db.recent_runs(limit=10)
    assert len(runs) == 2

    db.close()
