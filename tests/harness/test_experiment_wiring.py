"""Validation tests for P2 Bug #7 — ExperimentMemory auto-wired into ProjectMemory.

Tests verify:
1. pm.experiments is an ExperimentMemory instance (not None) after construction.
2. respond() automatically records start_run / finish_run.
3. Strategy label is stored correctly in the experiment record.
4. propose_then_verify and multi_candidate no longer crash (memory.experiments exists).
5. RunReporter works against the auto-populated experiments database.
6. Experiments persist across respond() calls in the same session.
7. ExperimentMemory path is inside the project directory (isolated per project).
8. Graceful: if ExperimentMemory construction fails, experiments=None and no crash.
"""

from __future__ import annotations


from pathlib import Path

from .runner import test_group
from .mocks import TempDir, MockEngine, unique_session


def _pm(d: Path, engine=None):
    from engram.project_memory import ProjectMemory
    pm = ProjectMemory(
        project_id="exp_test",
        project_type="general_assistant",
        base_dir=d,
        llm_engine=engine or MockEngine(),
        session_id=unique_session(),
    )
    return pm


def _answering_engine(answer="The answer."):
    engine = MockEngine()
    engine.generate = lambda prompt="", **kw: answer
    return engine


# ── Attribute presence ─────────────────────────────────────────────────────────

@test_group("P2-7: ExperimentMemory Wiring")
def test_pm_has_experiments_attribute():
    with TempDir() as d:
        pm = _pm(d)
        try:
            assert hasattr(pm, "experiments"), "ProjectMemory missing 'experiments' attribute"
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_pm_experiments_is_experiment_memory():
    from engram.memory.experiment_memory import ExperimentMemory
    with TempDir() as d:
        pm = _pm(d)
        try:
            assert isinstance(pm.experiments, ExperimentMemory), (
                f"Expected ExperimentMemory, got {type(pm.experiments)}"
            )
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_experiments_db_inside_project_dir():
    """experiments.db lives inside the project's own directory."""
    with TempDir() as d:
        pm = _pm(d)
        try:
            db_path = Path(pm.experiments.db_path)
            project_dir = d / "exp_test"
            assert db_path.is_relative_to(project_dir), (
                f"experiments.db {db_path} is not inside project dir {project_dir}"
            )
        finally:
            pm.close()

# ── respond() auto-records ─────────────────────────────────────────────────────

@test_group("P2-7: ExperimentMemory Wiring")
def test_respond_records_one_run():
    """A single respond() call creates exactly one experiment record."""
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            pm.respond("What is asyncio?")
            runs = pm.experiments.recent_runs(limit=10)
            assert len(runs) == 1
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_respond_run_status_succeeded():
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            pm.respond("What is asyncio?")
            run = pm.experiments.recent_runs(limit=1)[0]
            assert run["status"] == "succeeded"
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_respond_run_has_model_name():
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            pm.respond("Test question")
            run = pm.experiments.recent_runs(limit=1)[0]
            assert run.get("model_name") == "mock-model"
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_respond_strategy_label_recorded():
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            pm.respond("Test question", strategy="direct_answer")
            run = pm.experiments.recent_runs(limit=1)[0]
            assert run.get("strategy") == "direct_answer"
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_respond_strategy_none_when_not_provided():
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            pm.respond("Test question")
            run = pm.experiments.recent_runs(limit=1)[0]
            assert run.get("strategy") is None
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_respond_metrics_populated():
    """Run metrics contain duration_ms and prompt_tokens."""
    import json
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            pm.respond("Test question")
            run = pm.experiments.recent_runs(limit=1)[0]
            metrics_raw = run.get("metrics_json") or run.get("metrics") or "{}"
            if isinstance(metrics_raw, str):
                metrics = json.loads(metrics_raw)
            else:
                metrics = metrics_raw
            assert "duration_ms" in metrics
            assert metrics["duration_ms"] > 0
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_respond_multiple_calls_multiple_runs():
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            for i in range(4):
                pm.respond(f"Question {i}")
            runs = pm.experiments.recent_runs(limit=10)
            assert len(runs) == 4
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_respond_runs_include_session_id():
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            pm.respond("Question")
            run = pm.experiments.recent_runs(limit=1)[0]
            assert run.get("session_id") == pm.session_id
        finally:
            pm.close()

# ── Strategy wiring ────────────────────────────────────────────────────────────

@test_group("P2-7: ExperimentMemory Wiring")
def test_direct_answer_strategy_records_run():
    from engram.strategies.direct_answer import DirectAnswerStrategy
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            DirectAnswerStrategy().run(pm, "Test question")
            runs = pm.experiments.recent_runs(limit=5)
            assert len(runs) == 1
            assert runs[0]["strategy"] == "direct_answer"
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_propose_then_verify_no_longer_crashes():
    """propose_then_verify previously crashed with AttributeError on memory.experiments."""
    from engram.strategies.propose_then_verify import ProposeThenVerifyStrategy
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            result = ProposeThenVerifyStrategy().run(pm, "What is 6 times 7?")
            # propose_then_verify returns "reply" key (direct_answer uses "answer")
            assert "reply" in result or "answer" in result
            assert result["strategy"] == "propose_then_verify"
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_propose_then_verify_records_run():
    from engram.strategies.propose_then_verify import ProposeThenVerifyStrategy
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            ProposeThenVerifyStrategy().run(pm, "Test question")
            runs = pm.experiments.recent_runs(limit=5)
            assert any(r["strategy"] == "propose_then_verify" for r in runs)
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_multi_candidate_no_longer_crashes():
    """multi_candidate previously crashed with AttributeError on memory.experiments."""
    from engram.strategies.multi_candidate import MultiCandidateStrategy
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            result = MultiCandidateStrategy().run(pm, "What is 7 times 8?")
            # multi_candidate returns "reply" key (direct_answer uses "answer")
            assert "reply" in result or "answer" in result
            assert result["strategy"] == "multi_candidate"
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_multi_candidate_records_run():
    from engram.strategies.multi_candidate import MultiCandidateStrategy
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            MultiCandidateStrategy().run(pm, "Test question")
            runs = pm.experiments.recent_runs(limit=5)
            assert any(r["strategy"] == "multi_candidate" for r in runs)
        finally:
            pm.close()

@test_group("P2-7: ExperimentMemory Wiring")
def test_strategy_runner_dispatches_and_records():
    """StrategyRunner end-to-end: register → dispatch → run recorded."""
    from engram.strategies.runner import StrategyRunner
    from engram.strategies.direct_answer import DirectAnswerStrategy
    with TempDir() as d:
        pm = _pm(d, _answering_engine("Paris."))
        try:
            runner = StrategyRunner()
            runner.register(DirectAnswerStrategy())
            result = runner.run(pm, "direct_answer", "Capital of France?")
            assert result["answer"] == "Paris."
            runs = pm.experiments.recent_runs(limit=5)
            assert len(runs) == 1
        finally:
            pm.close()

# ── RunReporter integration ────────────────────────────────────────────────────

@test_group("P2-7: ExperimentMemory Wiring")
def test_run_reporter_sees_respond_runs():
    """RunReporter surfaces runs that were auto-recorded by respond()."""
    from engram.reporting.run_reporter import RunReporter
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            pm.respond("First question", strategy="direct_answer")
            pm.respond("Second question", strategy="direct_answer")
        finally:
            pm.close()
        reporter = RunReporter(pm.experiments)
        summaries = reporter.recent_run_summaries(limit=10)
        assert len(summaries) == 2
        assert all(s["strategy"] == "direct_answer" for s in summaries)


@test_group("P2-7: ExperimentMemory Wiring")
def test_run_reporter_strategy_summary():
    """strategy_summary aggregates across strategy types."""
    from engram.reporting.run_reporter import RunReporter
    from engram.strategies.direct_answer import DirectAnswerStrategy
    from engram.strategies.propose_then_verify import ProposeThenVerifyStrategy
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            DirectAnswerStrategy().run(pm, "Q1")
            ProposeThenVerifyStrategy().run(pm, "Q2")
        finally:
            pm.close()
        reporter = RunReporter(pm.experiments)
        summary = reporter.strategy_summary(limit=100)
        strategy_names = [s["strategy"] for s in summary]
        assert "direct_answer" in strategy_names
        assert "propose_then_verify" in strategy_names


# ── Graceful degradation ───────────────────────────────────────────────────────

@test_group("P2-7: ExperimentMemory Wiring")
def test_respond_works_when_experiments_is_none():
    """If experiments is None (construction failed), respond() still works."""
    with TempDir() as d:
        pm = _pm(d, _answering_engine())
        try:
            pm.experiments = None  # simulate construction failure
            result = pm.respond("Does this still work?")
            assert result["answer"] == "The answer."
            assert pm.working.get_message_count() == 2
        finally:
            pm.close()

# ── Project isolation ──────────────────────────────────────────────────────────

@test_group("P2-7: ExperimentMemory Wiring")
def test_experiments_isolated_per_project():
    """Two ProjectMemory instances have separate ExperimentMemory databases."""
    from engram.project_memory import ProjectMemory
    with TempDir() as d:
        with ProjectMemory(
            project_id="project_a", project_type="general_assistant",
            base_dir=d, llm_engine=_answering_engine("Answer A"),
            session_id=unique_session(),
        ) as pm1, ProjectMemory(
            project_id="project_b", project_type="general_assistant",
            base_dir=d, llm_engine=_answering_engine("Answer B"),
            session_id=unique_session(),
        ) as pm2:
            pm1.respond("Question for A")
            pm1.respond("Another question for A")
            pm2.respond("Question for B")

            runs_a = pm1.experiments.recent_runs(limit=10)
            runs_b = pm2.experiments.recent_runs(limit=10)

            assert len(runs_a) == 2
            assert len(runs_b) == 1
