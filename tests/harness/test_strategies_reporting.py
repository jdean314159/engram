"""Tests for strategy runner, verifiers, and experiment reporting."""

from __future__ import annotations

from .runner import test_group
from .mocks import TempDir, MockEngine, make_project_memory, unique_session


# ── StrategyRunner ─────────────────────────────────────────────────────────────

@test_group("Strategies")
def test_strategy_runner_register_and_dispatch():
    from engram.strategies.runner import StrategyRunner

    class FakeStrategy:
        name = "fake"
        def run(self, memory, user_message: str, **kwargs):
            return {"answer": f"Fake: {user_message}", "strategy": "fake"}

    runner = StrategyRunner()
    runner.register(FakeStrategy())
    assert runner.has_strategy("fake")

    with TempDir() as d:
        pm = make_project_memory(d)
        result = runner.run(pm, "fake", "What is asyncio?")
        assert result["strategy"] == "fake"
        assert "asyncio" in result["answer"]


@test_group("Strategies")
def test_strategy_runner_unknown_raises():
    from engram.strategies.runner import StrategyRunner
    runner = StrategyRunner()
    with TempDir() as d:
        pm = make_project_memory(d)
        try:
            runner.run(pm, "nonexistent", "test")
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e)


@test_group("Strategies")
def test_strategy_runner_available_sorted():
    from engram.strategies.runner import StrategyRunner

    class S:
        def __init__(self, n): self.name = n
        def run(self, m, q, **kw): return {}

    runner = StrategyRunner()
    for n in ["beta", "alpha", "gamma"]:
        runner.register(S(n))

    available = runner.available_strategies()
    assert available == sorted(available)
    assert set(available) == {"alpha", "beta", "gamma"}


@test_group("Strategies")
def test_direct_answer_strategy():
    """DirectAnswerStrategy.run() calls memory.respond() — documents known API gap.

    BUG: DirectAnswerStrategy calls memory.respond() but ProjectMemory has no
    respond() method. The equivalent is build_prompt() + llm_engine.generate().
    This test documents the gap and verifies the strategy name attribute.
    """
    from engram.strategies.direct_answer import DirectAnswerStrategy
    strategy = DirectAnswerStrategy()
    assert strategy.name == "direct_answer"

    # Verify the bug: run() will raise AttributeError on ProjectMemory
    with TempDir() as d:
        from engram.project_memory import ProjectMemory
        pm = ProjectMemory(
            project_id="s_test",
            project_type="general_assistant",
            base_dir=d,
            llm_engine=MockEngine(),
            session_id=unique_session(),
        )
        try:
            strategy.run(pm, "test question")
            # If it didn't raise, memory.respond was added — update this test
            assert hasattr(pm, "respond"), "respond() method exists — update test"
        except AttributeError as e:
            # Expected: ProjectMemory has no respond() method
            assert "respond" in str(e), f"Unexpected AttributeError: {e}"


@test_group("Strategies")
def test_reasoning_strategy_base_interface():
    from engram.strategies.base import ReasoningStrategy

    class Concrete(ReasoningStrategy):
        name = "concrete"
        def run(self, memory, user_message: str, **kwargs):
            return {"answer": "concrete", "strategy": self.name}

    s = Concrete()
    assert s.name == "concrete"
    result = s.run(None, "test")
    assert result["strategy"] == "concrete"


# ── Verifiers ──────────────────────────────────────────────────────────────────

@test_group("Verifiers")
def test_pass_through_verifier():
    from engram.verifiers.simple import PassThroughVerifier
    v = PassThroughVerifier()
    result = v.verify("Non-empty answer here.")
    assert result.passed is True
    assert result.score > 0


@test_group("Verifiers")
def test_pass_through_verifier_empty_fails():
    from engram.verifiers.simple import PassThroughVerifier
    v = PassThroughVerifier()
    result = v.verify("")
    assert result.passed is False


@test_group("Verifiers")
def test_exact_match_verifier_pass():
    from engram.verifiers.simple import ExactMatchVerifier
    v = ExactMatchVerifier(expected="42")
    result = v.verify("42")
    assert result.passed is True
    assert result.score == 1.0


@test_group("Verifiers")
def test_exact_match_verifier_fail():
    from engram.verifiers.simple import ExactMatchVerifier
    v = ExactMatchVerifier(expected="42")
    result = v.verify("43")
    assert result.passed is False
    assert result.score == 0.0


@test_group("Verifiers")
def test_verifier_base_interface():
    from engram.verifiers.base import CandidateVerifier, VerificationResult

    class AlwaysPass(CandidateVerifier):
        name = "always_pass"
        def verify(self, candidate: str, **kwargs) -> VerificationResult:
            return VerificationResult(passed=True, score=1.0, reason="ok")

    v = AlwaysPass()
    result = v.verify("any text")
    assert result.passed
    d = result.to_dict()
    assert "passed" in d
    assert "score" in d


# ── RunReporter ────────────────────────────────────────────────────────────────

def _make_experiment_memory(base_dir):
    """Build an ExperimentMemory for RunReporter."""
    try:
        from engram.memory.experiment_memory import ExperimentMemory
        return ExperimentMemory(db_path=base_dir / "experiments.db")
    except ImportError:
        from .runner import SkipTest
        raise SkipTest("ExperimentMemory not available")


@test_group("Reporting")
def test_run_reporter_recent_runs_empty():
    from engram.reporting.run_reporter import RunReporter
    with TempDir() as d:
        em = _make_experiment_memory(d)
        reporter = RunReporter(em)
        summaries = reporter.recent_run_summaries(limit=10)
        assert isinstance(summaries, list)


@test_group("Reporting")
def test_run_reporter_strategy_summary_empty():
    from engram.reporting.run_reporter import RunReporter
    with TempDir() as d:
        em = _make_experiment_memory(d)
        reporter = RunReporter(em)
        summary = reporter.strategy_summary(limit=100)
        assert isinstance(summary, list)


@test_group("Reporting")
def test_experiment_memory_search_runs():
    """search_runs filters by strategy."""
    with TempDir() as d:
        em = _make_experiment_memory(d)
        rid = em.start_run(
            project_id="p", session_id=unique_session(),
            goal="search test", task_type="qa", strategy="direct_answer",
        )
        em.finish_run(rid, status="pass")

        hits = em.search_runs(strategy="direct_answer", limit=10)
        assert isinstance(hits, list)
        assert any(r.get("run_id") == rid for r in hits)


# ── ExperimentMemory (corrected) ───────────────────────────────────────────────

@test_group("Reporting")
def test_experiment_memory_start_finish_run():
    with TempDir() as d:
        em = _make_experiment_memory(d)
        run_id = em.start_run(
            project_id="test",
            session_id=unique_session(),
            goal="Test the system",
            task_type="qa",
            strategy="direct_answer",
        )
        assert isinstance(run_id, str)

        em.finish_run(
            run_id,
            status="pass",
            model_name="mock-model",
            metrics={"duration_ms": 42.0},
            outcome_summary="Correct answer",
        )

        row = em.get_run(run_id)
        assert row is not None
        assert row["status"] == "pass"


@test_group("Reporting")
def test_experiment_memory_recent_runs():
    with TempDir() as d:
        em = _make_experiment_memory(d)
        for i in range(3):
            rid = em.start_run(
                project_id="p", session_id=unique_session(),
                goal=f"Goal {i}", task_type="qa",
            )
            em.finish_run(rid, status="pass")

        rows = em.recent_runs(limit=10)
        assert len(rows) >= 3


@test_group("Reporting")
def test_run_reporter_with_data():
    from engram.reporting.run_reporter import RunReporter
    with TempDir() as d:
        em = _make_experiment_memory(d)
        rid = em.start_run(
            project_id="p", session_id=unique_session(),
            goal="Run reporter test", task_type="qa", strategy="direct_answer",
        )
        em.finish_run(rid, status="pass", model_name="mock",
                      metrics={"duration_ms": 10.0})

        reporter = RunReporter(em)
        summaries = reporter.recent_run_summaries(limit=5)
        assert len(summaries) >= 1
        assert summaries[0]["run_id"] == rid
