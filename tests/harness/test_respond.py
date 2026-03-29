"""Validation tests for ProjectMemory.respond() — P0 fix #1.

These tests are additive: they import the existing harness runner and
register into the same RUNNER so `python run_tests.py` picks them up
automatically via tests/harness/__init__.py.

Run standalone:
    cd engram
    PYTHONPATH=. python -m pytest tests/harness/test_respond.py -v
    # or via harness runner after adding to __init__.py:
    PYTHONPATH=. python run_tests.py --groups "ProjectMemory: respond()"
"""

from __future__ import annotations


import threading
from pathlib import Path

from .runner import test_group
from .mocks import TempDir, MockEngine, unique_session


def _pm(d: Path, engine=None):
    from engram.project_memory import ProjectMemory
    pm = ProjectMemory(
        project_id="respond_test",
        project_type="general_assistant",
        base_dir=d,
        llm_engine=engine or MockEngine(),
        session_id=unique_session(),
    )
    return pm


# ── Basic contract ─────────────────────────────────────────────────────────────

@test_group("ProjectMemory: respond()")
def test_respond_returns_required_keys():
    with TempDir() as d:
        pm = _pm(d)
        try:
            result = pm.respond("What is 2 + 2?")
            for key in ("answer", "prompt", "prompt_tokens", "memory_tokens",
                        "compressed", "strategy"):
                assert key in result, f"Missing key: {key!r}"
        finally:
            pm.close()

@test_group("ProjectMemory: respond()")
def test_respond_answer_is_engine_output():
    with TempDir() as d:
        engine = MockEngine()
        engine.generate = lambda prompt="", **kw: "Forty-two."
        pm = _pm(d, engine)
        try:
            result = pm.respond("What is 6 * 7?")
            assert result["answer"] == "Forty-two."
        finally:
            pm.close()

@test_group("ProjectMemory: respond()")
def test_respond_records_both_turns():
    """respond() must store user turn AND assistant turn in working memory."""
    with TempDir() as d:
        pm = _pm(d)
        try:
            assert pm.working.get_message_count() == 0
            pm.respond("Hello there")
            assert pm.working.get_message_count() == 2
            turns = pm.get_recent_turns(n=2)
            roles = {t.role for t in turns}
            assert "user" in roles
            assert "assistant" in roles
        finally:
            pm.close()

@test_group("ProjectMemory: respond()")
def test_respond_user_message_in_working_memory():
    with TempDir() as d:
        pm = _pm(d)
        try:
            pm.respond("Tell me about asyncio")
            recent = pm.get_recent_turns(n=5)
            contents = [t.content for t in recent]
            assert any("asyncio" in c for c in contents)
        finally:
            pm.close()

@test_group("ProjectMemory: respond()")
def test_respond_assistant_answer_in_working_memory():
    with TempDir() as d:
        engine = MockEngine()
        engine.generate = lambda prompt="", **kw: "asyncio uses an event loop."
        pm = _pm(d, engine)
        try:
            pm.respond("Explain asyncio")
            recent = pm.get_recent_turns(n=5)
            assistant_turns = [t for t in recent if t.role == "assistant"]
            assert len(assistant_turns) == 1
            assert assistant_turns[0].content == "asyncio uses an event loop."
        finally:
            pm.close()

@test_group("ProjectMemory: respond()")
def test_respond_prompt_contains_user_message():
    """The assembled prompt must include the user's message."""
    with TempDir() as d:
        pm = _pm(d)
        try:
            result = pm.respond("What is the Pythagorean theorem?")
            assert "Pythagorean" in result["prompt"]
        finally:
            pm.close()

@test_group("ProjectMemory: respond()")
def test_respond_prompt_contains_prior_context():
    """On the second turn, the prompt must include the first exchange."""
    with TempDir() as d:
        engine = MockEngine()
        engine.generate = lambda prompt="", **kw: "asyncio is for async I/O."
        pm = _pm(d, engine)
        try:

            pm.respond("What is asyncio?")
        finally:
            pm.close()
        prompts = []
        engine.generate = lambda prompt="", **kw: prompts.append(prompt) or "Great question."
        pm.respond("Can you elaborate?")

        assert len(prompts) == 1
        assert "asyncio" in prompts[0].lower()


@test_group("ProjectMemory: respond()")
def test_respond_strategy_label_passed_through():
    with TempDir() as d:
        pm = _pm(d)
        try:
            result = pm.respond("Test question", strategy="direct_answer")
            assert result["strategy"] == "direct_answer"
        finally:
            pm.close()

@test_group("ProjectMemory: respond()")
def test_respond_strategy_none_by_default():
    with TempDir() as d:
        pm = _pm(d)
        try:
            result = pm.respond("Test question")
            assert result["strategy"] is None
        finally:
            pm.close()

@test_group("ProjectMemory: respond()")
def test_respond_without_engine_raises():
    """respond() with no llm_engine must raise RuntimeError immediately."""
    with TempDir() as d:
        from engram.project_memory import ProjectMemory
        pm = ProjectMemory(
            project_id="no_engine",
            project_type="general_assistant",
            base_dir=d,
            llm_engine=None,
            session_id=unique_session(),
        )
        try:
            pm.respond("Will this work?")
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "llm_engine" in str(e).lower() or "respond" in str(e).lower()


@test_group("ProjectMemory: respond()")
def test_respond_engine_kwargs_forwarded():
    """Extra kwargs are forwarded to engine.generate()."""
    with TempDir() as d:
        received = {}
        engine = MockEngine()
        engine.generate = lambda prompt="", **kw: received.update(kw) or "ok"
        pm = _pm(d, engine)
        try:
            pm.respond("Test", temperature=0.1, max_tokens=128)
            assert received.get("temperature") == 0.1
            assert received.get("max_tokens") == 128
        finally:
            pm.close()

# ── Multi-turn coherence ───────────────────────────────────────────────────────

@test_group("ProjectMemory: respond()")
def test_respond_multi_turn_accumulates():
    """Working memory grows with each respond() call."""
    with TempDir() as d:
        pm = _pm(d)
        try:
            for i in range(5):
                pm.respond(f"Question {i}")
            # 5 questions + 5 answers = 10 turns
            assert pm.working.get_message_count() == 10
        finally:
            pm.close()

@test_group("ProjectMemory: respond()")
def test_respond_new_session_clears_context():
    """After new_session(), the prior exchange is not in the next prompt."""
    with TempDir() as d:
        engine = MockEngine()
        engine.generate = lambda prompt="", **kw: "I know about bananas."
        pm = _pm(d, engine)
        try:

            pm.respond("Tell me about bananas")
        finally:
            pm.close()
        prompts = []
        engine.generate = lambda prompt="", **kw: prompts.append(prompt) or "Fresh start."
        pm.new_session(unique_session())
        pm.respond("What did we talk about?")

        assert prompts, "No prompt captured"
        assert "bananas" not in prompts[0].lower()


@test_group("ProjectMemory: respond()")
def test_respond_memory_context_improves_over_turns():
    """Memory grows: prompt on turn 3 is longer than prompt on turn 1."""
    with TempDir() as d:
        prompts = []
        engine = MockEngine()
        engine.generate = lambda prompt="", **kw: prompts.append(prompt) or "Noted."
        pm = _pm(d, engine)
        try:

            pm.respond("My name is Jeff")
            pm.respond("I prefer Python")
            pm.respond("I work in cybersecurity")
        finally:
            pm.close()
        assert len(prompts) == 3
        # Each successive prompt should be longer (more context)
        assert len(prompts[2]) >= len(prompts[0])


# ── DirectAnswerStrategy now works ────────────────────────────────────────────

@test_group("ProjectMemory: respond()")
def test_direct_answer_strategy_uses_respond():
    """DirectAnswerStrategy.run() now succeeds via ProjectMemory.respond()."""
    from engram.strategies.direct_answer import DirectAnswerStrategy
    from engram.project_memory import ProjectMemory

    engine = MockEngine()
    engine.generate = lambda prompt="", **kw: "The answer is 42."

    strategy = DirectAnswerStrategy()
    with TempDir() as d:
        pm = ProjectMemory(
            project_id="strategy_test",
            project_type="general_assistant",
            base_dir=d,
            llm_engine=engine,
            session_id=unique_session(),
        )
        result = strategy.run(pm, "What is 6 times 7?")
        assert isinstance(result, dict)
        assert result["strategy"] == "direct_answer"
        assert result["answer"] == "The answer is 42."
        # Both turns recorded
        assert pm.working.get_message_count() == 2


@test_group("ProjectMemory: respond()")
def test_strategy_runner_dispatches_via_respond():
    """StrategyRunner end-to-end: register → dispatch → answer returned."""
    from engram.strategies.runner import StrategyRunner
    from engram.strategies.direct_answer import DirectAnswerStrategy
    from engram.project_memory import ProjectMemory

    engine = MockEngine()
    engine.generate = lambda prompt="", **kw: "Paris."

    runner = StrategyRunner()
    runner.register(DirectAnswerStrategy())

    with TempDir() as d:
        pm = ProjectMemory(
            project_id="runner_test",
            project_type="general_assistant",
            base_dir=d,
            llm_engine=engine,
            session_id=unique_session(),
        )
        result = runner.run(pm, "direct_answer", "What is the capital of France?")
        assert result["answer"] == "Paris."
        assert result["strategy"] == "direct_answer"
