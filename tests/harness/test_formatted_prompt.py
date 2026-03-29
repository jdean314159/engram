"""Validation tests for P3 item #10 — ContextResult.to_formatted_prompt().

Tests verify:
1. Returns a non-empty string.
2. Contains the user_message formatted as "User: …\nAssistant:".
3. Contains the memory safety wrapper when context is non-empty.
4. Contains the system prompt prefix when provided.
5. Empty context produces a minimal prompt (no spurious blank sections).
6. Matches the format produced by build_prompt() for the same inputs.
7. Neural hint is injected when provided.
8. Works with no arguments (defaults to empty strings).
"""

from __future__ import annotations


from pathlib import Path

from .runner import test_group
from .mocks import TempDir, unique_session


def _pm(d: Path):
    from engram.project_memory import ProjectMemory
    pm = ProjectMemory(
        project_id="fmt_test",
        project_type="general_assistant",
        base_dir=d,
        llm_engine=None,
        session_id=unique_session(),
    )
    return pm


def _ctx_with_data(pm):
    """Add a couple of turns and return a populated ContextResult."""
    pm.add_turn("user", "I love asyncio and concurrency patterns")
    pm.add_turn("assistant", "asyncio is great for IO-bound tasks")
    return pm.get_context(query="asyncio", max_tokens=500)


# ── Basic contract ─────────────────────────────────────────────────────────────

@test_group("P3-10: to_formatted_prompt()")
def test_returns_non_empty_string():
    with TempDir() as d:
        ctx = _ctx_with_data(_pm(d))
        result = ctx.to_formatted_prompt(user_message="Tell me more.")
        assert isinstance(result, str)
        assert len(result) > 0


@test_group("P3-10: to_formatted_prompt()")
def test_user_message_appended_correctly():
    with TempDir() as d:
        ctx = _ctx_with_data(_pm(d))
        result = ctx.to_formatted_prompt(user_message="What is the event loop?")
        assert "What is the event loop?" in result
        assert "User: What is the event loop?" in result
        assert "Assistant:" in result


@test_group("P3-10: to_formatted_prompt()")
def test_memory_wrapper_present_when_context_nonempty():
    with TempDir() as d:
        ctx = _ctx_with_data(_pm(d))
        result = ctx.to_formatted_prompt(user_message="Test query")
        assert "----- BEGIN RETRIEVED MEMORY -----" in result
        assert "----- END RETRIEVED MEMORY -----" in result


@test_group("P3-10: to_formatted_prompt()")
def test_working_memory_in_output():
    with TempDir() as d:
        ctx = _ctx_with_data(_pm(d))
        result = ctx.to_formatted_prompt(user_message="asyncio question")
        assert "asyncio" in result.lower()
        assert "[WORKING]" in result


@test_group("P3-10: to_formatted_prompt()")
def test_system_prompt_prepended():
    with TempDir() as d:
        ctx = _ctx_with_data(_pm(d))
        result = ctx.to_formatted_prompt(
            user_message="Test",
            system_prompt="You are a Python expert.",
        )
        # System prompt must appear before the memory block
        assert "You are a Python expert." in result
        sys_pos = result.find("You are a Python expert.")
        mem_pos = result.find("BEGIN RETRIEVED MEMORY")
        assert sys_pos < mem_pos, "System prompt must precede memory block"


@test_group("P3-10: to_formatted_prompt()")
def test_no_system_prompt_no_stray_prefix():
    """With no system_prompt, the prompt starts with the memory block or user message."""
    with TempDir() as d:
        ctx = _ctx_with_data(_pm(d))
        result = ctx.to_formatted_prompt(user_message="Test", system_prompt="")
        # Must not have a leading blank line from an empty system prefix
        assert not result.startswith("\n")


@test_group("P3-10: to_formatted_prompt()")
def test_neural_hint_injected():
    with TempDir() as d:
        ctx = _ctx_with_data(_pm(d))
        result = ctx.to_formatted_prompt(
            user_message="Test",
            neural_hint="[familiar pattern: asyncio event loop]",
        )
        assert "[NEURAL CONTEXT]" in result
        assert "familiar pattern" in result


@test_group("P3-10: to_formatted_prompt()")
def test_no_args_returns_minimal_prompt():
    """to_formatted_prompt() with no args produces a syntactically valid prompt."""
    with TempDir() as d:
        pm = _pm(d)
        try:
            ctx = pm.get_context(query="anything", max_tokens=200)
            result = ctx.to_formatted_prompt()
            assert isinstance(result, str)
            # Empty context: no memory block, just "User: \nAssistant:"
            assert "BEGIN RETRIEVED MEMORY" not in result
            assert "Assistant:" in result
        finally:
            pm.close()

# ── Format consistency with build_prompt ──────────────────────────────────────

@test_group("P3-10: to_formatted_prompt()")
def test_format_matches_build_prompt_structure():
    """to_formatted_prompt() produces the same structure as build_prompt()."""
    with TempDir() as d:
        from engram.project_memory import ProjectMemory

        class FakeEngine:
            model_name = "fake"; is_cloud = False; system_prompt = "Be helpful."
            def count_tokens(self, t): return len(t) // 4
            def compress_prompt(self, p, n): return p
            def generate_with_logprobs(self, p, **kw):
                from engram.engine.base import LogprobResult, TokenLogprob
                return LogprobResult("ok", [TokenLogprob("t", -2.0)])
            def generate(self, p, **kw): return "answer"

        pm = ProjectMemory(
            project_id="cmp_test", project_type="general_assistant",
            base_dir=d, llm_engine=FakeEngine(), session_id=unique_session(),
        )
        pm.add_turn("user", "asyncio is great for concurrent I/O")
        pm.add_turn("assistant", "Absolutely, especially for network tasks")

        query = "Tell me more about asyncio."
        ctx = pm.get_context(query=query, max_tokens=500)

        # build_prompt path
        built = pm.build_prompt(query)
        bp_prompt = built["prompt"]

        # to_formatted_prompt path
        fp_prompt = ctx.to_formatted_prompt(
            user_message=query,
            system_prompt=FakeEngine.system_prompt,
        )

        # Both must contain the key structural elements
        for element in ("BEGIN RETRIEVED MEMORY", "END RETRIEVED MEMORY",
                        query, "Assistant:"):
            assert element in bp_prompt, f"build_prompt missing: {element!r}"
            assert element in fp_prompt, f"to_formatted_prompt missing: {element!r}"


# ── Section ordering ──────────────────────────────────────────────────────────

@test_group("P3-10: to_formatted_prompt()")
def test_section_ordering():
    """Sections appear in working → episodic → semantic → cold order."""
    with TempDir() as d:
        ctx = _ctx_with_data(_pm(d))
        result = ctx.to_formatted_prompt(user_message="test")

        positions = {}
        for section in ("WORKING", "EPISODIC", "SEMANTIC", "COLD"):
            tag = f"[{section}]"
            if tag in result:
                positions[section] = result.find(tag)

        present = sorted(positions.keys(), key=lambda k: positions[k])
        expected_order = ["WORKING", "EPISODIC", "SEMANTIC", "COLD"]
        # Only check sections that are actually present
        filtered_expected = [s for s in expected_order if s in positions]
        assert present == filtered_expected, (
            f"Unexpected section order: {present} (expected {filtered_expected})"
        )


@test_group("P3-10: to_formatted_prompt()")
def test_user_message_always_last():
    """User message must always appear after the memory block."""
    with TempDir() as d:
        ctx = _ctx_with_data(_pm(d))
        query = "What is the event loop in asyncio?"
        result = ctx.to_formatted_prompt(user_message=query)

        mem_end = result.find("----- END RETRIEVED MEMORY -----")
        user_pos = result.find(f"User: {query}")
        assert user_pos > mem_end, (
            "User message must appear after the END RETRIEVED MEMORY marker"
        )


# ── Callable from get_context result ──────────────────────────────────────────

@test_group("P3-10: to_formatted_prompt()")
def test_usable_directly_from_get_context():
    """Canonical usage pattern: get_context() → to_formatted_prompt() → engine."""
    with TempDir() as d:
        pm = _pm(d)
        try:
            pm.add_turn("user", "I prefer pytest for all testing")
            pm.add_turn("assistant", "Noted, pytest is excellent.")
        finally:
            pm.close()
        ctx = pm.get_context(query="testing preferences", max_tokens=400)
        prompt = ctx.to_formatted_prompt(
            user_message="What testing framework do I prefer?",
            system_prompt="You are a helpful coding assistant.",
        )

        # The prompt is a complete, self-contained string ready for an LLM
        assert isinstance(prompt, str)
        assert len(prompt) > 50
        # No pending substitution tokens or placeholder text
        assert "{" not in prompt or "}" not in prompt
