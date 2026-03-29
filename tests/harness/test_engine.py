"""Tests for LLM engine layer — base types, router, privacy, structured output, Ollama."""

from __future__ import annotations

from .runner import test_group, require
from .mocks import MockEngine


# ── Base engine types ──────────────────────────────────────────────────────────

@test_group("Engine: Base")
def test_mock_engine_token_counter():
    eng = MockEngine()
    count = eng.count_tokens("Hello, world! This is a test.")
    assert count > 0
    assert isinstance(count, int)


@test_group("Engine: Base")
def test_mock_engine_generate():
    eng = MockEngine()
    result = eng.generate("What is 2+2?")
    assert isinstance(result, str)
    assert len(result) > 0


@test_group("Engine: Base")
def test_mock_engine_logprobs():
    require("pydantic")
    from engram.engine.base import LogprobResult
    result = MockEngine().generate_with_logprobs("Test prompt")
    assert isinstance(result, LogprobResult)
    assert len(result.token_logprobs) > 0
    assert isinstance(result.perplexity, float)
    assert result.perplexity > 0


@test_group("Engine: Base")
def test_logprob_result_perplexity():
    require("pydantic")
    import math
    from engram.engine.base import LogprobResult, TokenLogprob
    lp = LogprobResult(
        text="test",
        token_logprobs=[TokenLogprob("a", -1.0), TokenLogprob("b", -2.0)],
    )
    expected = math.exp(1.5)
    assert abs(lp.perplexity - expected) < 0.01


@test_group("Engine: Base")
def test_logprob_result_mean_logprob():
    require("pydantic")
    from engram.engine.base import LogprobResult, TokenLogprob
    lp = LogprobResult(
        text="test",
        token_logprobs=[TokenLogprob("a", -1.0), TokenLogprob("b", -3.0)],
    )
    assert abs(lp.mean_logprob - (-2.0)) < 1e-6


@test_group("Engine: Base")
def test_logprob_empty():
    require("pydantic")
    from engram.engine.base import LogprobResult
    lp = LogprobResult(text="empty", token_logprobs=[])
    assert lp.perplexity == 1.0
    assert lp.mean_logprob == 0.0
    assert lp.token_count == 0


@test_group("Engine: Base")
def test_token_counter():
    require("pydantic")
    from engram.engine.base import _count_tokens
    count = _count_tokens("Hello, this is a test sentence with several words.")
    assert count > 0


# ── FailoverEngine (Router) ────────────────────────────────────────────────────

def _make_failover(engines, policy=None):
    """Instantiate FailoverEngine — requires pydantic; skip if unavailable."""
    try:
        from engram.engine.router import FailoverEngine, FailoverPolicy
    except ImportError:
        require("pydantic")  # will raise SkipTest
    from engram.engine.router import FailoverEngine, FailoverPolicy
    return FailoverEngine(
        engines=engines,
        policy=policy or FailoverPolicy(max_attempts=2, transient_retry=False),
    )


@test_group("Engine: Router")
def test_router_primary_success():
    require("pydantic")
    primary = MockEngine()
    primary.generate = lambda prompt="", **kw: "Primary result"
    router = _make_failover([primary])
    result = router.generate("Test prompt")
    assert result == "Primary result"


@test_group("Engine: Router")
def test_router_failover_to_secondary():
    require("pydantic")

    class FailingEngine(MockEngine):
        def generate(self, prompt, **kwargs):
            raise RuntimeError("Primary failed")

    secondary = MockEngine()
    secondary.generate = lambda prompt="", **kw: "Secondary result"
    from engram.engine.router import FailoverPolicy
    router = _make_failover(
        [FailingEngine(), secondary],
        policy=FailoverPolicy(max_attempts=3, transient_retry=False),
    )
    result = router.generate("Test prompt")
    assert result == "Secondary result"


@test_group("Engine: Router")
def test_router_circuit_breaker_trips():
    require("pydantic")
    from engram.engine.router import _EngineHealth, FailoverPolicy
    health = _EngineHealth()
    policy = FailoverPolicy(circuit_breaker_failures=3, circuit_breaker_cooldown_s=60.0)

    health.record_failure(policy)
    health.record_failure(policy)
    assert health.is_healthy()
    health.record_failure(policy)
    assert not health.is_healthy()


@test_group("Engine: Router")
def test_router_circuit_breaker_recovery():
    require("pydantic")
    from engram.engine.router import _EngineHealth, FailoverPolicy
    policy = FailoverPolicy(circuit_breaker_failures=2, circuit_breaker_cooldown_s=60.0)
    health = _EngineHealth()
    health.record_failure(policy)
    health.record_failure(policy)
    assert not health.is_healthy()
    health.record_success()
    assert health.is_healthy()


@test_group("Engine: Router")
def test_router_all_engines_fail_raises():
    require("pydantic")
    from engram.engine.router import FailoverPolicy

    class AlwaysFail(MockEngine):
        def generate(self, p, **kw): raise RuntimeError("Always fails")

    router = _make_failover(
        [AlwaysFail()],
        policy=FailoverPolicy(max_attempts=1, transient_retry=False),
    )
    try:
        router.generate("Test")
        assert False, "Expected an exception"
    except Exception:
        pass


# ── Privacy / cloud sanitisation ──────────────────────────────────────────────

@test_group("Engine: Privacy")
def test_privacy_query_only_strips_memory():
    from engram.engine.utils.privacy import sanitize_prompt_for_cloud
    # Engram uses these exact delimiters (from engram.engine.utils.privacy._BEGIN/_END)
    prompt = (
        "----- BEGIN RETRIEVED MEMORY -----\n"
        "User prefers Python.\n"
        "----- END RETRIEVED MEMORY -----\n\n"
        "What is the capital of France?"
    )
    sanitized = sanitize_prompt_for_cloud(prompt, policy="query_only")
    assert "User prefers Python" not in sanitized
    assert "France" in sanitized


@test_group("Engine: Privacy")
def test_privacy_full_context_unchanged():
    from engram.engine.utils.privacy import sanitize_prompt_for_cloud
    prompt = "----- BEGIN RETRIEVED MEMORY -----\nSome data\n----- END RETRIEVED MEMORY -----\nQuery"
    result = sanitize_prompt_for_cloud(prompt, policy="full_context")
    assert result == prompt


@test_group("Engine: Privacy")
def test_privacy_no_memory_block_unchanged():
    """Prompts without a memory block pass through unchanged for any policy."""
    from engram.engine.utils.privacy import sanitize_prompt_for_cloud
    prompt = "What is asyncio? Please explain."
    for policy in ("query_only", "none", "query_plus_summary"):
        result = sanitize_prompt_for_cloud(prompt, policy=policy)
        assert "asyncio" in result


# ── Structured output ──────────────────────────────────────────────────────────

class _AnyDict:
    """Minimal pydantic-compatible model for testing."""
    pass

@test_group("Engine: Structured Output")
def test_structured_output_parse_valid_json():
    """parse() returns the validated model directly on success."""
    require("pydantic")
    from engram.engine.utils.structured_output import StructuredOutputHandler
    from pydantic import BaseModel

    class Schema(BaseModel):
        name: str
        type: str

    # parse() returns the model instance directly (raises StructuredOutputError on fail)
    result = StructuredOutputHandler.parse('{"name": "asyncio", "type": "library"}', Schema)
    assert result is not None
    assert result.name == "asyncio"
    assert result.type == "library"


@test_group("Engine: Structured Output")
def test_structured_output_strips_fences():
    """parse() handles markdown code fence wrapping transparently."""
    require("pydantic")
    from engram.engine.utils.structured_output import StructuredOutputHandler
    from pydantic import BaseModel

    class Schema(BaseModel):
        key: str

    raw = '```json\n{"key": "value"}\n```'
    result = StructuredOutputHandler.parse(raw, Schema)
    assert result.key == "value"


@test_group("Engine: Structured Output")
def test_structured_output_fails_on_garbage():
    """parse() raises StructuredOutputError on unparseable input."""
    require("pydantic")
    from engram.engine.utils.structured_output import (
        StructuredOutputHandler, StructuredOutputError
    )
    from pydantic import BaseModel

    class Schema(BaseModel):
        key: str

    # parse() raises StructuredOutputError; parse_with_details() returns ParseResult
    try:
        StructuredOutputHandler.parse("This is not JSON at all, sorry.", Schema)
        assert False, "Expected StructuredOutputError"
    except StructuredOutputError:
        pass  # expected


@test_group("Engine: Structured Output")
def test_structured_output_parse_with_details():
    """parse_with_details() returns a ParseResult with .success and .data fields."""
    require("pydantic")
    from engram.engine.utils.structured_output import StructuredOutputHandler
    from pydantic import BaseModel

    class Schema(BaseModel):
        key: str

    result = StructuredOutputHandler.parse_with_details('{"key": "hello"}', Schema)
    assert result.success is True
    assert result.data.key == "hello"

    bad = StructuredOutputHandler.parse_with_details("not json", Schema)
    assert bad.success is False


# ── Ollama engine ──────────────────────────────────────────────────────────────

@test_group("Engine: Ollama")
def test_ollama_engine_instantiation():
    """OllamaEngine can be instantiated without a live server."""
    require("openai", "pydantic")
    from engram.engine.ollama_engine import OllamaEngine
    engine = OllamaEngine(model_name="test-model", base_url="http://localhost:11434/v1")
    assert engine.model_name == "test-model"


@test_group("Engine: Ollama")
def test_ollama_engine_live():
    """Integration: generate against a live Ollama server."""
    require("openai", "pydantic")
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
    except Exception:
        from .runner import SkipTest
        raise SkipTest("No live Ollama server at localhost:11434")

    from engram.engine.ollama_engine import OllamaEngine
    engine = OllamaEngine(model_name="qwen2.5:7b", base_url="http://localhost:11434/v1")
    result = engine.generate("Reply with exactly the word: HELLO", max_tokens=10)
    assert isinstance(result, str)
    assert len(result) > 0
