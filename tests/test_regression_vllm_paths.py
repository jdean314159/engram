"""Regression tests for vLLM failure paths, token clamping, and failover routing.

Scenarios covered:
  1. Healthy vLLM path — generate returns text
  2. Wrong served model — classify_vllm_generation_failure returns wrong_model
  3. vLLM down — classify_vllm_generation_failure returns not_running
  4. Server-side 500 — classify_vllm_generation_failure returns engine_crashed
  5. Token budget exceeded — 400/max_model_len → parse + clamp + retry
  6. Token clamping proactive — _effective_max_tokens clamps when limit known
  7. Reasoning leak suppressed — _strip_visible_reasoning removes think blocks
  8. Reasoning leak: clean answer preserved — no stripping when no markers
  9. Router error classification — token_budget classified correctly
 10. Router context-overflow no bare raise — falls through to next engine
 11. Failover: token_budget breaks to next engine (no re-raise)
 12. Error classifier covers all expected kinds

All tests are CPU-only with no network, no GPU, no LLM calls.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vllm_engine(max_context: int = 4096, reasoning_visibility: str = "auto"):
    """Construct a VLLMEngine with a mock OpenAI client."""
    from engram.engine.vllm_engine import VLLMEngine
    eng = VLLMEngine.__new__(VLLMEngine)
    # Minimal attribute init (bypass __init__ to avoid OpenAI import requirement)
    eng.base_url = "http://localhost:8000/v1"
    eng.model_name = "test-model"
    eng.configured_model_name = "test-model"
    eng.system_prompt = "You are a test assistant."
    eng._max_context = max_context
    eng.adaptive_context_limit = None
    eng.reasoning_visibility = reasoning_visibility
    eng._last_raw_response = None
    eng._last_cleaned_response = None
    eng._last_reasoning_detected = False
    eng._server_max_tokens = None
    eng.compression_strategy = None
    eng.max_retries = 1
    eng.timeout = 30
    eng.discovery_resolution = {}
    eng.client = MagicMock()
    eng.async_client = MagicMock()
    return eng


def _make_response(text: str):
    """Build a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# 1. Healthy vLLM path
# ---------------------------------------------------------------------------

def test_healthy_generate_returns_text():
    eng = _make_vllm_engine()
    eng.client.chat.completions.create.return_value = _make_response("Hello world")
    result = eng._generate_with_retry(
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.7,
        max_tokens=256,
    )
    assert result == "Hello world"


# ---------------------------------------------------------------------------
# 2. Wrong served model
# ---------------------------------------------------------------------------

def test_classify_wrong_model():
    from engram.engine.runtime_status import classify_vllm_generation_failure

    def list_models(endpoint):
        return [{"id": "other-model"}]

    exc = RuntimeError("404 model not found")
    result = classify_vllm_generation_failure("eng1", {
        "base_url": "http://localhost:8000/v1",
        "model": "expected-model",
    }, exc, list_models)
    assert result["failure_kind"] == "wrong_model"


# ---------------------------------------------------------------------------
# 3. vLLM down
# ---------------------------------------------------------------------------

def test_classify_not_running():
    from engram.engine.runtime_status import classify_vllm_generation_failure

    def list_models(endpoint):
        raise ConnectionError("Connection refused")

    exc = ConnectionError("Connection refused")
    result = classify_vllm_generation_failure("eng1", {
        "base_url": "http://localhost:8000/v1",
        "model": "some-model",
    }, exc, list_models)
    assert result["failure_kind"] == "not_running"


# ---------------------------------------------------------------------------
# 4. Server-side 500
# ---------------------------------------------------------------------------

def test_classify_engine_crashed():
    from engram.engine.runtime_status import classify_vllm_generation_failure

    def list_models(endpoint):
        return [{"id": "some-model"}]

    exc = RuntimeError("500 InternalServerError EngineCore")
    result = classify_vllm_generation_failure("eng1", {
        "base_url": "http://localhost:8000/v1",
        "model": "some-model",
    }, exc, list_models)
    assert result["failure_kind"] == "engine_crashed"


# ---------------------------------------------------------------------------
# 5. Token budget exceeded: 400 auto-retry path
# ---------------------------------------------------------------------------

def test_token_budget_400_retries_with_clamped_value():
    """400 max_model_len error should be caught, limit cached, retried once."""
    eng = _make_vllm_engine(max_context=4096)

    call_count = {"n": 0}

    def mock_create(model, messages, temperature, max_tokens, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            err = Exception(
                "400 BadRequestError: max_tokens=2048 cannot be greater than max_model_len=1024"
            )
            raise err
        # Second call succeeds
        return _make_response("Clamped response")

    eng.client.chat.completions.create.side_effect = mock_create
    result = eng._generate_with_retry(
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.7,
        max_tokens=2048,
    )
    assert result == "Clamped response"
    assert call_count["n"] == 2
    # Limit should be cached
    assert eng._server_max_tokens == 1024


# ---------------------------------------------------------------------------
# 6. Token clamping proactive
# ---------------------------------------------------------------------------

def test_effective_max_tokens_clamps_when_limit_known():
    eng = _make_vllm_engine()
    eng._server_max_tokens = 512
    assert eng._effective_max_tokens(2048) == 512 - 64
    assert eng._effective_max_tokens(100) == 100  # under limit, no clamp


def test_parse_max_model_len_from_error():
    from engram.engine.vllm_engine import VLLMEngine
    text = "max_tokens=2048 cannot be greater than max_model_len=1024"
    assert VLLMEngine._parse_max_model_len_from_error(text) == 1024
    assert VLLMEngine._parse_max_model_len_from_error("unrelated error") is None


# ---------------------------------------------------------------------------
# 7. Reasoning leak suppressed
# ---------------------------------------------------------------------------

def test_strip_think_tags():
    eng = _make_vllm_engine()
    raw = "<think>I need to plan...\nStep 1: do A\nStep 2: do B</think>\n\nThe answer is 42."
    result = eng._strip_visible_reasoning(raw)
    assert result == "The answer is 42."
    assert "think" not in result.lower()


def test_strip_reasoning_markers():
    eng = _make_vllm_engine()
    raw = "Thinking Process:\nLet me consider the options...\n\nFinal answer: Paris."
    result = eng._strip_visible_reasoning(raw)
    assert "Paris" in result
    assert "Thinking Process" not in result


def test_reasoning_visibility_show_preserves_raw():
    eng = _make_vllm_engine(reasoning_visibility="show")
    raw = "<think>internal</think>\nAnswer: hello"
    result = eng._apply_reasoning_visibility(raw)
    assert result == raw  # show mode: no stripping


def test_reasoning_visibility_hide_strips():
    eng = _make_vllm_engine(reasoning_visibility="hide")
    raw = "<think>internal</think>\nAnswer: hello"
    result = eng._apply_reasoning_visibility(raw)
    assert "think" not in result.lower()
    assert "hello" in result


# ---------------------------------------------------------------------------
# 8. Clean answer preserved — no false positives
# ---------------------------------------------------------------------------

def test_clean_answer_not_stripped():
    eng = _make_vllm_engine()
    clean = "The capital of France is Paris. It is known for the Eiffel Tower."
    result = eng._strip_visible_reasoning(clean)
    assert result == clean


# ---------------------------------------------------------------------------
# 9. Router error classification
# ---------------------------------------------------------------------------

def test_router_classify_token_budget():
    from engram.engine.router import FailoverEngine
    fe = FailoverEngine.__new__(FailoverEngine)
    fe.name = "test"
    fe._health = {}
    fe.policy = MagicMock()
    fe.engines = []
    fe.telemetry = MagicMock()

    exc = Exception("400: max_tokens=2048 cannot be greater than max_model_len=512")
    assert fe._classify_error(exc) == "token_budget"


def test_router_classify_context():
    from engram.engine.router import FailoverEngine
    fe = FailoverEngine.__new__(FailoverEngine)
    exc = Exception("context length exceeded: too many tokens in input")
    assert fe._classify_error(exc) == "context"


def test_router_classify_oom():
    from engram.engine.router import FailoverEngine
    fe = FailoverEngine.__new__(FailoverEngine)
    exc = RuntimeError("CUDA out of memory")
    assert fe._classify_error(exc) == "oom"


def test_router_classify_transient():
    from engram.engine.router import FailoverEngine
    fe = FailoverEngine.__new__(FailoverEngine)
    exc = TimeoutError("Connection timed out")
    assert fe._classify_error(exc) == "transient"


def test_router_classify_other():
    from engram.engine.router import FailoverEngine
    fe = FailoverEngine.__new__(FailoverEngine)
    exc = ValueError("unexpected schema")
    assert fe._classify_error(exc) == "other"


# ---------------------------------------------------------------------------
# 10. Context-overflow no bare raise
# ---------------------------------------------------------------------------

def test_context_overflow_falls_through_to_failover():
    """When context overflow compression budget is too small, engine switch not raise."""
    from engram.engine.router import FailoverEngine, FailoverPolicy
    from engram.engine.base import LLMEngine

    call_order = []

    class FailFirst(LLMEngine):
        system_prompt = ""
        is_cloud = False
        max_context_length = 512

        def __init__(self):
            super().__init__(model_name="fail-engine")

        def generate(self, prompt, **kwargs):
            call_order.append("fail")
            raise RuntimeError("context length exceeded: too many tokens")

        def generate_with_logprobs(self, prompt, **kwargs):
            raise NotImplementedError

        def stream(self, prompt, **kwargs):
            return iter(())

        def count_tokens(self, text):
            return len(text) // 4

        def compress_prompt(self, prompt, target_tokens):
            raise RuntimeError("compression failed")

    class SucceedSecond(LLMEngine):
        system_prompt = ""
        is_cloud = False
        max_context_length = 512

        def __init__(self):
            super().__init__(model_name="fallback-engine")

        def generate(self, prompt, **kwargs):
            call_order.append("succeed")
            return "fallback answer"

        def generate_with_logprobs(self, prompt, **kwargs):
            raise NotImplementedError

        def stream(self, prompt, **kwargs):
            return iter(())

        def count_tokens(self, text):
            return len(text) // 4

        def compress_prompt(self, prompt, target_tokens):
            return prompt[:target_tokens * 4]

    from engram.telemetry import Telemetry, LoggingSink

    policy = FailoverPolicy(
        allow_cloud_failover=False,
        max_attempts=4,
        compress_on_context_overflow=True,
    )
    fe = FailoverEngine(
        name="test",
        engines=[FailFirst(), SucceedSecond()],
        policy=policy,
        telemetry=Telemetry(sink=LoggingSink(), enabled=True),
    )

    out = fe.generate("x" * 10000)
    assert out == "fallback answer"
    assert call_order == ["fail", "succeed"]
    
# ---------------------------------------------------------------------------
# 11. Failover: token_budget breaks to next engine
# ---------------------------------------------------------------------------

def test_token_budget_fails_over_to_next_engine():
    from engram.engine.router import FailoverEngine, FailoverPolicy
    from engram.engine.base import LLMEngine
    from engram.telemetry import Telemetry, LoggingSink

    call_order = []

    class BudgetFail(LLMEngine):
        system_prompt = ""
        is_cloud = False
        max_context_length = 1024

        def __init__(self):
            super().__init__(model_name="budget-fail")

        def generate(self, prompt, **kwargs):
            call_order.append("budget_fail")
            raise RuntimeError(
                "400: max_tokens=2048 cannot be greater than max_model_len=512"
            )

        def generate_with_logprobs(self, prompt, **kwargs):
            raise NotImplementedError

        def stream(self, prompt, **kwargs):
            return iter(())

        def count_tokens(self, text):
            return len(text) // 4

        def compress_prompt(self, p, target_tokens):
            return p

    class Fallback(LLMEngine):
        system_prompt = ""
        is_cloud = False
        max_context_length = 1024

        def __init__(self):
            super().__init__(model_name="fallback")

        def generate(self, prompt, **kwargs):
            call_order.append("fallback")
            return "fallback response"

        def generate_with_logprobs(self, prompt, **kwargs):
            raise NotImplementedError

        def stream(self, prompt, **kwargs):
            return iter(())

        def count_tokens(self, text):
            return len(text) // 4

        def compress_prompt(self, p, target_tokens):
            return p

    policy = FailoverPolicy(allow_cloud_failover=False, max_attempts=4)
    fe = FailoverEngine(
        name="test",
        engines=[BudgetFail(), Fallback()],
        policy=policy,
        telemetry=Telemetry(sink=LoggingSink(), enabled=True),
    )

    out = fe.generate("hello", max_tokens=2048)
    assert out == "fallback response"
    assert call_order == ["budget_fail", "fallback"]
    
# ---------------------------------------------------------------------------
# 12. Token budget exceeded classification in runtime_status
# ---------------------------------------------------------------------------

def test_classify_token_budget_exceeded():
    from engram.engine.runtime_status import classify_vllm_generation_failure

    def list_models(endpoint):
        return [{"id": "some-model"}]

    exc = RuntimeError(
        "400 BadRequestError: max_tokens=2048 cannot be greater than max_model_len=1024"
    )
    result = classify_vllm_generation_failure("eng1", {
        "base_url": "http://localhost:8000/v1",
        "model": "some-model",
    }, exc, list_models)
    assert result["failure_kind"] == "token_budget_exceeded"
    assert "1024" in result["failure_message"]
