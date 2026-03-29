"""Tests for LlamaCppEngine.

All tests are CPU-only and do not require a live llama-server instance.
Network calls are intercepted via unittest.mock so the tests run in any
environment.

Coverage:
  1.  Construction and attribute defaults
  2.  Config loader dispatch (type: llama_cpp / llama-cpp / llamacpp)
  3.  generate(): response extraction, reasoning stripping, token clamping
  4.  generate(): connection error raises cleanly
  5.  max_context_length property
  6.  count_tokens delegates to _count_tokens
  7.  is_cloud is False
  8.  supports_logprobs: probe returns True when logprobs present
  9.  supports_logprobs: probe returns False when logprobs absent/error
 10.  supports_logprobs: result is cached (probe called only once)
 11.  build_llama_cpp_launch_command: full set of params
 12.  build_llama_cpp_launch_command: no gguf_path → None
 13.  build_llama_cpp_launch_command: n_gpu_layers=0 (CPU-only)
 14.  build_llama_cpp_launch_command: n_gpu_layers=40 (split)
 15.  build_llama_cpp_launch_command: extra_args forwarded
 16.  classify_llama_cpp_generation_failure: all failure kinds
 17.  build_llama_cpp_recovery_guidance: includes launch command
 18.  Reasoning visibility: auto strips <think> blocks
 19.  Reasoning visibility: show preserves reasoning
 20.  Reasoning visibility: hide always strips
 21.  get_diagnostics returns expected keys
 22.  Failover profile can reference llama_cpp engine (config round-trip)
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Lightweight stubs for optional packages absent in the test environment
# ---------------------------------------------------------------------------

def _stub_pydantic():
    if "pydantic" not in sys.modules:
        stub = types.ModuleType("pydantic")
        class BaseModel: pass
        stub.BaseModel = BaseModel
        stub.Field = lambda *a, **kw: None
        sys.modules["pydantic"] = stub
        sys.modules["pydantic.v1"] = stub

def _stub_openai():
    if "openai" not in sys.modules:
        stub = types.ModuleType("openai")
        stub.OpenAI = MagicMock
        stub.AsyncOpenAI = MagicMock
        sys.modules["openai"] = stub

_stub_pydantic()
_stub_openai()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(**kwargs):
    """Construct a LlamaCppEngine with sensible defaults."""
    from engram.engine.llama_cpp_engine import LlamaCppEngine, CompressionStrategy
    defaults = dict(
        model_name="test_model",
        base_url="http://127.0.0.1:8080/v1",
        max_context=4096,
        gguf_path="/models/test.gguf",
        n_gpu_layers=0,
    )
    defaults.update(kwargs)
    return LlamaCppEngine(**defaults)


def _mock_completion(content: str, logprobs=None):
    """Build a minimal mock OpenAI chat completion response."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        logprobs=logprobs,
    )
    return SimpleNamespace(choices=[choice])


# ---------------------------------------------------------------------------
# 1. Construction and defaults
# ---------------------------------------------------------------------------

def test_construction_defaults():
    eng = _make_engine()
    assert eng.model_name == "test_model"
    assert eng.base_url == "http://127.0.0.1:8080/v1"
    assert eng._max_context == 4096
    assert eng.gguf_path == "/models/test.gguf"
    assert eng.n_gpu_layers == 0
    assert eng.reasoning_visibility == "auto"
    assert eng._supports_logprobs is None   # not yet probed
    assert eng._last_raw_response is None


def test_construction_gpu_layers():
    eng = _make_engine(n_gpu_layers=40)
    assert eng.n_gpu_layers == 40


def test_construction_no_gguf_path():
    from engram.engine.llama_cpp_engine import LlamaCppEngine
    eng = LlamaCppEngine(model_name="m", gguf_path=None)
    assert eng.gguf_path is None


# ---------------------------------------------------------------------------
# 2. Config loader dispatch
# ---------------------------------------------------------------------------

def _engine_from_type(type_str: str):
    """Create engine via config_loader using a minimal in-memory config."""
    from engram.engine.config_loader import create_engine
    cfg = {
        "engines": {
            "test_llama": {
                "type": type_str,
                "model": "my_model",
                "base_url": "http://127.0.0.1:8080/v1",
                "max_context": 8192,
                "gguf_path": "/models/q.gguf",
                "n_gpu_layers": 20,
                "timeout": 120,
                "system_prompt": "You are helpful.",
                "compression_strategy": "compress",
                "max_retries": 1,
            }
        }
    }
    with patch("engram.engine.config_loader.load_config", return_value=cfg):
        return create_engine("test_llama")


def test_config_loader_llama_cpp():
    from engram.engine.llama_cpp_engine import LlamaCppEngine
    eng = _engine_from_type("llama_cpp")
    assert isinstance(eng, LlamaCppEngine)
    assert eng.n_gpu_layers == 20
    assert eng._max_context == 8192
    assert eng.gguf_path == "/models/q.gguf"


def test_config_loader_llama_cpp_hyphen():
    from engram.engine.llama_cpp_engine import LlamaCppEngine
    eng = _engine_from_type("llama-cpp")
    assert isinstance(eng, LlamaCppEngine)


def test_config_loader_llamacpp():
    from engram.engine.llama_cpp_engine import LlamaCppEngine
    eng = _engine_from_type("llamacpp")
    assert isinstance(eng, LlamaCppEngine)


# ---------------------------------------------------------------------------
# 3. generate(): response extraction and token clamping
# ---------------------------------------------------------------------------

def test_generate_returns_content():
    eng = _make_engine()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion("Hello!")
    eng.client = mock_client

    result = eng.generate("Say hello")
    assert result == "Hello!"
    assert eng._last_raw_response == "Hello!"


def test_generate_clamps_max_tokens():
    """max_tokens exceeding max_context should be silently clamped."""
    eng = _make_engine(max_context=512)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion("ok")
    eng.client = mock_client

    eng.generate("test", max_tokens=9999)
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["max_tokens"] <= 512


def test_generate_system_prompt_injected():
    eng = _make_engine()
    eng.system_prompt = "Be concise."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion("ok")
    eng.client = mock_client

    eng.generate("test")
    messages = mock_client.chat.completions.create.call_args[1]["messages"]
    assert messages[0]["role"] == "system"
    assert "Be concise" in messages[0]["content"]


def test_generate_no_client_raises():
    eng = _make_engine()
    eng.client = None
    try:
        eng.generate("test")
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "openai" in str(e).lower()


# ---------------------------------------------------------------------------
# 4. generate(): connection error propagates
# ---------------------------------------------------------------------------

def test_generate_connection_error_propagates():
    eng = _make_engine()
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ConnectionError("refused")
    eng.client = mock_client

    try:
        eng.generate("test")
        assert False, "Expected ConnectionError"
    except ConnectionError:
        pass


# ---------------------------------------------------------------------------
# 5–7. Properties
# ---------------------------------------------------------------------------

def test_max_context_length():
    eng = _make_engine(max_context=32768)
    assert eng.max_context_length == 32768


def test_count_tokens_returns_int():
    eng = _make_engine()
    n = eng.count_tokens("Hello world test")
    assert isinstance(n, int)
    assert n > 0


def test_is_cloud_false():
    assert _make_engine().is_cloud is False


# ---------------------------------------------------------------------------
# 8–10. supports_logprobs probe and caching
# ---------------------------------------------------------------------------

def test_supports_logprobs_true_when_present():
    eng = _make_engine()
    logprob_content = [SimpleNamespace(token="Hi", logprob=-0.1, top_logprobs=[])]
    mock_logprobs = SimpleNamespace(content=logprob_content)
    resp = _mock_completion("Hi", logprobs=mock_logprobs)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = resp
    eng.client = mock_client

    assert eng.supports_logprobs is True
    assert eng._supports_logprobs is True


def test_supports_logprobs_false_when_none():
    eng = _make_engine()
    resp = _mock_completion("Hi", logprobs=None)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = resp
    eng.client = mock_client

    assert eng.supports_logprobs is False


def test_supports_logprobs_false_on_error():
    eng = _make_engine()
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("400 bad request")
    eng.client = mock_client

    assert eng.supports_logprobs is False


def test_supports_logprobs_cached():
    """Probe should only be called once; subsequent accesses use cached value."""
    eng = _make_engine()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion("Hi", logprobs=None)
    eng.client = mock_client

    _ = eng.supports_logprobs
    _ = eng.supports_logprobs
    _ = eng.supports_logprobs

    assert mock_client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# 11–15. build_llama_cpp_launch_command
# ---------------------------------------------------------------------------

def test_launch_command_basic():
    from engram.engine.runtime_status import build_llama_cpp_launch_command
    cfg = {
        "type": "llama_cpp",
        "model": "qwen7b",
        "gguf_path": "/models/qwen7b.gguf",
        "base_url": "http://127.0.0.1:8080/v1",
        "max_context": 32768,
        "n_gpu_layers": 0,
    }
    cmd = build_llama_cpp_launch_command(cfg)
    assert cmd is not None
    assert "/models/qwen7b.gguf" in cmd
    assert "--host" in cmd
    assert "--port" in cmd
    assert "8080" in cmd
    assert "-c" in cmd
    assert "32768" in cmd
    assert "--n-gpu-layers" in cmd
    assert " 0 " in cmd or cmd.endswith(" 0")


def test_launch_command_no_gguf_path_returns_none():
    from engram.engine.runtime_status import build_llama_cpp_launch_command
    cfg = {"type": "llama_cpp", "model": "m", "base_url": "http://localhost:8080/v1"}
    assert build_llama_cpp_launch_command(cfg) is None


def test_launch_command_wrong_type_returns_none():
    from engram.engine.runtime_status import build_llama_cpp_launch_command
    cfg = {"type": "vllm", "model": "m", "gguf_path": "/m.gguf"}
    assert build_llama_cpp_launch_command(cfg) is None


def test_launch_command_n_gpu_layers_split():
    from engram.engine.runtime_status import build_llama_cpp_launch_command
    cfg = {
        "type": "llama_cpp",
        "gguf_path": "/m.gguf",
        "base_url": "http://127.0.0.1:8081/v1",
        "max_context": 16384,
        "n_gpu_layers": 40,
    }
    cmd = build_llama_cpp_launch_command(cfg)
    assert "40" in cmd
    assert "8081" in cmd


def test_launch_command_extra_args():
    from engram.engine.runtime_status import build_llama_cpp_launch_command
    cfg = {
        "type": "llama_cpp",
        "gguf_path": "/m.gguf",
        "base_url": "http://127.0.0.1:8080/v1",
        "max_context": 4096,
        "n_gpu_layers": 0,
        "launch": {"extra_args": ["--threads", "8"]},
    }
    cmd = build_llama_cpp_launch_command(cfg)
    assert "--threads" in cmd
    assert "8" in cmd


# ---------------------------------------------------------------------------
# 16. classify_llama_cpp_generation_failure
# ---------------------------------------------------------------------------

def test_classify_not_running():
    from engram.engine.runtime_status import classify_llama_cpp_generation_failure
    assert classify_llama_cpp_generation_failure("Connection refused") == "not_running"
    assert classify_llama_cpp_generation_failure("Cannot connect to host") == "not_running"
    assert classify_llama_cpp_generation_failure("failed to connect") == "not_running"


def test_classify_context_overflow():
    from engram.engine.runtime_status import classify_llama_cpp_generation_failure
    assert classify_llama_cpp_generation_failure("prompt is too long for context") == "context_overflow"
    assert classify_llama_cpp_generation_failure("tokens exceed context window") == "context_overflow"
    assert classify_llama_cpp_generation_failure("KV cache exhausted") == "context_overflow"


def test_classify_engine_crashed():
    from engram.engine.runtime_status import classify_llama_cpp_generation_failure
    assert classify_llama_cpp_generation_failure("500 internal server error") == "engine_crashed"
    assert classify_llama_cpp_generation_failure("llama_decode failed") == "engine_crashed"
    assert classify_llama_cpp_generation_failure("slot error occurred") == "engine_crashed"


def test_classify_generation_failed_fallback():
    from engram.engine.runtime_status import classify_llama_cpp_generation_failure
    assert classify_llama_cpp_generation_failure("some unknown error xyz") == "generation_failed"
    assert classify_llama_cpp_generation_failure("") == "generation_failed"


# ---------------------------------------------------------------------------
# 17. build_llama_cpp_recovery_guidance
# ---------------------------------------------------------------------------

def test_recovery_guidance_not_running():
    from engram.engine.runtime_status import build_llama_cpp_recovery_guidance
    cfg = {
        "type": "llama_cpp",
        "gguf_path": "/m.gguf",
        "base_url": "http://127.0.0.1:8080/v1",
        "max_context": 4096,
        "n_gpu_layers": 0,
    }
    guidance = build_llama_cpp_recovery_guidance("not_running", cfg)
    assert "llama-server" in guidance.lower() or "not running" in guidance.lower()
    assert "127.0.0.1:8080" in guidance or "/m.gguf" in guidance


def test_recovery_guidance_context_overflow_mentions_options():
    from engram.engine.runtime_status import build_llama_cpp_recovery_guidance
    guidance = build_llama_cpp_recovery_guidance("context_overflow", {
        "type": "llama_cpp", "gguf_path": "/m.gguf",
        "base_url": "http://127.0.0.1:8080/v1", "max_context": 4096, "n_gpu_layers": 0,
    })
    assert "context" in guidance.lower() or "4096" in guidance


def test_recovery_guidance_engine_crashed_mentions_gpu_layers():
    from engram.engine.runtime_status import build_llama_cpp_recovery_guidance
    guidance = build_llama_cpp_recovery_guidance("engine_crashed", {
        "type": "llama_cpp", "gguf_path": "/m.gguf",
        "base_url": "http://127.0.0.1:8080/v1", "max_context": 4096, "n_gpu_layers": 40,
    })
    assert "gpu" in guidance.lower() or "n_gpu_layers" in guidance.lower()


# ---------------------------------------------------------------------------
# 18–20. Reasoning visibility
# ---------------------------------------------------------------------------

def _gen_with_visibility(content: str, visibility: str) -> str:
    eng = _make_engine(reasoning_visibility=visibility)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion(content)
    eng.client = mock_client
    return eng.generate("test")


def test_reasoning_auto_strips_think_block():
    raw = "<think>Let me think step by step...</think>\n\nThe answer is 42."
    result = _gen_with_visibility(raw, "auto")
    assert "<think>" not in result
    assert "42" in result


def test_reasoning_show_preserves_think_block():
    raw = "<think>reasoning here</think>\n\nAnswer: yes."
    result = _gen_with_visibility(raw, "show")
    assert "<think>" in result


def test_reasoning_hide_always_strips():
    raw = "<think>internal plan</think>\n\nFinal answer."
    result = _gen_with_visibility(raw, "hide")
    assert "<think>" not in result
    assert "Final answer" in result


def test_reasoning_auto_no_change_when_clean():
    raw = "This is a straightforward response."
    result = _gen_with_visibility(raw, "auto")
    assert result == raw


# ---------------------------------------------------------------------------
# 21. get_diagnostics
# ---------------------------------------------------------------------------

def test_get_diagnostics_keys():
    eng = _make_engine(gguf_path="/m.gguf", n_gpu_layers=20)
    d = eng.get_diagnostics()
    for key in ("engine", "model_name", "base_url", "gguf_path",
                "n_gpu_layers", "max_context", "reasoning_visibility",
                "last_reasoning_detected", "supports_logprobs"):
        assert key in d, f"Missing key: {key}"
    assert d["engine"] == "llama_cpp"
    assert d["n_gpu_layers"] == 20
    assert d["gguf_path"] == "/m.gguf"


# ---------------------------------------------------------------------------
# 22. Config round-trip: failover profile with llama_cpp
# ---------------------------------------------------------------------------

def test_config_roundtrip_failover_profile():
    """Failover engine can include a llama_cpp backend without error."""
    from engram.engine.config_loader import create_failover_engine
    cfg = {
        "engines": {
            "llama7b": {
                "type": "llama_cpp",
                "model": "q7b",
                "base_url": "http://127.0.0.1:8080/v1",
                "gguf_path": "/m.gguf",
                "max_context": 8192,
                "n_gpu_layers": 0,
                "timeout": 60,
                "compression_strategy": "compress",
                "max_retries": 1,
            }
        },
        "profiles": {
            "cpu_only": {
                "engines": ["llama7b"],
                "allow_cloud_failover": False,
                "max_attempts": 2,
            }
        }
    }
    with patch("engram.engine.config_loader.load_config", return_value=cfg):
        fe = create_failover_engine("cpu_only")
    from engram.engine.router import FailoverEngine
    assert isinstance(fe, FailoverEngine)
    assert len(fe.engines) == 1
    from engram.engine.llama_cpp_engine import LlamaCppEngine
    assert isinstance(fe.engines[0], LlamaCppEngine)
