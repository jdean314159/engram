from __future__ import annotations

import json
from pathlib import Path

import yaml

from engram.engine.base import LLMEngine, LogprobResult
from engram.engine.config_loader import create_engine, create_failover_engine
from engram.engine.model_discovery import match_discovered_model, resolve_vllm_model, DiscoveryResolution
from apps.sandbox import runtime_manager


class DummyEngine(LLMEngine):
    def generate(self, *args, **kwargs):
        return "ok"

    def generate_with_logprobs(self, *args, **kwargs):
        return LogprobResult(text="ok", token_logprobs=[])

    async def stream(self, *args, **kwargs):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def max_context_length(self) -> int:
        return 8192


def _write_cfg(tmp_path: Path) -> Path:
    cfg = {
        "engines": {
            "local_a": {"type": "anthropic", "model": "claude-x"},
            "local_b": {"type": "anthropic", "model": "claude-y"},
            "vllm_qwen": {
                "type": "vllm",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "base_url": "http://localhost:8000/v1",
                "model_resolution": "prefer_discovered",
                "model_match": "suffix_or_exact",
            },
            "managed_vllm": {
                "type": "vllm",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "base_url": "http://localhost:8000/v1",
                "managed": True,
                "start_command": "python -m http.server 9999",
            },
        },
        "profiles": {
            "test_profile": {
                "engines": ["local_a", "local_b"],
                "allow_cloud_failover": False,
                "cloud_policy": "query_only",
            },
            "managed_profile": {
                "engines": ["managed_vllm"],
            },
        },
    }
    path = tmp_path / "llm_engines.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_failover_override_order_and_cloud_overrides(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)

    def fake_create_engine(name, config_path=None, engine_config_override=None):
        return DummyEngine(model_name=name)

    monkeypatch.setattr("engram.engine.config_loader.create_engine", fake_create_engine)

    eng = create_failover_engine(
        "test_profile",
        config_path=str(cfg_path),
        override_engines=["local_b", "local_a"],
        override_allow_cloud_failover=True,
        override_cloud_policy="full_context",
    )

    assert [e.model_name for e in eng.engines] == ["local_b", "local_a"]
    assert eng.policy.allow_cloud_failover is True
    assert eng.policy.cloud_policy == "full_context"


def test_match_and_resolve_discovered_vllm_model(monkeypatch):
    assert match_discovered_model("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct")
    assert match_discovered_model("org/model-awq", "model")
    assert not match_discovered_model("foo/bar", "baz", strategy="exact")

    monkeypatch.setattr(
        "engram.engine.model_discovery.list_vllm_models",
        lambda base_url, timeout_s=2.0: [type("D", (), {"id": "Qwen2.5-7B-Instruct"})()],
    )
    resolved = resolve_vllm_model(
        base_url="http://localhost:8000/v1",
        configured_model="Qwen/Qwen2.5-7B-Instruct",
    )
    assert isinstance(resolved, DiscoveryResolution)
    assert resolved.resolved_model == "Qwen2.5-7B-Instruct"
    assert resolved.source == "discovered"


def test_create_engine_uses_discovered_vllm_model(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)

    monkeypatch.setattr(
        "engram.engine.config_loader.resolve_vllm_model",
        lambda **kwargs: DiscoveryResolution(
            requested_model="Qwen/Qwen2.5-7B-Instruct",
            resolved_model="Qwen2.5-7B-Instruct",
            source="discovered",
            match_strategy="suffix_or_exact",
            discovered_ids=["Qwen2.5-7B-Instruct"],
        ),
    )

    eng = create_engine("vllm_qwen", config_path=str(cfg_path))
    assert eng.model_name == "Qwen2.5-7B-Instruct"
    assert eng.configured_model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert eng.discovery_resolution["source"] == "discovered"


def test_runtime_manager_tracks_sandbox_owned_process(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    state_path = tmp_path / "runtime_state.json"
    monkeypatch.setattr(runtime_manager, "_state_path", lambda: state_path)

    class FakeProc:
        pid = 4242

    started = {}

    def fake_popen(*args, **kwargs):
        started["args"] = args
        return FakeProc()

    monkeypatch.setattr(runtime_manager.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runtime_manager, "_is_pid_running", lambda pid: pid == 4242)
    monkeypatch.setattr(runtime_manager, "_healthcheck", lambda url, timeout_s=1.0: False)

    status = runtime_manager.start_managed_engine("managed_vllm", config_path=str(cfg_path))
    assert status.running is True
    assert status.ownership == "sandbox-managed"
    assert json.loads(state_path.read_text())["managed_vllm"]["pid"] == 4242

    killed = {}
    monkeypatch.setattr(runtime_manager.os, "kill", lambda pid, sig: killed.update({"pid": pid, "sig": sig}))
    monkeypatch.setattr(runtime_manager, "_is_pid_running", lambda pid: False)

    stopped = runtime_manager.stop_managed_engine("managed_vllm", config_path=str(cfg_path))
    assert killed["pid"] == 4242
    assert stopped.running is False
