"""Managed local runtime helpers for the sandbox app.

This module intentionally keeps ownership semantics simple:
- If an engine config marks itself as managed, the sandbox may start it.
- The sandbox will only stop processes it started itself.
- User-managed servers are discovered and shown, but never killed implicitly.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from engram.engine.config_loader import load_config


@dataclass
class ManagedRuntimeStatus:
    engine_name: str
    managed: bool
    running: bool
    ownership: str
    pid: Optional[int]
    health_url: Optional[str]
    detail: str = ""


def _state_path() -> Path:
    p = Path("~/.engram/runtime_state.json").expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_state() -> Dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    _state_path().write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _engine_config(engine_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_config(config_path)
    engines = cfg.get("engines") or {}
    if engine_name not in engines:
        raise ValueError(f"Unknown engine: {engine_name}")
    return dict(engines[engine_name] or {})


def _health_url(engine_cfg: Dict[str, Any]) -> Optional[str]:
    base_url = str(engine_cfg.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        return None
    etype = str(engine_cfg.get("type") or "").lower()
    if etype == "ollama":
        return base_url.replace("/v1", "") + "/api/tags"
    if etype in {"vllm", "llama_cpp", "llama-cpp", "llamacpp",
                 "openai-compatible", "openai_compatible"}:
        return base_url + "/models"
    return None


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _healthcheck(url: Optional[str], timeout_s: float = 1.0) -> bool:
    if not url:
        return False
    try:
        r = requests.get(url, timeout=timeout_s)
        return r.ok
    except Exception:
        return False


def get_managed_engine_status(engine_name: str, config_path: Optional[str] = None) -> ManagedRuntimeStatus:
    cfg = _engine_config(engine_name, config_path=config_path)
    state = _load_state().get(engine_name) or {}
    managed = bool(cfg.get("managed", False))
    pid = state.get("pid")
    health_url = str(cfg.get("healthcheck_url") or _health_url(cfg) or "") or None
    running = False
    ownership = "user-managed"
    detail = ""

    if pid and _is_pid_running(int(pid)):
        running = True
        ownership = "sandbox-managed"
        detail = f"Sandbox started PID {pid}."
    elif _healthcheck(health_url):
        running = True
        ownership = "user-managed"
        detail = "Reachable via health check."
    elif managed:
        detail = "Managed engine is stopped."
    else:
        detail = "No reachable server detected."

    return ManagedRuntimeStatus(
        engine_name=engine_name,
        managed=managed,
        running=running,
        ownership=ownership,
        pid=int(pid) if pid else None,
        health_url=health_url,
        detail=detail,
    )


def start_managed_engine(engine_name: str, config_path: Optional[str] = None) -> ManagedRuntimeStatus:
    cfg = _engine_config(engine_name, config_path=config_path)
    if not bool(cfg.get("managed", False)):
        raise ValueError(f"Engine '{engine_name}' is not marked managed in config")

    cmd = str(cfg.get("start_command") or "").strip()

    # For llama_cpp engines, auto-build the launch command if start_command is absent
    if not cmd:
        etype = str(cfg.get("type") or "").lower()
        if etype in {"llama_cpp", "llama-cpp", "llamacpp"}:
            from engram.engine.runtime_status import build_llama_cpp_launch_command
            cmd = build_llama_cpp_launch_command(cfg) or ""
            if not cmd:
                raise ValueError(
                    f"Engine '{engine_name}' is llama_cpp but has no gguf_path set. "
                    "Set gguf_path in llm_engines.yaml to enable managed start."
                )

    if not cmd:
        raise ValueError(f"Engine '{engine_name}' has no start_command")

    status = get_managed_engine_status(engine_name, config_path=config_path)
    if status.running:
        return status

    proc = subprocess.Popen(
        cmd if os.name == "nt" else shlex.split(cmd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        shell=(os.name == "nt"),
        start_new_session=True,
    )

    state = _load_state()
    state[engine_name] = {
        "pid": proc.pid,
        "started_at": time.time(),
        "ownership": "sandbox-managed",
        "start_command": cmd,
    }
    _save_state(state)
    return get_managed_engine_status(engine_name, config_path=config_path)


def stop_managed_engine(engine_name: str, config_path: Optional[str] = None) -> ManagedRuntimeStatus:
    cfg = _engine_config(engine_name, config_path=config_path)
    state = _load_state()
    info = state.get(engine_name) or {}
    pid = info.get("pid")
    if not pid:
        return get_managed_engine_status(engine_name, config_path=config_path)

    stop_cmd = str(cfg.get("stop_command") or "").strip()
    if stop_cmd:
        subprocess.run(
            stop_cmd if os.name == "nt" else shlex.split(stop_cmd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            shell=(os.name == "nt"),
            check=False,
        )
    else:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            pass

    state.pop(engine_name, None)
    _save_state(state)
    return get_managed_engine_status(engine_name, config_path=config_path)


def describe_profile_runtime(profile_name: str, config_path: Optional[str] = None) -> list[ManagedRuntimeStatus]:
    cfg = load_config(config_path)
    profile = (cfg.get("profiles") or {}).get(profile_name) or {}
    names = list(profile.get("engines") or [])
    return [get_managed_engine_status(name, config_path=config_path) for name in names]
