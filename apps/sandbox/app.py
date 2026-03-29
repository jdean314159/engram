"""Engram Sandbox reference app (lightweight, portable).

Design goals:
- One default project (per user), easy to start.
- Expose key configuration options with hover help.
- Provide a chat sandbox for testing.
- Make it easy for a moderately-skilled developer to adapt.

Run:
    streamlit run apps/sandbox/app.py

Install:
    pip install "engram[ui,engines]"
    # plus any memory layer extras you use, e.g.:
    # pip install "engram[episodic,semantic]"
"""

from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Any, Dict, List, Optional
import shutil
import subprocess

import streamlit as st

from engram.engine import load_config
from engram.utils.logging_setup import setup_logging_if_needed
from engram.engine.model_discovery import list_ollama_models, list_vllm_models

from apps.sandbox.config_store import UIConfig, load_ui_config, save_ui_config
from apps.sandbox.engine_factory import RuntimeOverrides, create_project_memory
from apps.sandbox.session_state import UISession, ChatMessage
from apps.sandbox.diagnostics_bridge import doctor as doctor_call, recommend as recommend_call
from apps.sandbox.model_management import render_model_management
from apps.sandbox.runtime_manager import describe_profile_runtime, start_managed_engine, stop_managed_engine
from apps.sandbox.memory_inspector import render_memory_tab
from engram.engine.runtime_status import (
    build_vllm_launch_command,
    classify_vllm_generation_failure,
    probe_vllm_runtime_state,
    build_recovery_guidance,
    build_llama_cpp_launch_command,
    classify_llama_cpp_generation_failure,
    build_llama_cpp_recovery_guidance,
)


APP_TITLE = "Engram — Memory System Sandbox"
DEFAULT_PROJECT_NAME = "default_project"


def _infer_artifact_family(engine_cfg: Dict[str, Any]) -> str:
    """Infer quantization family (awq/gptq/gguf/fp16) from engine config."""
    parts = " ".join(filter(None, [
        str(engine_cfg.get("model") or ""),
        str(engine_cfg.get("hf_repo_id") or ""),
        str(engine_cfg.get("local_model_dir") or ""),
        str((engine_cfg.get("launch") or {}).get("source") or ""),
    ])).lower()
    if "awq" in parts:
        return "awq"
    if "gptq" in parts:
        return "gptq"
    if "gguf" in parts:
        return "gguf"
    return "fp16"


def _get_selected_primary_engine_cfg(engine_cfg: Dict[str, Any], ui_cfg) -> Optional[Dict[str, Any]]:
    profiles = engine_cfg.get("profiles", {}) or {}
    engines = engine_cfg.get("engines", {}) or {}
    profile = profiles.get(ui_cfg.profile, {}) if getattr(ui_cfg, "profile", None) else {}
    order = profile.get("engines", []) or profile.get("engine_order", []) or []
    if not order:
        return None
    return engines.get(order[0])
    
def _render_vllm_not_running_message(engine_name: str, engine: Dict[str, Any], total_vram_gb: Optional[float]) -> None:
    endpoint = str(engine.get("base_url") or "http://localhost:8000/v1")
    cmd = build_vllm_launch_command(engine) or "(no launch command available; check launch.source / hf_repo_id / local_model_dir)"

    st.warning(
        f"This profile uses a vLLM-backed engine (`{engine_name}`), but no vLLM server is reachable at `{endpoint}`."
    )
    st.markdown("**Start vLLM with:**")
    st.code(cmd, language="bash")

    family = _infer_artifact_family(engine)
    notes = [
        "Start vLLM before chatting with this engine.",
        "Only one model can be served on this endpoint at a time.",
        "If another GPU-backed runtime is already using the GPU, this command may fail with OOM.",
    ]
    if family == "awq":
        notes.append("This command assumes an AWQ checkpoint.")
    elif family == "gptq":
        notes.append("This command assumes a GPTQ checkpoint.")

    for note in notes:
        st.caption(note)    


def _render_vllm_failure_message(
    engine_name: str,
    engine: Dict[str, Any],
    failure: Dict[str, Any],
    total_vram_gb: Optional[float],
    technical_error: str,
) -> None:
    endpoint = str(failure.get("endpoint") or engine.get("base_url") or "http://localhost:8000/v1")
    expected_model = str(failure.get("expected_model") or engine.get("model") or "")
    served_models = failure.get("served_models") or []
    cmd = build_vllm_launch_command(engine) or "(no launch command available; check launch.source / hf_repo_id / local_model_dir)"

    kind = str(failure.get("failure_kind") or "unknown")
    if kind == "not_running":
        st.error(f"vLLM server for `{engine_name}` is not reachable.")
    elif kind == "wrong_model":
        st.error(f"vLLM server for `{engine_name}` is running, but the expected model is not loaded.")
    elif kind == "engine_crashed":
        st.error(f"vLLM server for `{engine_name}` appears to have failed during generation.")
    elif kind == "token_budget_exceeded":
        st.error(
            f"vLLM server for `{engine_name}` rejected the request: output tokens exceed the server's context limit. "
            "Engram will retry automatically with a clamped value. If this persists, reduce `max_tokens` in `llm_engines.yaml` "
            "or restart vLLM with a larger `--max-model-len`."
        )
    else:
        st.error(f"Generation failed for vLLM engine `{engine_name}`.")

    if expected_model:
        st.caption(f"Expected served model: `{expected_model}`")
    if endpoint:
        st.caption(f"Endpoint: `{endpoint}`")

    if served_models:
        with st.expander("Discovered served models", expanded=False):
            for model_id in served_models:
                st.write(f"`{model_id}`")

    st.markdown("**Launch / restart command**")
    st.code(cmd, language="bash")

    guidance = build_recovery_guidance(
        backend=str(engine.get("type") or ""),
        failure_kind=kind,
        launch_command=cmd,
    )

    st.markdown("**Suggested next step**")
    for step in guidance.get("steps", []):
        st.caption(step)

    advanced = guidance.get("advanced_steps", [])
    if advanced:
        with st.expander("Advanced recovery", expanded=False):
            for step in advanced:
                st.write(f"- {step}")

    with st.expander("Technical details", expanded=False):
        st.code(technical_error or "No technical details available.", language="text")

def _gpu_runtime_summary() -> Dict[str, Any]:
    """
    Very small nvidia-smi summary:
    - whether nvidia-smi is available
    - total / used memory for GPU 0
    - compute process count
    """
    result = {
        "available": False,
        "used_mb": None,
        "total_mb": None,
        "process_count": 0,
        "detail": "",
    }

    if shutil.which("nvidia-smi") is None:
        result["detail"] = "nvidia-smi not found"
        return result

    try:
        mem_proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if mem_proc.returncode != 0:
            result["detail"] = (mem_proc.stderr or mem_proc.stdout or "").strip()
            return result

        first = (mem_proc.stdout or "").strip().splitlines()[0]
        used_mb_str, total_mb_str = [x.strip() for x in first.split(",")]
        result["used_mb"] = int(used_mb_str)
        result["total_mb"] = int(total_mb_str)

        proc_proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc_proc.returncode == 0:
            pids = [ln.strip() for ln in (proc_proc.stdout or "").splitlines() if ln.strip()]
            result["process_count"] = len(pids)

        result["available"] = True
        return result
    except Exception as e:
        result["detail"] = str(e)
        return result

def _default_project_dir() -> Path:
    # Keep UI data under a predictable per-user directory inside the project folder.
    # Users can change this easily by editing this function.
    return Path.home() / ".engram" / DEFAULT_PROJECT_NAME

def _get_profile_engine_cfgs(engine_cfg: Dict[str, Any], ui_cfg) -> list[tuple[str, Dict[str, Any]]]:
    profiles = engine_cfg.get("profiles", {}) or {}
    profile = profiles.get(getattr(ui_cfg, "profile", None), {}) if getattr(ui_cfg, "profile", None) else {}
    order = profile.get("engines", []) or profile.get("engine_order", []) or []

    engines = engine_cfg.get("engines", {}) or {}
    out: list[tuple[str, Dict[str, Any]]] = []
    for name in order:
        cfg = engines.get(name)
        if cfg:
            out.append((name, cfg))
    return out
    
def _find_relevant_vllm_failure(engine_cfg: Dict[str, Any], ui_cfg, ex: Exception) -> tuple[str, Dict[str, Any], Dict[str, Any]] | None:
    candidates = _get_profile_engine_cfgs(engine_cfg, ui_cfg)

    vllm_like = []
    llama_cpp_like = []
    for engine_name, cfg in candidates:
        etype = str(cfg.get("type") or "").lower().strip()
        if etype in {"vllm", "openai-compatible", "openai_compatible"}:
            vllm_like.append((engine_name, cfg))
        elif etype in {"llama_cpp", "llama-cpp", "llamacpp"}:
            llama_cpp_like.append((engine_name, cfg))

    for engine_name, cfg in vllm_like:
        failure = classify_vllm_generation_failure(
            engine_name,
            cfg,
            ex,
            list_vllm_models,
        )
        if failure.get("failure_kind") in {"not_running", "wrong_model", "engine_crashed", "generation_failed", "token_budget_exceeded"}:
            return engine_name, cfg, failure

    # Check llama.cpp engines for connection errors
    err_text = str(ex)
    for engine_name, cfg in llama_cpp_like:
        kind = classify_llama_cpp_generation_failure(err_text)
        if kind in {"not_running", "context_overflow", "engine_crashed", "generation_failed"}:
            return engine_name, cfg, {
                "failure_kind": kind,
                "endpoint": cfg.get("base_url", ""),
                "backend": "llama_cpp",
            }

    return None


def _is_cuda_runtime_init_error(exc: Exception) -> bool:
    text = str(exc).lower()
    needles = (
        "cuda error",
        "cudaerrorunknown",
        "cuda unknown error",
        "failed call to cuinit",
    )
    return any(n in text for n in needles)


def _render_llama_cpp_not_running_message(engine_name: str, engine: Dict[str, Any]) -> None:
    """Show 'llama-server not running' banner with launch command."""
    endpoint = str(engine.get("base_url") or "http://127.0.0.1:8080/v1")
    cmd = build_llama_cpp_launch_command(engine) or (
        "(no launch command available; set gguf_path in llm_engines.yaml)"
    )
    n_gpu_layers = int(engine.get("n_gpu_layers") or 0)
    gpu_note = (
        "full CPU execution (n_gpu_layers=0)" if n_gpu_layers == 0
        else f"split GPU+CPU ({n_gpu_layers} layers on GPU)" if n_gpu_layers < 999
        else "full GPU execution"
    )

    st.warning(
        f"This profile uses a llama.cpp-backed engine (`{engine_name}`), "
        f"but no llama-server is reachable at `{endpoint}`."
    )
    st.caption(f"GPU offload: {gpu_note}")
    st.markdown("**Start llama-server with:**")
    st.code(cmd, language="bash")
    st.caption("Install llama-cpp-python[server] or build llama.cpp from source if llama-server is not yet installed.")


def _render_llama_cpp_failure_message(
    engine_name: str,
    engine: Dict[str, Any],
    failure: Dict[str, Any],
    technical_error: str,
) -> None:
    """Render llama.cpp failure banner with guidance."""
    kind = str(failure.get("failure_kind") or "generation_failed")
    endpoint = str(failure.get("endpoint") or engine.get("base_url") or "")
    cmd = build_llama_cpp_launch_command(engine) or "(set gguf_path in llm_engines.yaml)"
    n_gpu_layers = int(engine.get("n_gpu_layers") or 0)

    if kind == "not_running":
        st.error(f"llama-server for `{engine_name}` is not reachable at `{endpoint}`.")
    elif kind == "context_overflow":
        st.error(f"llama-server for `{engine_name}` rejected the request: prompt exceeds context window.")
    elif kind == "engine_crashed":
        st.error(f"llama-server for `{engine_name}` returned a server error.")
        if n_gpu_layers > 0:
            st.caption(
                f"This may be a GPU OOM crash. Try reducing `n_gpu_layers` below {n_gpu_layers} in llm_engines.yaml."
            )
    else:
        st.error(f"Generation failed for llama.cpp engine `{engine_name}`.")

    if endpoint:
        st.caption(f"Endpoint: `{endpoint}`")

    st.markdown("**Launch / restart command**")
    st.code(cmd, language="bash")

    guidance = build_llama_cpp_recovery_guidance(kind, engine)
    st.markdown("**Suggested next step**")
    st.caption(guidance)

    with st.expander("Technical details", expanded=False):
        st.code(technical_error, language="text")


def _render_cuda_runtime_help(exc: Exception) -> None:
    st.error(
        "Failed to create engine because CUDA could not be initialized in this process.\n\n"
        "This can happen after Linux + NVIDIA suspend/resume even when the GPU is still visible to the driver.\n\n"
        f"Original error: {exc}"
    )
    st.info(
        "Recovery steps:\n"
        "1. Close GPU-using apps such as Streamlit, Chrome, Ollama, or vLLM.\n"
        "2. Reload the NVIDIA UVM module: `sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm`\n"
        "3. Verify PyTorch CUDA again, then restart Engram.\n"
        "4. If the module cannot be unloaded or CUDA still fails, reboot.\n\n"
        "Go to the **Models** tab to review configured engines and endpoints."
    )

def _load_engine_config() -> Dict[str, Any]:
    """Load llm_engines.yaml as a single dict.

    The UI needs both profiles and engine definitions to provide helpful guidance
    (e.g., warning when a selected profile includes a "reasoning"/"thinking" model).
    """
    return load_config(None) or {}


def _profile_looks_like_thinking(profile_name: str, profile_cfg: Dict[str, Any], engines_cfg: Dict[str, Any]) -> bool:
    """Best-effort heuristic: flag profiles likely to use "thinking" models.

    We avoid a hard dependency on a specific naming convention. Instead, we check:
    - profile name
    - engine name
    - model id/tag
    """

    def _has_hint(s: str) -> bool:
        s = (s or "").lower()
        return any(k in s for k in ("reason", "reasoning", "think", "thinking", "deep"))

    if _has_hint(profile_name):
        return True

    engine_names = profile_cfg.get("engines") or []
    for eng_name in engine_names:
        if _has_hint(str(eng_name)):
            return True
        eng = engines_cfg.get(str(eng_name)) or {}
        if _has_hint(str(eng.get("model", ""))):
            return True
    return False


def _resolve_current_profile_order(ui_cfg: UIConfig, engine_cfg: Dict[str, Any]) -> List[str]:
    profiles = engine_cfg.get("profiles", {}) or {}
    prof = profiles.get(ui_cfg.profile) or {}
    order = [s.strip() for s in (ui_cfg.engine_order or "").split(",") if s.strip()]
    return order or list(prof.get("engines") or [])
    
def _collect_endpoint_rows(engines_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build one row per unique endpoint.
    """
    endpoint_map: Dict[str, Dict[str, Any]] = {}

    for engine_name, ecfg in (engines_cfg or {}).items():
        backend = str(ecfg.get("type") or "").strip()
        endpoint = str(ecfg.get("base_url") or "").strip()

        if not endpoint:
            continue

        key = endpoint.rstrip("/")
        if key not in endpoint_map:
            endpoint_map[key] = {
                "backend": backend,
                "endpoint": key,
                "reachable": False,
                "served_models": [],
                "detail": "",
                "engine_names": [],
            }

        endpoint_map[key]["engine_names"].append(engine_name)

    rows: List[Dict[str, Any]] = []
    for row in endpoint_map.values():
        backend = row["backend"].lower()
        endpoint = row["endpoint"]

        try:
            if backend == "ollama":
                models = list_ollama_models(endpoint.replace("/v1", "").rstrip("/"))
                row["reachable"] = True
                row["served_models"] = [m.id for m in models]
            elif backend in {"vllm", "openai", "openai-compatible", "openai_compatible",
                              "llama_cpp", "llama-cpp", "llamacpp"}:
                models = list_vllm_models(endpoint)
                row["reachable"] = True
                row["served_models"] = [m.id for m in models]
            else:
                row["detail"] = f"Unsupported backend for endpoint probing: {backend}"
        except Exception as e:
            row["reachable"] = False
            row["detail"] = str(e)

        rows.append(row)

    return rows    
    
def _normalize_backend_label(backend: str) -> str:
    b = (backend or "").lower().strip()
    if b == "ollama":
        return "Ollama"
    if b in {"vllm", "openai-compatible", "openai_compatible"}:
        return "vLLM"
    if b == "openai":
        return "OpenAI"
    if b in {"llama_cpp", "llama-cpp", "llamacpp"}:
        return "llama.cpp"
    return backend or "Unknown"


def _status_icon(status: str) -> str:
    s = (status or "").lower().strip()
    if s in {"reachable", "ok", "available", "served"}:
        return "🟢"
    if s in {"warning", "degraded"}:
        return "🟡"
    if s in {"unreachable", "failed", "missing"}:
        return "🔴"
    return "⚪"


def _friendly_endpoint_error(endpoint: str, err_text: str) -> str:
    text = (err_text or "").lower()

    if "401" in text or "unauthorized" in text:
        return "API key missing or invalid."

    if "connection refused" in text or "failed to establish a new connection" in text:
        if "8000" in endpoint:
            return "vLLM server not running."
        if "11434" in endpoint:
            return "Ollama service not running."
        if "8080" in endpoint or "8081" in endpoint:
            return "llama-server not running."
        return "Endpoint is not running."

    if "timed out" in text or "timeout" in text:
        return "Endpoint timed out."

    return "Endpoint unavailable."


def _maybe_render_model_list(models: List[str], max_inline: int = 3) -> None:
    models = models or []
    if not models:
        st.caption("No models reported.")
        return

    if len(models) <= max_inline:
        st.caption(", ".join(models))
        return

    st.caption(f"{len(models)} models available")
    with st.expander("Show models", expanded=False):
        for m in models:
            st.write(f"`{m}`")


def _availability_label_for_backend(backend: str) -> str:
    b = (backend or "").lower().strip()
    if b == "ollama":
        return "available"
    if b in {"vllm", "openai", "openai-compatible", "openai_compatible",
             "llama_cpp", "llama-cpp", "llamacpp"}:
        return "served"
    return "running"


def _engine_summary_row(engine_name: str, engines_cfg: Dict[str, Any]) -> Dict[str, str]:
    e = engines_cfg.get(engine_name) or {}
    return {
        "name": engine_name or "(none)",
        "backend": str(e.get("type") or "?"),
        "endpoint": str(e.get("base_url") or ""),
        "model": str(e.get("model") or ""),
    }

def _nvidia_smi_status() -> tuple[str, str]:
    """Return a simple health/status check for nvidia-smi itself."""
    if shutil.which("nvidia-smi") is None:
        return ("Unavailable", "nvidia-smi not found on PATH")

    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as e:
        return ("Unavailable", f"nvidia-smi execution failed: {e}")

    if proc.returncode == 0:
        first_line = (proc.stdout or "").strip().splitlines()
        detail = first_line[0] if first_line else "nvidia-smi responded"
        return ("OK", detail)

    err = (proc.stderr or proc.stdout or "").strip()
    return ("Failed", err or f"nvidia-smi exited with code {proc.returncode}")

def _sidebar_health_snapshot(engine_cfg: Dict[str, Any], ui_cfg: UIConfig) -> List[tuple[str, str]]:
    rows: List[tuple[str, str]] = []
    repo_root = Path(__file__).resolve().parents[2]
    rows.append(("Repo", str(repo_root)))
    rows.append(("Config", str(Path("~/.engram/llm_engines.yaml").expanduser())))

    try:
        import torch
        rows.append(("PyTorch", "OK"))
        rows.append(("CUDA", "OK" if torch.cuda.is_available() else "Unavailable"))
    except Exception:
        rows.append(("PyTorch", "Unavailable"))
        rows.append(("CUDA", "Unavailable"))

    smi_status, _smi_detail = _nvidia_smi_status()
    rows.append(("nvidia-smi", smi_status))

    current_order = _resolve_current_profile_order(ui_cfg, engine_cfg)
    if current_order:
        first = _engine_summary_row(current_order[0], engine_cfg.get("engines", {}) or {})
        backend = first["backend"].lower()
        endpoint = first["endpoint"]

        if backend == "ollama" and endpoint:
            try:
                list_ollama_models(endpoint.replace("/v1", "").rstrip("/"))
                rows.append(("Ollama", "Reachable"))
            except Exception:
                rows.append(("Ollama", "Unreachable"))

        elif backend in {"vllm", "openai", "openai-compatible", "openai_compatible"} and endpoint:
            try:
                list_vllm_models(endpoint)
                rows.append(("vLLM", "Reachable"))
            except Exception:
                rows.append(("vLLM", "Unreachable"))

        elif backend in {"llama_cpp", "llama-cpp", "llamacpp"} and endpoint:
            try:
                list_vllm_models(endpoint)  # llama-server speaks OpenAI /v1/models
                rows.append(("llama.cpp", "Reachable"))
            except Exception:
                rows.append(("llama.cpp", "Unreachable"))

    return rows


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    setup_logging_if_needed()

    st.title(APP_TITLE)

    # ------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------
    if "ui_cfg" not in st.session_state:
        st.session_state.ui_cfg = load_ui_config()
    if "session" not in st.session_state:
        st.session_state.session = UISession()
    if "pm" not in st.session_state:
        st.session_state.pm = None  # lazily created after selecting profile

    ui_cfg: UIConfig = st.session_state.ui_cfg
    session: UISession = st.session_state.session

    engine_cfg = _load_engine_config()
    profiles = engine_cfg.get("profiles", {}) or {}
    engines_cfg = engine_cfg.get("engines", {}) or {}
    if not profiles:
        st.error("No profiles found in llm_engines.yaml. Check your installation.")
        return

    # ------------------------------------------------------------
    # Sidebar: configuration
    # ------------------------------------------------------------
    with st.sidebar:
        st.header("Current Session")

        profile_names = sorted(profiles.keys())
        thinking_hint = _profile_looks_like_thinking(
            ui_cfg.profile,
            profiles.get(ui_cfg.profile, {}) if ui_cfg.profile in profiles else profiles[sorted(profiles.keys())[0]],
            engines_cfg,
        )

        ui_cfg.profile = st.selectbox(
            "Engine profile",
            profile_names,
            index=profile_names.index(ui_cfg.profile) if ui_cfg.profile in profile_names else 0,
            help=(
                "Select an engine profile from llm_engines.yaml. Profiles define primary/fallback/cloud engines "
                "and failover behavior.\n\n"
                "Note: profiles that use reasoning/thinking models can take longer to respond due to extra internal "
                "deliberation. If a thinking model feels slow, reduce output tokens, reduce prompt size, or switch "
                "to a fast profile."
            ),
        )

        current_order = _resolve_current_profile_order(ui_cfg, engine_cfg)
        current_engine_name = current_order[0] if current_order else "(none)"
        current_engine = _engine_summary_row(current_engine_name, engines_cfg)
        st.caption(f"Profile: **{ui_cfg.profile}**")
        st.caption(f"Engine: **{current_engine['name']}**")
        st.caption(f"Backend: **{current_engine['backend']}**")
        st.caption(f"Endpoint: `{current_engine['endpoint'] or 'n/a'}`")

        st.divider()
        st.subheader("Environment Health")
        for label, value in _sidebar_health_snapshot(engine_cfg, ui_cfg):
            st.caption(f"**{label}:** {value}")

        st.divider()
        st.subheader("Session Mode")
        ui_cfg.endpoint_only_mode = st.toggle(
            "Endpoint-only mode",
            value=bool(getattr(ui_cfg, 'endpoint_only_mode', False)),
            help="Prefer already-running Ollama/vLLM/OpenAI-compatible endpoints and avoid loading an extra local GPU-backed model in the same session.",
        )
        if ui_cfg.endpoint_only_mode:
            st.info("Endpoint-only mode is on. Configure your profile to point at running inference servers on a single-GPU system.")

        st.divider()
        st.subheader("Configuration")

        # After selection, show a targeted hint if the chosen profile likely uses a thinking model.
        prof_cfg = profiles.get(ui_cfg.profile, {})
        if _profile_looks_like_thinking(ui_cfg.profile, prof_cfg, engines_cfg):
            with st.expander("Thinking model guidance", expanded=False):
                st.markdown(
                    "**This profile appears to use a reasoning/thinking model.** These models may spend extra time "
                    "deliberating, even on simple prompts. If it feels like the chat is stuck, try one of the "
                    "mitigations below."
                )
                st.markdown(
                    "**Mitigations:**\n"
                    "- Reduce *Reserve output tokens* (e.g., 128–256).\n"
                    "- Reduce *Total prompt token cap* (smaller context = faster).\n"
                    "- Ask for brevity: *'Answer directly in 3 sentences; no internal reasoning.'*\n"
                    "- Switch to a fast profile for casual chat; use thinking only for hard problems."
                )

                preset_a, preset_b, preset_c = st.columns(3)
                with preset_a:
                    if st.button("Preset: Fast", help="Lower caps for quick interactive replies."):
                        ui_cfg.total_prompt_tokens = min(int(ui_cfg.total_prompt_tokens), 4096)
                        ui_cfg.reserve_output_tokens = min(int(ui_cfg.reserve_output_tokens), 256)
                        st.info("Applied Fast preset (prompt<=4096, output<=256).")
                with preset_b:
                    if st.button("Preset: Normal", help="Balanced caps for typical chat."):
                        ui_cfg.total_prompt_tokens = min(int(ui_cfg.total_prompt_tokens), 6144)
                        ui_cfg.reserve_output_tokens = min(int(ui_cfg.reserve_output_tokens), 512)
                        st.info("Applied Normal preset (prompt<=6144, output<=512).")
                with preset_c:
                    if st.button("Preset: Deep", help="Higher caps for longer reasoning (slower)."):
                        ui_cfg.total_prompt_tokens = max(int(ui_cfg.total_prompt_tokens), 8192)
                        ui_cfg.reserve_output_tokens = max(int(ui_cfg.reserve_output_tokens), 768)
                        st.info("Applied Deep preset (prompt>=8192, output>=768).")

        # Cloud failover toggles (only affects routing; it doesn't force cloud usage).
        ui_cfg.allow_cloud_failover = st.toggle(
            "Allow cloud failover",
            value=ui_cfg.allow_cloud_failover,
            help=(
                "If enabled, the router may fall back to a cloud LLM if local engines fail. "
                "Requires an API key in the environment. See README for cloud_policy."
            ),
        )
        ui_cfg.cloud_policy = st.selectbox(
            "Cloud policy",
            ["query_only", "query_plus_summary", "full_context"],
            index=["query_only", "query_plus_summary", "full_context"].index(ui_cfg.cloud_policy)
            if ui_cfg.cloud_policy in ["query_only", "query_plus_summary", "full_context"] else 1,
            disabled=not ui_cfg.allow_cloud_failover,
            help=(
                "Controls what data is sent to the cloud on failover. "
                "query_only sends just the current user message; "
                "query_plus_summary sends a compact summary of retrieved memory; "
                "full_context sends the full prompt (least private)."
            ),
        )
        ui_cfg.engine_order = st.text_input(
            "Runtime engine order override",
            value=ui_cfg.engine_order or "",
            help=(
                "Optional comma-separated engine names. This only affects the current chat session unless you save UI settings. "
                "Example: qwen3_8b,qwen3_32b,openai_fallback"
            ),
        ) or None

        with st.expander("Discovered models (local)", expanded=False):
            """Show models already available on the primary engine.

            This is a best-effort helper intended to reduce "guesswork".
            - Ollama: lists local models via /api/tags.
            - vLLM/OpenAI-compatible: lists models via /v1/models.

            Failures are non-fatal; we show a warning and continue.
            """
            cfg_profiles = engine_cfg.get("profiles") or {}
            cfg_engines = engine_cfg.get("engines") or {}
            prof = cfg_profiles.get(ui_cfg.profile) or {}

            order = [s.strip() for s in (ui_cfg.engine_order or "").split(",") if s.strip()]
            if not order:
                # If the user has not overridden the order, use the profile's declared order.
                order = list(prof.get("engines") or [])

            primary = order[0] if order else None
            if not primary:
                st.info("No primary engine selected.")
            else:
                e = cfg_engines.get(primary) or {}
                etype = (e.get("type") or "").lower()
                base_url = str(e.get("base_url") or "").strip()

                try:
                    if etype == "ollama" and base_url:
                        # Ollama config may include /v1; discovery uses the Ollama root.
                        root = base_url.replace("/v1", "").rstrip("/")
                        models = list_ollama_models(root)
                        st.write(f"Primary engine: **{primary}** (Ollama)")
                        st.write("Local models:")
                        st.code("\n".join(m.id for m in models) or "(none)")

                    elif etype == "vllm" and base_url:
                        models = list_vllm_models(base_url)
                        st.write(f"Primary engine: **{primary}** (vLLM/OpenAI API)")
                        st.write("Served model IDs:")
                        st.code("\n".join(m.id for m in models) or "(none)")

                    elif etype in {"llama_cpp", "llama-cpp", "llamacpp"} and base_url:
                        # llama-server doesn't expose rich model metadata; probe reachability
                        try:
                            models = list_vllm_models(base_url)  # uses OpenAI /v1/models
                            n_gpu = e.get("n_gpu_layers", 0)
                            gguf = e.get("gguf_path", "(not set)")
                            gpu_note = (
                                "CPU-only" if int(n_gpu) == 0
                                else f"split GPU+CPU ({n_gpu} layers on GPU)" if int(n_gpu) < 999
                                else "full GPU"
                            )
                            st.write(f"Primary engine: **{primary}** (llama.cpp — {gpu_note})")
                            st.caption(f"GGUF: `{gguf}`")
                            st.write("Served model IDs:")
                            st.code("\n".join(m.id for m in models) or "(none)")
                        except Exception:
                            st.write(f"Primary engine: **{primary}** (llama.cpp)")
                            st.caption("llama-server not reachable — check that it is running.")
                            cmd = build_llama_cpp_launch_command(e)
                            if cmd:
                                st.code(cmd, language="bash")

                    else:
                        st.info(f"Discovery not supported for engine type: {etype!r}.")

                except Exception as ex:
                    st.warning(f"Model discovery failed: {ex}")



        with st.expander("Managed runtimes", expanded=False):
            st.caption("For Ollama, status reflects service/model availability. For vLLM/llama.cpp/OpenAI-compatible endpoints, status reflects endpoint reachability. Start/stop only applies to engines marked `managed: true`. For llama.cpp engines, the sandbox auto-generates the `llama-server` launch command from `gguf_path` and `n_gpu_layers` — no `start_command` entry needed.")
            runtime_rows = describe_profile_runtime(ui_cfg.profile)
            if not runtime_rows:
                st.info("No engines found for this profile.")

            for row in runtime_rows:
                row_engine_cfg = engines_cfg.get(row.engine_name, {})
                etype = (row_engine_cfg.get("type") or "").lower().strip()

                cols = st.columns([2,2,2,1,1])
                cols[0].write(f"**{row.engine_name}**")

                if etype == "ollama":
                    status_label = "available" if row.running else "unavailable"
                elif etype in {"vllm", "openai", "openai-compatible", "openai_compatible"}:
                    status_label = "reachable" if row.running else "unreachable"
                elif etype in {"llama_cpp", "llama-cpp", "llamacpp"}:
                    status_label = "reachable" if row.running else "unreachable"
                else:
                    status_label = "running" if row.running else "stopped"

                cols[1].write(status_label)
                cols[2].write(row.ownership)

                if row.managed:
                    if cols[3].button("Start", key=f"rt_start_{row.engine_name}", disabled=row.running):
                        try:
                            start_managed_engine(row.engine_name)
                            st.rerun()
                        except Exception as ex:
                            st.warning(f"Failed to start {row.engine_name}: {ex}")
                    if cols[4].button("Stop", key=f"rt_stop_{row.engine_name}", disabled=not row.running or row.ownership != 'sandbox-managed'):
                        try:
                            stop_managed_engine(row.engine_name)
                            st.rerun()
                        except Exception as ex:
                            st.warning(f"Failed to stop {row.engine_name}: {ex}")
                else:
                    cols[3].write("—")
                    cols[4].write("—")
                if row.detail:
                    st.caption(f"{row.engine_name}: {row.detail}")

        st.divider()
        st.subheader("Prompt & Retrieval")

        ui_cfg.total_prompt_tokens = st.slider(
            "Total prompt token cap",
            min_value=1024,
            max_value=32000,
            value=int(ui_cfg.total_prompt_tokens),
            step=256,
            help="Hard cap for prompt tokens. Engram will compress or truncate to stay within this limit.",
        )
        ui_cfg.reserve_output_tokens = st.slider(
            "Reserve output tokens",
            min_value=128,
            max_value=4096,
            value=int(ui_cfg.reserve_output_tokens),
            step=128,
            help="Tokens reserved for the model's response. Higher values reduce available prompt space.",
        )

        ui_cfg.include_cold_fallback = st.toggle(
            "Enable cold-storage fallback",
            value=ui_cfg.include_cold_fallback,
            help=(
                "If episodic/semantic recall is weak, Engram can fallback to cold storage retrieval "
                "and include those excerpts in the prompt."
            ),
        )
        ui_cfg.store_overflow_summary = st.toggle(
            "Store overflow summary",
            value=ui_cfg.store_overflow_summary,
            help=(
                "If prompt compression triggers (context overflow), store the compressed memory blob back "
                "into episodic memory as a low-importance summary. This can improve future recall."
            ),
        )

        st.subheader("Advanced budgets (optional)")
        ui_cfg.working_tokens = st.number_input(
            "Working memory tokens",
            min_value=0,
            max_value=20000,
            value=int(ui_cfg.working_tokens) if ui_cfg.working_tokens is not None else 0,
            step=100,
            help="Token budget for working memory. Set to 0 to use Engram defaults.",
        )
        ui_cfg.episodic_tokens = st.number_input(
            "Episodic tokens",
            min_value=0,
            max_value=20000,
            value=int(ui_cfg.episodic_tokens) if ui_cfg.episodic_tokens is not None else 0,
            step=100,
            help="Token budget for episodic retrieval. Set to 0 to use Engram defaults.",
        )
        ui_cfg.semantic_tokens = st.number_input(
            "Semantic tokens",
            min_value=0,
            max_value=20000,
            value=int(ui_cfg.semantic_tokens) if ui_cfg.semantic_tokens is not None else 0,
            step=100,
            help="Token budget for semantic retrieval. Set to 0 to use Engram defaults.",
        )
        ui_cfg.cold_tokens = st.number_input(
            "Cold tokens",
            min_value=0,
            max_value=20000,
            value=int(ui_cfg.cold_tokens) if ui_cfg.cold_tokens is not None else 0,
            step=100,
            help="Token budget for cold-storage fallback excerpts. Set to 0 to use Engram defaults.",
        )

        ui_cfg.show_debug_panels = st.toggle(
            "Show debug panels",
            value=ui_cfg.show_debug_panels,
            help="Show context/routing metadata panels after each response.",
        )

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Save UI settings", help="Persist current settings to your user config directory."):
                path = save_ui_config(ui_cfg)
                st.success(f"Saved to {path}")
        with col_b:
            if st.button("New chat session", help="Start a fresh working-memory session (does not delete storage)."):
                session.session_id = session.session_id = session.session_id = session.session_id = session.session_id  # no-op for mypy
                session.messages.clear()
                # Recreate a session ID; the ProjectMemory will get a new working session on next init.
                from apps.sandbox.session_state import new_session_id
                session.session_id = new_session_id()
                st.session_state.pm = None
                st.info(f"New session: {session.session_id}")

        st.divider()
        st.subheader("Diagnostics")
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            if st.button("Run doctor", help="Check engine reachability, models present, and hardware hints."):
                report = doctor_call(profile=ui_cfg.profile, pull_missing=False)
                st.session_state.last_doctor = report
        with diag_col2:
            if st.button("Recommend", help="Suggest a profile/model tier based on detected hardware."):
                rec = recommend_call(profile=ui_cfg.profile)
                st.session_state.last_recommend = rec


    # ------------------------------------------------------------
    # Main area: tabs for Chat and Model Management
    # ------------------------------------------------------------
    chat_tab, models_tab, memory_tab, diagnostics_tab = st.tabs(
        ["Chat", "Models", "Memory", "Diagnostics"]
    )

    # ============================================================
    # TAB: Chat
    # ============================================================
    with chat_tab:
        # Create ProjectMemory lazily (after profile selection)
        if st.session_state.pm is None:
            overrides = RuntimeOverrides(
                total_prompt_tokens=ui_cfg.total_prompt_tokens,
                reserve_output_tokens=ui_cfg.reserve_output_tokens,
                include_cold_fallback=ui_cfg.include_cold_fallback,
                store_overflow_summary=ui_cfg.store_overflow_summary,
                working_tokens=ui_cfg.working_tokens or None,
                episodic_tokens=ui_cfg.episodic_tokens or None,
                semantic_tokens=ui_cfg.semantic_tokens or None,
                cold_tokens=ui_cfg.cold_tokens or None,
                engine_order=[s.strip() for s in (ui_cfg.engine_order or "").split(",") if s.strip()] or None,
                allow_cloud_failover=ui_cfg.allow_cloud_failover,
                cloud_policy=ui_cfg.cloud_policy,
            )
            project_dir = _default_project_dir()
            project_dir.mkdir(parents=True, exist_ok=True)

            try:
                pm = create_project_memory(profile=ui_cfg.profile, project_dir=project_dir, overrides=overrides)
                pm.new_session(session.session_id)
                st.session_state.pm = pm
            except Exception as e:
                if _is_cuda_runtime_init_error(e):
                    _render_cuda_runtime_help(e)
                else:
                    st.error(f"Failed to create engine: {e}\n\nGo to the **Models** tab to configure engines and profiles.")
                st.stop()

        pm = st.session_state.pm

        # Chat UI
        st.subheader("Chat")

        for msg in session.messages:
            with st.chat_message(msg.role):
                st.markdown(msg.content)

        user_text = st.chat_input("Type a message to test Engram…")

        if user_text:
            # Store to UI transcript
            session.messages.append(ChatMessage(role="user", content=user_text))
            with st.chat_message("user"):
                st.markdown(user_text)

            # Add to working memory
            pm.add_turn("user", user_text)

            # Build prompt (Engram handles retrieval + pressure valve)
            result = pm.build_prompt(
                user_message=user_text,
                query=user_text,
                max_prompt_tokens=ui_cfg.total_prompt_tokens,
                reserve_output_tokens=ui_cfg.reserve_output_tokens,
                include_cold_fallback=ui_cfg.include_cold_fallback,
                store_overflow_summary=ui_cfg.store_overflow_summary,
            )

            prompt = result["prompt"]

            # Memory / retrieval visibility
            context = result.get("context")
            context_dict = context.to_dict() if hasattr(context, "to_dict") else None

            st.session_state["last_user_query"] = user_text
            st.session_state["last_prompt_preview"] = str(prompt)[:4000]
            st.session_state["last_context_dict"] = context_dict

            trace = {
                "user_text": user_text,
                "prompt_preview": str(prompt)[:500],
                "result_keys": list(result.keys()),
                "compressed": result.get("compressed"),
                "prompt_tokens": result.get("prompt_tokens"),
                "memory_tokens": result.get("memory_tokens"),
            }

            if context_dict:
                trace["context"] = context_dict
                trace["token_counts"] = context_dict.get("token_counts", {})
                trace["prompt_sections"] = (
                    context.to_prompt_sections()
                    if hasattr(context, "to_prompt_sections")
                    else {}
                )
                trace["selected_counts"] = {
                    "working": len(context_dict.get("working", [])),
                    "episodic": len(context_dict.get("episodic", [])),
                    "semantic": len(context_dict.get("semantic", [])),
                    "cold": len(context_dict.get("cold", [])),
                }

            for key in (
                "episodic_count",
                "semantic_count",
                "graph_count",
                "neural_count",
                "retrieved_items",
                "context_items",
                "memory_hits",
                "trace",
                "retrieval_trace",
            ):
                if key in result:
                    trace[key] = result[key]

            st.session_state["last_retrieval_trace"] = trace

            try:
                response = pm.llm_engine.generate(prompt)
            except Exception as ex:
                total_vram_gb = None
                try:
                    gpu_summary = _gpu_runtime_summary()
                    total_mb = gpu_summary.get("total_mb")
                    if total_mb:
                        total_vram_gb = float(total_mb) / 1024.0
                except Exception:
                    pass

                vllm_failure = _find_relevant_vllm_failure(engine_cfg, ui_cfg, ex)
                if vllm_failure is not None:
                    engine_name, engine, failure = vllm_failure
                    st.session_state["last_engine_failure"] = {
                        **failure,
                        "technical_error": str(ex),
                    }

                    backend = str(failure.get("backend") or engine.get("type") or "").lower()
                    if backend in {"llama_cpp", "llama-cpp", "llamacpp"}:
                        _render_llama_cpp_failure_message(
                            engine_name=engine_name,
                            engine=engine,
                            failure=failure,
                            technical_error=str(ex),
                        )
                    else:
                        _render_vllm_failure_message(
                            engine_name=engine_name,
                            engine=engine,
                            failure=failure,
                            total_vram_gb=total_vram_gb,
                            technical_error=str(ex),
                        )
                    st.stop()

                raise
                
            if hasattr(response, "text") and response.text is not None:
                assistant_text = response.text
            else:
                assistant_text = str(response)

            st.session_state["last_engine_failure"] = None

            # Capture reasoning visibility metadata for Diagnostics tab
            engine = getattr(pm, "llm_engine", None)
            # Walk into FailoverEngine to find the underlying VLLMEngine
            active_engine = engine
            if hasattr(engine, "engines"):
                for e in getattr(engine, "engines", []):
                    if hasattr(e, "_last_reasoning_detected"):
                        active_engine = e
                        break
            st.session_state["last_reasoning_meta"] = {
                "reasoning_detected": getattr(active_engine, "_last_reasoning_detected", False),
                "reasoning_visibility": getattr(active_engine, "reasoning_visibility", "n/a"),
                "raw_length": len(getattr(active_engine, "_last_raw_response", None) or ""),
                "cleaned_length": len(getattr(active_engine, "_last_cleaned_response", None) or ""),
            }

            pm.add_turn("assistant", assistant_text)
            session.messages.append(ChatMessage(role="assistant", content=assistant_text))

            with st.chat_message("assistant"):
                st.markdown(assistant_text)

            # Debug panels
            if ui_cfg.show_debug_panels:
                with st.expander("Run details", expanded=False):
                    debug_info = {
                        "compressed": result.get("compressed"),
                        "prompt_tokens": result.get("prompt_tokens"),
                        "memory_tokens": result.get("memory_tokens"),
                        "engine": getattr(pm.llm_engine, "name", pm.llm_engine.__class__.__name__),
                    }
                    # Neural memory info
                    ctx = result.get("context")
                    if ctx and getattr(ctx, "neural_meta", None):
                        debug_info["neural"] = ctx.neural_meta
                    st.write(debug_info)
                with st.expander("Context sections", expanded=False):
                    ctx = result["context"]
                    st.code(json.dumps(ctx.to_dict(), indent=2), language="json")

        st.caption("Tip: Enable ENGRAM_TELEMETRY=1 for structured routing events.")

    with memory_tab:
        render_memory_tab(st.session_state.get("pm"), st.session_state)
        
    with diagnostics_tab:
        st.subheader("Diagnostics")

        failure = st.session_state.get("last_engine_failure")
        if failure:
            st.markdown("**Last engine failure**")

            failure_kind = str(failure.get("failure_kind") or "unknown")
            if failure_kind == "not_running":
                st.error("The vLLM server was not reachable during the last generation attempt.")
            elif failure_kind == "wrong_model":
                st.error("The vLLM server was reachable, but the expected model was not loaded.")
            elif failure_kind == "engine_crashed":
                st.error("The vLLM server appears to have failed during generation.")
            elif failure_kind == "token_budget_exceeded":
                st.error(
                    "The generation request was rejected: output tokens exceed the server's "
                    "context limit. Engram auto-retried with a clamped value — if this persists, "
                    "reduce `max_tokens` in `llm_engines.yaml` or restart vLLM with a larger "
                    "`--max-model-len`."
                )
            else:
                st.error("The last generation attempt failed.")

            info_col1, info_col2 = st.columns(2)
            with info_col1:
                if failure.get("engine_name"):
                    st.caption(f"Engine: `{failure['engine_name']}`")
                if failure.get("endpoint"):
                    st.caption(f"Endpoint: `{failure['endpoint']}`")
                if failure.get("expected_model"):
                    st.caption(f"Expected model: `{failure['expected_model']}`")

            with info_col2:
                if failure.get("message"):
                    st.caption(failure["message"])
                if failure.get("failure_message"):
                    st.caption(failure["failure_message"])

            served_models = failure.get("served_models") or []
            if served_models:
                with st.expander("Discovered served models", expanded=False):
                    for model_id in served_models:
                        st.write(f"`{model_id}`")

            with st.expander("Technical details", expanded=False):
                st.code(
                    str(failure.get("technical_error") or "No technical details available."),
                    language="text",
                )

            with st.expander("Raw engine failure record", expanded=False):
                st.code(json.dumps(failure, indent=2), language="json")
        else:
            st.caption("No engine failures recorded in this session.")

        # Reasoning visibility panel
        reasoning_meta = st.session_state.get("last_reasoning_meta")
        if reasoning_meta:
            st.divider()
            st.markdown("**Last response — reasoning filter**")
            col1, col2, col3 = st.columns(3)
            detected = reasoning_meta.get("reasoning_detected", False)
            col1.metric("Reasoning detected", "yes" if detected else "no")
            col2.metric("Mode", reasoning_meta.get("reasoning_visibility", "n/a"))
            raw_len = reasoning_meta.get("raw_length", 0)
            clean_len = reasoning_meta.get("cleaned_length", 0)
            stripped = raw_len - clean_len
            col3.metric("Chars stripped", stripped if stripped > 0 else 0)
            if detected:
                st.caption(
                    f"Raw response was {raw_len} chars; cleaned response is {clean_len} chars. "
                    "Internal reasoning was filtered before display."
                )
    
    # ============================================================
    # TAB: Model Management
    # ============================================================
    with models_tab:
        render_model_management()


if __name__ == "__main__":
    main()
