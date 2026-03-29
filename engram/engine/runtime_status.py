from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import shlex


def parse_host_port_from_base_url(base_url: str) -> Tuple[str, int]:
    url = (base_url or "").strip()
    if not url:
        return "127.0.0.1", 8000

    try:
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8000
        return host, port
    except Exception:
        return "127.0.0.1", 8000


def build_vllm_launch_command(engine_cfg: Dict[str, Any]) -> Optional[str]:
    etype = str(engine_cfg.get("type") or "").lower().strip()
    if etype not in {"vllm", "openai-compatible", "openai_compatible"}:
        return None

    launch = engine_cfg.get("launch") or {}
    source = (
        launch.get("source")
        or engine_cfg.get("hf_repo_id")
        or engine_cfg.get("local_model_dir")
    )
    if not source:
        return None

    served_model_name = str(engine_cfg.get("model") or "").strip()
    if not served_model_name:
        return None

    default_host, default_port = parse_host_port_from_base_url(str(engine_cfg.get("base_url") or ""))
    host = str(launch.get("host") or default_host)
    port = int(launch.get("port") or default_port)

    parts = [
        "vllm",
        "serve",
        str(source),
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
    ]

    for arg in launch.get("extra_args", []) or []:
        parts.append(str(arg))

    return shlex.join(parts)


def build_llama_cpp_launch_command(engine_cfg: Dict[str, Any]) -> Optional[str]:
    """Generate a ``llama-server`` launch command from engine YAML config.

    Example output::

        llama-server -m /models/qwen2.5-32b-q4_k_m.gguf \\
            --host 127.0.0.1 --port 8080 \\
            -c 32768 --n-gpu-layers 40

    The ``gguf_path`` field must be set in the engine config for a launch
    command to be generated.  ``n_gpu_layers`` controls GPU offload:

      ``n_gpu_layers: 0``    → full CPU execution
      ``n_gpu_layers: 999``  → all layers on GPU (use model's actual layer count)
      ``n_gpu_layers: N``    → split: first N layers on GPU, rest on CPU

    Args:
        engine_cfg: Engine configuration dict (as loaded from YAML).

    Returns:
        Shell-quoted launch command string, or None if gguf_path is not set.
    """
    etype = str(engine_cfg.get("type") or "").lower().strip()
    if etype not in {"llama_cpp", "llama-cpp", "llamacpp"}:
        return None

    gguf_path = engine_cfg.get("gguf_path")
    if not gguf_path:
        return None

    default_host, default_port = parse_host_port_from_base_url(
        str(engine_cfg.get("base_url") or "http://127.0.0.1:8080/v1")
    )
    launch = engine_cfg.get("launch") or {}
    host = str(launch.get("host") or default_host)
    port = int(launch.get("port") or default_port)
    max_context = int(engine_cfg.get("max_context") or 4096)
    n_gpu_layers = int(engine_cfg.get("n_gpu_layers") or 0)

    parts = [
        "llama-server",
        "-m", str(gguf_path),
        "--host", host,
        "--port", str(port),
        "-c", str(max_context),
        "--n-gpu-layers", str(n_gpu_layers),
    ]

    for arg in launch.get("extra_args", []) or []:
        parts.append(str(arg))

    return shlex.join(parts)


def classify_llama_cpp_generation_failure(error_text: str) -> str:
    """Classify a llama-server generation failure into a recoverable kind.

    Returns one of:
      ``not_running``      — server not reachable
      ``context_overflow`` — prompt exceeds configured context window
      ``engine_crashed``   — server returned 500 or internal error
      ``generation_failed`` — other / unknown

    Note: llama-server does not have a "wrong_model" failure mode because
    it always serves the single GGUF file it was launched with.
    """
    low = (error_text or "").lower()

    if any(k in low for k in ("connection refused", "cannot connect", "unreachable",
                               "failed to connect", "connection error")):
        return "not_running"

    if any(k in low for k in ("context", "prompt is too long", "exceed",
                               "tokens exceed", "kv cache")):
        return "context_overflow"

    if any(k in low for k in ("500", "internal server error", "server error",
                               "llama_decode", "slot error")):
        return "engine_crashed"

    return "generation_failed"


def build_llama_cpp_recovery_guidance(failure_kind: str, engine_cfg: Dict[str, Any]) -> str:
    """Human-readable recovery guidance for a llama-server failure."""
    gguf = engine_cfg.get("gguf_path", "(unknown path)")
    base_url = engine_cfg.get("base_url", "http://127.0.0.1:8080/v1")
    launch_cmd = build_llama_cpp_launch_command(engine_cfg) or "(set gguf_path to generate)"

    if failure_kind == "not_running":
        return (
            f"llama-server is not running at {base_url}.\n"
            f"Start it with:\n  {launch_cmd}"
        )
    if failure_kind == "context_overflow":
        ctx = engine_cfg.get("max_context", 4096)
        return (
            f"Prompt exceeded llama-server context window ({ctx} tokens).\n"
            f"Options:\n"
            f"  1. Increase -c in the launch command and max_context in YAML.\n"
            f"  2. Load the GGUF with a larger context if the model supports it.\n"
            f"  3. Reduce prompt size via compression_strategy: truncate_end."
        )
    if failure_kind == "engine_crashed":
        return (
            f"llama-server crashed or returned a server error.\n"
            f"Check the server log and restart:\n  {launch_cmd}\n"
            f"If using GPU offload (n_gpu_layers > 0), try reducing it to avoid OOM."
        )
    return f"llama-server generation failed. Check server log at {base_url}."


def classify_runtime_state(
    backend: str,
    reachable: Optional[bool],
    served: Optional[bool],
    endpoint: str,
    model: str,
) -> Tuple[str, str]:
    backend = (backend or "").lower().strip()

    if backend in {"anthropic", "claude", "openai"}:
        return "remote", "Remote/cloud runtime."

    if backend not in {"vllm", "openai-compatible", "openai_compatible", "ollama"}:
        return "unknown", "Runtime state unavailable."

    if reachable is False:
        if backend == "ollama":
            return "not_running", "Ollama endpoint is not reachable."
        return "not_running", "vLLM endpoint is not reachable."

    if reachable and model and served:
        return "running", "Expected model is currently being served."

    if reachable and model and served is False:
        return "wrong_model", "Endpoint is reachable, but the expected model is not being served."

    if reachable:
        return "reachable", "Endpoint is reachable."

    return "unknown", "Runtime state unavailable."


def probe_vllm_runtime_state(
    engine_name: str,
    engine_cfg: Dict[str, Any],
    list_models_fn,
) -> Dict[str, Any]:
    endpoint = str(engine_cfg.get("base_url") or "http://localhost:8000/v1").strip()
    expected_model = str(engine_cfg.get("model") or "").strip()

    try:
        served_models = list_models_fn(endpoint)
        served_ids: List[str] = []
        for item in served_models or []:
            if isinstance(item, dict):
                model_id = item.get("id")
            else:
                model_id = getattr(item, "id", None)
                if model_id is None:
                    model_id = str(item)

            if model_id:
                served_ids.append(str(model_id))

        served_ids = [m for m in served_ids if m]

        if expected_model and expected_model not in served_ids:
            return {
                "state": "wrong_model",
                "engine_name": engine_name,
                "endpoint": endpoint,
                "expected_model": expected_model,
                "served_models": served_ids,
                "message": "The vLLM endpoint is reachable, but it is not serving the model Engram expects.",
            }

        return {
            "state": "running",
            "engine_name": engine_name,
            "endpoint": endpoint,
            "expected_model": expected_model,
            "served_models": served_ids,
            "message": "The vLLM endpoint is reachable and serving the expected model.",
        }

    except Exception as ex:
        return {
            "state": "not_running",
            "engine_name": engine_name,
            "endpoint": endpoint,
            "expected_model": expected_model,
            "served_models": [],
            "message": "The vLLM endpoint is not reachable.",
            "error": str(ex),
        }


def classify_vllm_generation_failure(
    engine_name: str,
    engine_cfg: Dict[str, Any],
    exc: Exception,
    list_models_fn,
) -> Dict[str, Any]:
    runtime = probe_vllm_runtime_state(engine_name, engine_cfg, list_models_fn)

    err_text = " ".join(
        str(x).lower()
        for x in (
            exc,
            getattr(exc, "__cause__", None),
            getattr(exc, "__context__", None),
        )
        if x is not None
    )

    if runtime["state"] == "not_running":
        runtime["failure_kind"] = "not_running"
        runtime["failure_message"] = "The vLLM server is not reachable."
        return runtime

    if runtime["state"] == "wrong_model":
        runtime["failure_kind"] = "wrong_model"
        runtime["failure_message"] = "The vLLM server is up, but the expected model is not loaded."
        return runtime
        
    if any(marker in err_text for marker in ("404", "notfounderror", "does not exist", "model `")):
        runtime["failure_kind"] = "wrong_model"
        runtime["failure_message"] = "The vLLM server is reachable, but the expected model is not loaded."
        return runtime

    if any(marker in err_text for marker in ("500", "internalservererror", "enginecore", "server error")):
        runtime["failure_kind"] = "engine_crashed"
        runtime["failure_message"] = "The vLLM server appears to have failed during generation."
        return runtime

    if "max_model_len" in err_text and ("max_tokens" in err_text or "400" in err_text):
        import re
        m = re.search(r"max_model_len\s*=\s*(\d+)", err_text)
        limit_note = f" (server limit: {m.group(1)} tokens)" if m else ""
        runtime["failure_kind"] = "token_budget_exceeded"
        runtime["failure_message"] = (
            f"Engram requested more output tokens than the server allows{limit_note}. "
            "Reduce max_tokens in the engine config or restart vLLM with a larger --max-model-len."
        )
        return runtime

    runtime["failure_kind"] = "generation_failed"
    runtime["failure_message"] = "The vLLM server is reachable, but the generation request failed."
    return runtime


def build_recovery_guidance(
    backend: str,
    failure_kind: str,
    launch_command: Optional[str],
) -> Dict[str, Any]:
    backend = (backend or "").lower().strip()
    failure_kind = (failure_kind or "unknown").lower().strip()

    steps: List[str] = []
    advanced: List[str] = []

    if backend in {"vllm", "openai-compatible", "openai_compatible"}:
        if failure_kind == "not_running":
            steps.append("Start the vLLM server with the launch command below, then retry.")
        elif failure_kind == "wrong_model":
            steps.append("Restart vLLM with the expected served model name, or switch Engram to the model that is actually running.")
        elif failure_kind == "engine_crashed":
            steps.append("Restart the vLLM server, then retry the request.")
            advanced.extend([
                "If GPU memory remains occupied after the crash, stop stale GPU users.",
                "If CUDA remains wedged, use your normal nvidia_uvm recovery procedure.",
                "If recovery fails repeatedly, reboot the machine.",
            ])
        elif failure_kind == "token_budget_exceeded":
            steps.append("Reduce max_tokens in the engine config, or restart vLLM with --max-model-len set to a larger value.")
            advanced.append("Add max_context: <N> to the engine entry in llm_engines.yaml to match the server's --max-model-len.")
        else:
            steps.append("Verify the endpoint and served model, then retry.")

    elif backend == "ollama":
        steps.append("Verify the Ollama server is running and the expected model is installed.")

    elif backend in {"llama_cpp", "llama-cpp", "llamacpp"}:
        if failure_kind == "not_running":
            steps.append("Start llama-server with the launch command below, then retry.")
            steps.append("Install llama-cpp-python[server] if llama-server is not yet available: pip install llama-cpp-python[server]")
        elif failure_kind == "context_overflow":
            steps.append("Increase -c in the launch command and max_context in llm_engines.yaml, then restart llama-server.")
            advanced.append("Alternatively reduce prompt length via compression_strategy: truncate_end in the engine config.")
        elif failure_kind == "engine_crashed":
            steps.append("Restart llama-server. If using GPU offload (n_gpu_layers > 0), try reducing n_gpu_layers in llm_engines.yaml.")
            advanced.extend([
                "Check available VRAM: nvidia-smi",
                "Reduce n_gpu_layers by ~5 and retry until stable.",
                "Set n_gpu_layers: 0 for full CPU execution if GPU offload causes repeated crashes.",
            ])
        else:
            steps.append("Check that llama-server is running and the GGUF file path is correct in llm_engines.yaml.")

    return {
        "launch_command": launch_command,
        "steps": steps,
        "advanced_steps": advanced,
    }
