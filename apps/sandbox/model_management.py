"""Model management UI for Engram.

Provides:
  - Hardware detection display
  - HuggingFace model search with format/engine recommendations
  - Local model import (GGUF or safetensors directory)
  - Primary / fallback model selection
  - Profile editing (add/remove/reorder engines)

All heavy logic lives in engram.engine.model_manager; this module
is purely Streamlit layout and state management.

Author: Jeffrey Dean
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import shutil
import subprocess
import streamlit as st
import yaml

from engram.engine.model_manager import (
    GPUInfo,
    SystemInfo,
    HFModelInfo,
    FormatRecommendation,
    LocalModelInfo,
    detect_system,
    search_hf_models,
    recommend_format_for_hf_model,
    classify_hf_model_format,
    scan_local_model,
    download_ollama_model,
    download_hf_model,
    default_local_model_dir,
    default_model_root,
    import_gguf_to_ollama,
    register_model_in_config,
    add_engine_to_profile,
    recommend_and_register,
    # Model fit scoring
    get_gpu_memory_snapshot as _get_gpu_memory_snapshot,
    artifact_family_from_text as _artifact_family_from_text,
    extract_model_size_b as _extract_model_size_b,
    estimate_model_risk as _estimate_model_risk,
    score_model_fit as _score_model_fit,
)
from engram.engine.model_discovery import list_ollama_models, list_vllm_models
from engram.engine.runtime_status import (
    build_vllm_launch_command,
    classify_runtime_state,
)

logger = logging.getLogger(__name__)

# Session state keys (namespaced to avoid collisions with app.py)
_K_SYSTEM = "mm_system_info"
_K_SEARCH = "mm_search_results"
_K_LOCAL = "mm_local_scan_result"


def _config_path() -> Path:
    return Path("~/.engram/llm_engines.yaml").expanduser()


def _load_config() -> Dict[str, Any]:
    p = _config_path()
    if not p.exists():
        # Fall back to bundled
        import engram.engine
        p = Path(engram.engine.__file__).resolve().parent / "llm_engines.yaml"
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _save_config(cfg: Dict[str, Any]):
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ------------------------------------------------------------------
# Hardware panel
# ------------------------------------------------------------------

def _render_hardware():
    """Show detected hardware."""
    st.subheader("System Hardware")

    if st.button("Detect hardware", key="mm_detect_hw"):
        with st.spinner("Detecting..."):
            st.session_state[_K_SYSTEM] = detect_system()

    sys_info: Optional[SystemInfo] = st.session_state.get(_K_SYSTEM)
    if sys_info is None:
        st.info("Click **Detect hardware** to scan your system.")
        return

    if getattr(sys_info, "warning", None):
        st.warning(sys_info.warning)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("System RAM", f"{sys_info.ram_gb or '?'} GB")
    with col2:
        st.metric("Total VRAM", f"{sys_info.total_vram_gb:.1f} GB")
    with col3:
        st.metric("Accelerator", sys_info.accelerator.upper())

    if sys_info.gpus:
        for g in sys_info.gpus:
            st.caption(f"GPU {g.index}: {g.name} — {g.vram_gb} GB")
    else:
        st.caption("No GPU detected. Models will run on CPU via Ollama.")


# ------------------------------------------------------------------
# HuggingFace search panel
# ------------------------------------------------------------------

def _render_hf_search():
    """Search HuggingFace and show recommendations."""
    st.subheader("Search HuggingFace Models")

    col_q, col_n = st.columns([3, 1])
    include_quantized = st.checkbox(
        "Include quantized repos (AWQ/GPTQ/GGUF)",
        value=True,
        key="mm_hf_include_quantized",
        help="Search base repos and common quantized variants instead of filtering to base models only.",
    )
    with col_q:
        query = st.text_input(
            "Search query",
            placeholder="e.g. Qwen2.5 7B, Llama 3, Mistral",
            key="mm_hf_query",
        )
    with col_n:
        limit = st.number_input("Max results", 5, 50, 15, key="mm_hf_limit")

    if st.button("Search", key="mm_hf_search", disabled=not query):
        with st.spinner(f"Searching HuggingFace for '{query}'..."):
            results = search_hf_models(
                query,
                limit=int(limit),
                include_quantized=include_quantized,
                vram_gb=(st.session_state.get(_K_SYSTEM).primary_vram_gb if st.session_state.get(_K_SYSTEM) else None),
            )
            st.session_state[_K_SEARCH] = results
            if not results:
                st.warning("No models found. Try a different query.")

    results: List[HFModelInfo] = st.session_state.get(_K_SEARCH, [])
    if not results:
        return

    sys_info: Optional[SystemInfo] = st.session_state.get(_K_SYSTEM)
    vram = sys_info.primary_vram_gb if sys_info else 0.0

    st.caption(f"Showing {len(results)} results. "
               f"VRAM budget: {vram:.1f} GB" + (" (detect hardware first for recommendations)" if vram == 0 else ""))

    for i, model in enumerate(results):
        with st.expander(
            f"**{model.repo_id}** — {model.size_str()} · "
            f"{model.downloads:,} downloads · "
            f"{'GGUF ' if model.has_gguf else ''}{'safetensors' if model.has_safetensors else ''}",
            expanded=False,
        ):
            repo_id = model.repo_id
            fit = _score_model_fit(
                model_text=repo_id,
                hf_repo_id=repo_id,
                local_model_dir="",
                backend="vllm",
            )

            if fit["fit"] == "likely_oom":
                st.warning("Likely too large for reliable local use on this system.")
            elif fit["fit"] == "better_ollama":
                st.info("This model is likely a better candidate for Ollama than vLLM on this system.")
            elif fit["fit"] == "blocked_now":
                st.info("This model should fit on this machine, but current GPU memory is already occupied.")
            elif fit["fit"] == "borderline":
                st.info("Borderline fit. Conservative context and memory settings may be required.")

            st.caption(
                f"{_fit_badge_label(fit['fit'])} | "
                f"{_runtime_badge_label(fit['runtime'])} | "
                f"format: {fit['artifact_family']}"
            )
            st.caption(fit["rationale"])

            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Author:** {model.author}")
                st.write(f"**Pipeline:** {model.pipeline_tag}")
                if model.estimated_params_b:
                    st.write(f"**Parameters:** {model.estimated_params_b:.1f}B")
                    st.write(f"**FP16 size:** ~{model.estimated_size_gb_fp16:.1f} GB")
            with col_b:
                st.write(f"**Downloads:** {model.downloads:,}")
                st.write(f"**Likes:** {model.likes:,}")
                if model.tags:
                    st.write(f"**Tags:** {', '.join(model.tags[:8])}")
                family = classify_hf_model_format(model)
                family_label = {
                    "gguf": "GGUF / Ollama-style",
                    "transformers_quantized": "Transformers quantized (AWQ/GPTQ/EXL2)",
                    "mixed": "Mixed signals (GGUF + Transformers quant)",
                    "base": "Base / generic",
                }.get(family, family)
                st.write(f"**Artifact family:** {family_label}")

            if model.estimated_params_b and vram > 0:
                rec = recommend_format_for_hf_model(model, vram)
                _render_recommendation(rec)
                _render_install_button(model, rec, i)
            elif model.estimated_params_b:
                st.info("Detect hardware to get a format recommendation.")
            else:
                st.warning("Cannot estimate model size from repo name. Check the model card for parameter count.")


def _render_recommendation(rec: FormatRecommendation):
    """Show format recommendation as a colored callout."""
    engine_label = {"vllm": "vLLM", "ollama": "Ollama", "llama_cpp": "llama.cpp"}.get(rec.engine, rec.engine.upper())
    if rec.fits_in_vram:
        st.success(
            f"**Recommended:** {engine_label} with {rec.quantization} "
            f"({rec.estimated_vram_gb:.1f} GB estimated)\n\n"
            f"{rec.reason}"
        )
    elif rec.engine == "llama_cpp":
        st.warning(
            f"**Recommended:** {engine_label} split offload — {rec.quantization} "
            f"({rec.num_gpu_layers} GPU layers, rest on CPU)\n\n"
            f"{rec.reason}\n\n"
            f"Download the GGUF file, then use the **llama.cpp** install option below."
        )
    else:
        st.warning(
            f"**Recommended:** {engine_label} with {rec.quantization} "
            f"(partial offload, {rec.num_gpu_layers} GPU layers)\n\n"
            f"{rec.reason}"
        )


def _render_install_button(model: HFModelInfo, rec: FormatRecommendation, idx: int):
    """Download/register button for a search result."""
    col_name, col_btn = st.columns([2, 1])

    # Generate a sensible engine name
    import re
    safe_name = re.sub(r"[^a-z0-9_]", "_", model.name.lower())
    if rec.quantization != "fp16":
        safe_name += f"_{rec.quantization.lower()}"

    key_suffix = re.sub(r"[^a-z0-9_]+", "_", model.repo_id.lower())

    with col_name:
        engine_name = st.text_input(
            "Engine name",
            value=safe_name,
            key=f"mm_eng_name_{key_suffix}",
            help="Name used in llm_engines.yaml and profiles. Bound to the selected Hugging Face result, not cached by row position.",
        )

    with col_btn:
        st.write("")  # Vertical alignment spacer
        if rec.engine == "ollama":
            tag = f"{model.name.lower()}:{rec.quantization.lower()}"
            if st.button(f"Pull via Ollama", key=f"mm_pull_{key_suffix}"):
                _do_ollama_pull(tag, engine_name, rec)
        elif rec.engine == "llama_cpp":
            # llama_cpp: must download the GGUF file, then use llama-server
            default_dir = default_local_model_dir(model.repo_id)
            target_dir = st.text_input(
                "Download directory",
                value=str(default_dir),
                key=f"mm_dl_dir_{key_suffix}",
                help="Where to save the GGUF file. llama-server loads it directly from this path.",
            )
            # Show what the gguf_path will be (best-guess filename)
            gguf_filename = f"{model.name.lower()}-{rec.quantization.lower()}.gguf"
            gguf_path_preview = str(Path(target_dir) / gguf_filename)
            st.caption(f"Expected GGUF path: `{gguf_path_preview}`")

            from engram.engine.runtime_status import build_llama_cpp_launch_command
            launch_cfg = {
                "type": "llama_cpp",
                "gguf_path": gguf_path_preview,
                "base_url": "http://127.0.0.1:8080/v1",
                "max_context": 16384,
                "n_gpu_layers": rec.num_gpu_layers or 0,
            }
            launch_cmd = build_llama_cpp_launch_command(launch_cfg) or ""
            if launch_cmd:
                st.info(f"After download, start with:\n```bash\n{launch_cmd}\n```")

            if st.button("Download GGUF + register", key=f"mm_dl_llama_{key_suffix}"):
                _do_llama_cpp_download_and_register(
                    model, engine_name, rec,
                    Path(target_dir).expanduser(),
                    gguf_filename,
                )
        else:
            default_dir = default_local_model_dir(model.repo_id)
            target_dir = st.text_input(
                "Download directory",
                value=str(default_dir),
                key=f"mm_dl_dir_{key_suffix}",
                help="Engram-managed local path where this Hugging Face repo will be downloaded.",
            )
            st.info(
                f"Recommended flow for vLLM: download the repo into Engram's local model store, "
                f"then start vLLM against that local path.\n\n"
                f"Example:\n```\nvllm serve {target_dir}\n```"
            )
            col_dl, col_reg = st.columns(2)
            with col_dl:
                if st.button("Download + register", key=f"mm_dl_reg_{key_suffix}"):
                    _do_hf_download_and_register(model, engine_name, rec, Path(target_dir).expanduser())
            with col_reg:
                if st.button("Register repo only", key=f"mm_reg_{key_suffix}"):
                    _do_register(engine_name, rec.engine, model.repo_id, rec)


def _do_ollama_pull(tag: str, engine_name: str, rec: FormatRecommendation):
    """Pull model via Ollama and register."""
    with st.spinner(f"Pulling {tag} via Ollama (this may take a while)..."):
        ok = download_ollama_model(tag)
    if ok:
        register_model_in_config(
            engine_name=engine_name,
            engine_type="ollama",
            model_id=tag,
            config_path=_config_path(),
            num_gpu=rec.num_gpu_layers if not rec.fits_in_vram else None,
        )
        st.success(f"Pulled **{tag}** and registered as **{engine_name}**.")
        st.info("Add it to a profile in the **Profile Editor** below.")
    else:
        st.error(f"Failed to pull {tag}. Is Ollama running?")


def _do_hf_download_and_register(model: HFModelInfo, engine_name: str, rec: FormatRecommendation, target_dir: Path):
    """Download a HF model into Engram-managed storage and register the local path."""
    with st.spinner(f"Downloading {model.repo_id} to {target_dir} ..."):
        result = download_hf_model(model.repo_id, local_dir=target_dir)
    if not result.success:
        st.error(f"Download failed: {result.error or 'unknown error'}")
        return
    else:
        fit = _score_model_fit(
            model_text=model.repo_id,
            hf_repo_id=model.repo_id,
            local_model_dir=str(target_dir),
            backend="vllm",
        )

        if fit["fit"] == "likely_oom":
            st.warning(
                "Downloaded successfully. This model is likely to OOM with vLLM on the detected GPU. "
                "Ollama may work better, though speed may be lower."
            )
        elif fit["fit"] == "borderline":
            st.info(
                "Downloaded successfully. This model is a borderline fit for vLLM on the detected GPU."
            )

    register_model_in_config(
        engine_name=engine_name,
        engine_type="vllm",
        model_id=str(result.local_dir),
        config_path=_config_path(),
        num_gpu=rec.num_gpu_layers if not rec.fits_in_vram else None,
        extra={
            "hf_repo_id": model.repo_id,
            "local_model_dir": str(result.local_dir),
            "managed": False,
        },
    )
    st.success(f"Downloaded **{model.repo_id}** to **{result.local_dir}** and registered **{engine_name}**.")
    st.caption(f"Start vLLM with: vllm serve {result.local_dir}")


def _do_llama_cpp_download_and_register(
    model: HFModelInfo,
    engine_name: str,
    rec: FormatRecommendation,
    target_dir: Path,
    gguf_filename: str,
):
    """Download a GGUF file from HF and register as a llama_cpp engine.

    Uses ``allow_patterns`` to fetch only GGUF files matching the
    recommended quantization rather than the full repo snapshot.
    After download, scans for the actual ``.gguf`` file, writes it as
    ``gguf_path`` in the engine config, and shows the ready-to-run
    ``llama-server`` launch command.
    """
    from engram.engine.runtime_status import build_llama_cpp_launch_command

    # Only fetch GGUF files matching the quantization — avoids pulling
    # safetensors shards, tokenizer data, etc. from mixed repos.
    quant = rec.quantization
    allow_patterns = [
        f"*{quant}*.gguf",
        f"*{quant.lower()}*.gguf",
        f"*.gguf",           # broad fallback: some repos use only one GGUF
    ]

    with st.spinner(f"Downloading {model.repo_id} ({quant}) GGUF to {target_dir} …"):
        result = download_hf_model(
            model.repo_id,
            local_dir=target_dir,
            allow_patterns=allow_patterns,
        )

    if not result.success:
        st.error(f"Download failed: {result.error or 'unknown error'}")
        return

    # Discover the downloaded GGUF file
    gguf_files = sorted(target_dir.glob("**/*.gguf"))
    if not gguf_files:
        st.warning(
            "Download completed but no `.gguf` file was found under "
            f"`{target_dir}`. The repo may use an unexpected filename. "
            "Set `gguf_path` manually in `llm_engines.yaml`."
        )
        gguf_path = str(target_dir / gguf_filename)
    else:
        # Prefer a file whose name contains the quantization string
        matched = [f for f in gguf_files if quant.lower() in f.name.lower()]
        chosen = matched[0] if matched else gguf_files[0]
        gguf_path = str(chosen)
        if len(gguf_files) > 1:
            st.caption(
                f"Multiple GGUF files found — using `{chosen.name}`. "
                "Change `gguf_path` in `llm_engines.yaml` to select a different variant."
            )

    n_gpu = rec.num_gpu_layers or 0
    register_model_in_config(
        engine_name=engine_name,
        engine_type="llama_cpp",
        model_id=engine_name,
        config_path=_config_path(),
        num_gpu=n_gpu,
        extra={
            "gguf_path": gguf_path,
            "hf_repo_id": model.repo_id,
        },
    )

    launch_cmd = build_llama_cpp_launch_command({
        "type": "llama_cpp",
        "gguf_path": gguf_path,
        "base_url": "http://127.0.0.1:8080/v1",
        "max_context": 16384,
        "n_gpu_layers": n_gpu,
    })

    st.success(
        f"Downloaded **{model.repo_id}** ({quant}) and registered "
        f"**{engine_name}** as a llama.cpp engine."
    )
    st.caption(f"GGUF: `{gguf_path}`")
    st.caption(
        f"GPU offload: {'CPU-only (n_gpu_layers=0)' if n_gpu == 0 else f'{n_gpu} layers on GPU, rest on CPU'}"
    )
    if launch_cmd:
        st.markdown("**Start llama-server with:**")
        st.code(launch_cmd, language="bash")
    st.info("Add this engine to a failover profile in the **Profile Editor** below.")


def _do_register(engine_name: str, engine_type: str, model_id: str, rec: FormatRecommendation):
    """Register a model without pulling (for vLLM or pre-downloaded)."""
    register_model_in_config(
        engine_name=engine_name,
        engine_type=engine_type,
        model_id=model_id,
        config_path=_config_path(),
        num_gpu=rec.num_gpu_layers if not rec.fits_in_vram else None,
    )
    st.success(f"Registered **{engine_name}** ({engine_type}/{model_id}).")


# ------------------------------------------------------------------
# Inventory / health helpers
# ------------------------------------------------------------------

def _profile_engine_order(cfg: Dict[str, Any], profile_name: str) -> List[str]:
    profiles = cfg.get("profiles", {}) or {}
    prof = profiles.get(profile_name) or {}
    return list(prof.get("engines") or [])


def _safe_exists(path_str: str) -> Optional[bool]:
    if not path_str:
        return None
    try:
        return Path(path_str).expanduser().exists()
    except Exception:
        return None


def _endpoint_status(url: str, kind: str) -> Tuple[bool, List[str], str]:
    try:
        if kind == "ollama":
            models = list_ollama_models(url)
            return True, [m.id for m in models], ""
        models = list_vllm_models(url)
        return True, [m.id for m in models], ""
    except Exception as exc:
        return False, [], str(exc)

def _runtime_state_badge(state: str) -> str:
    s = (state or "").lower().strip()
    if s == "running":
        return "🟢 Running"
    if s in {"reachable", "degraded"}:
        return "🟡 Reachable"
    if s == "wrong_model":
        return "🟠 Wrong model"
    if s in {"not_running", "unreachable"}:
        return "🔴 Not running"
    return "⚪ Unknown"


def _render_vllm_launch_guidance(row: Dict[str, Any]) -> None:
    backend = str(row.get("backend") or "").lower().strip()
    is_vllm = backend in {"vllm", "openai-compatible", "openai_compatible"}
    is_llama_cpp = backend in {"llama_cpp", "llama-cpp", "llamacpp"}

    if not is_vllm and not is_llama_cpp:
        return

    launch_command = row.get("launch_command")
    expected_model = row.get("model") or ""
    endpoint = row.get("endpoint") or ""

    if expected_model:
        st.caption(f"Expected served model: `{expected_model}`")
    if endpoint:
        st.caption(f"Expected endpoint: `{endpoint}`")

    if is_llama_cpp:
        # Show llama-specific metadata
        gguf = row.get("gguf_path") or "(not set)"
        n_gpu = row.get("n_gpu_layers", 0)
        gpu_note = (
            "CPU-only" if int(n_gpu) == 0
            else f"split GPU+CPU ({n_gpu} layers on GPU)" if int(n_gpu) < 999
            else "full GPU"
        )
        st.caption(f"GGUF: `{gguf}`  |  Offload: {gpu_note}")

    if launch_command:
        label = "Launch command" if is_vllm else "llama-server launch command"
        with st.expander(label, expanded=False):
            st.code(launch_command, language="bash")
    elif is_vllm:
        st.caption("No launch metadata available. Add `launch.source` to this engine config to generate a vLLM command.")
    else:
        st.caption("No launch command available. Set `gguf_path` in this engine config.")

def _build_engine_inventory(cfg: Dict[str, Any], selected_profile: str) -> List[Dict[str, Any]]:
    engines_cfg = cfg.get("engines", {}) or {}
    selected_engines = set(_profile_engine_order(cfg, selected_profile))

    endpoint_cache: Dict[Tuple[str, str], Tuple[bool, List[str], str]] = {}
    rows: List[Dict[str, Any]] = []

    for name, ecfg in sorted(engines_cfg.items()):
        etype = str(ecfg.get("type") or "").lower().strip()
        base_url = str(ecfg.get("base_url") or "").strip()
        model = str(ecfg.get("model") or "")
        local_model_dir = str(ecfg.get("local_model_dir") or "")
        hf_repo_id = str(ecfg.get("hf_repo_id") or "")

        if etype == "ollama":
            family = "ollama"
        elif any(k in (model + " " + hf_repo_id + " " + local_model_dir).lower() for k in ("awq", "gptq", "exl2")):
            family = "transformers-quantized"
        elif any(k in (model + " " + hf_repo_id + " " + local_model_dir).lower() for k in ("gguf", "q3_k", "q4_k", "q5_k", "q6_k")):
            family = "gguf"
        elif etype in {"vllm", "openai", "openai-compatible", "openai_compatible"}:
            family = "openai-compatible"
        elif etype in {"llama_cpp", "llama-cpp", "llamacpp"}:
            family = "llama.cpp"
        else:
            family = etype or "unknown"

        downloaded = _safe_exists(local_model_dir) if local_model_dir else None
        reachable = None
        served = None
        endpoint_error = ""
        served_models: List[str] = []

        if base_url and etype in {"ollama", "vllm", "openai", "openai-compatible",
                                    "openai_compatible", "llama_cpp", "llama-cpp", "llamacpp"}:
            kind = "ollama" if etype == "ollama" else "vllm"
            cache_key = (kind, base_url if kind == "vllm" else base_url.replace('/v1', '').rstrip('/'))
            if cache_key not in endpoint_cache:
                endpoint_cache[cache_key] = _endpoint_status(cache_key[1], kind)
            reachable, served_models, endpoint_error = endpoint_cache[cache_key]
            served = bool(model and model in served_models) if reachable else False

        launch_command = build_vllm_launch_command(ecfg)
        if not launch_command and etype in {"llama_cpp", "llama-cpp", "llamacpp"}:
            from engram.engine.runtime_status import build_llama_cpp_launch_command
            launch_command = build_llama_cpp_launch_command(ecfg)
        runtime_state, runtime_message = classify_runtime_state(
            etype,
            reachable,
            served,
            base_url,
            model,
        ) 

        rows.append({
            "engine_name": name,
            "backend": etype or "?",
            "artifact_family": family,
            "model": model,
            "hf_repo_id": hf_repo_id,
            "local_model_dir": local_model_dir,
            "downloaded": downloaded,
            "endpoint": base_url,
            "reachable": reachable,
            "served": served,
            "selected": name in selected_engines,
            "served_models": served_models,
            "endpoint_error": endpoint_error,
            "launch_command": launch_command,
            "runtime_state": runtime_state,
            "runtime_message": runtime_message,
            # llama_cpp-specific metadata
            "gguf_path": str(ecfg.get("gguf_path") or ""),
            "n_gpu_layers": ecfg.get("n_gpu_layers", 0),
        })
    return rows

def _render_status_badge(label: str, ok: Optional[bool]) -> str:
    if ok is None:
        return f"{label}: n/a"
    return f"{label}: {'yes' if ok else 'no'}"




def _fit_badge_label(fit: str) -> str:
    return {
        "good": "Good fit",
        "borderline": "Borderline",
        "blocked_now": "Blocked by current VRAM use",
        "better_ollama": "Better with Ollama",
        "likely_oom": "Likely too large",
        "remote": "Remote runtime",
        "unknown": "Unknown fit",
    }.get(fit, "Unknown fit")


def _runtime_badge_label(runtime: str) -> str:
    return {
        "vllm": "Best with vLLM",
        "ollama": "Better with Ollama",
        "remote": "Remote / cloud",
    }.get(runtime, "Runtime unclear")

    


def _render_local_import():
    """Import a model from local filesystem."""
    st.subheader("Import Local Model")
    st.caption(
        "Point to a GGUF file or a directory containing safetensors. "
        "Useful in air-gapped environments or for testing your own models."
    )

    model_path = st.text_input(
        "Model path",
        placeholder="/path/to/model.gguf or /path/to/model-directory/",
        key="mm_local_path",
    )

    if st.button("Scan", key="mm_local_scan", disabled=not model_path):
        p = Path(model_path).expanduser()
        if not p.exists():
            st.error(f"Path does not exist: {p}")
            st.session_state[_K_LOCAL] = None
            return

        result = scan_local_model(p)
        if result is None:
            st.error("No recognized model files found (expected .gguf or .safetensors).")
            st.session_state[_K_LOCAL] = None
        else:
            st.session_state[_K_LOCAL] = result

    local: Optional[LocalModelInfo] = st.session_state.get(_K_LOCAL)
    if local is None:
        return

    st.write(f"**Name:** {local.name}")
    st.write(f"**Format:** {local.format}")
    st.write(f"**Size:** {local.size_gb:.2f} GB")
    st.write(f"**Files:** {', '.join(local.files[:5])}"
             + (f" (+{len(local.files)-5} more)" if len(local.files) > 5 else ""))

    engine_name = st.text_input(
        "Engine name for this model",
        value=local.name.lower().replace("-", "_").replace(" ", "_"),
        key="mm_local_eng_name",
    )

    if local.format == "gguf":
        gguf_file = local.path if local.path.is_file() else local.path / local.files[0]

        st.markdown("**Option A — Ollama** (simpler, no GPU layer control)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Import to Ollama", key="mm_local_ollama"):
                with st.spinner(f"Importing {gguf_file.name} to Ollama..."):
                    ok = import_gguf_to_ollama(gguf_file, engine_name)
                if ok:
                    register_model_in_config(
                        engine_name=engine_name,
                        engine_type="ollama",
                        model_id=engine_name,
                        config_path=_config_path(),
                    )
                    st.success(f"Imported and registered as **{engine_name}**.")
                else:
                    st.error("Import failed. Check that Ollama is running.")
        with col2:
            if st.button("Register only (no import)", key="mm_local_reg_gguf"):
                register_model_in_config(
                    engine_name=engine_name,
                    engine_type="ollama",
                    model_id=engine_name,
                    config_path=_config_path(),
                )
                st.success(f"Registered **{engine_name}**. You'll need to import it to Ollama manually.")

        st.markdown("**Option B — llama.cpp / llama-server** (split GPU+CPU offload, no import needed)")
        n_gpu = st.number_input(
            "GPU layers (--n-gpu-layers)",
            min_value=0, max_value=999, value=0,
            key="mm_local_llama_gpu_layers",
            help="0 = full CPU. Set to the number of transformer layers to offload to GPU. "
                 "RTX 3090 + Q4_K_M 32B ≈ 40 layers.",
        )
        from engram.engine.runtime_status import build_llama_cpp_launch_command
        preview_cfg = {
            "type": "llama_cpp",
            "gguf_path": str(gguf_file),
            "base_url": "http://127.0.0.1:8080/v1",
            "max_context": 16384,
            "n_gpu_layers": int(n_gpu),
        }
        launch_cmd = build_llama_cpp_launch_command(preview_cfg)
        if launch_cmd:
            st.caption("Launch command preview:")
            st.code(launch_cmd, language="bash")

        if st.button("Register for llama.cpp", key="mm_local_reg_llama"):
            register_model_in_config(
                engine_name=engine_name,
                engine_type="llama_cpp",
                model_id=engine_name,
                config_path=_config_path(),
                num_gpu=int(n_gpu),
                extra={"gguf_path": str(gguf_file)},
            )
            st.success(f"Registered **{engine_name}** as a llama.cpp engine.")
            st.caption(f"GGUF: `{gguf_file}`  |  n_gpu_layers: {n_gpu}")
            if launch_cmd:
                st.info(f"Start llama-server with:\n```bash\n{launch_cmd}\n```")

    elif local.format == "safetensors":
        st.info(
            f"Safetensors model detected. Start vLLM with:\n"
            f"```\nvllm serve {local.path}\n```"
        )
        if st.button("Register for vLLM", key="mm_local_reg_st"):
            register_model_in_config(
                engine_name=engine_name,
                engine_type="vllm",
                model_id=str(local.path),
                config_path=_config_path(),
            )
            st.success(f"Registered **{engine_name}** pointing to {local.path}.")

    elif local.format == "pytorch":
        st.info("PyTorch format detected. vLLM can serve this. Start with:\n"
                f"```\nvllm serve {local.path}\n```")
        if st.button("Register for vLLM", key="mm_local_reg_pt"):
            register_model_in_config(
                engine_name=engine_name,
                engine_type="vllm",
                model_id=str(local.path),
                config_path=_config_path(),
            )
            st.success(f"Registered **{engine_name}**.")


# ------------------------------------------------------------------
# Running models panel
# ------------------------------------------------------------------

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
        return "Endpoint is not running."

    if "timed out" in text or "timeout" in text:
        return "Endpoint timed out."

    if "name or service not known" in text or "temporary failure in name resolution" in text:
        return "Host could not be resolved."

    return "Endpoint unavailable."


def _maybe_render_model_list(models: list[str], backend: str, max_inline: int = 3) -> None:
    models = models or []
    if not models:
        if (backend or "").lower().strip() == "ollama":
            st.caption("No models available in Ollama.")
        else:
            st.caption("No models served.")
        return

    noun = "models available" if (backend or "").lower().strip() == "ollama" else "models served"

    if len(models) <= max_inline:
        st.caption(", ".join(models))
        return

    st.caption(f"{len(models)} {noun}")
    with st.expander("Show models", expanded=False):
        for m in models:
            st.write(f"`{m}`")

def _render_system_recommendations(cfg: Dict[str, Any]) -> None:
    st.subheader("Recommended for this system")

    engines_cfg = cfg.get("engines", {}) or {}

    scored = []
    for engine_name, ecfg in engines_cfg.items():
        model_text = str(ecfg.get("model") or "")
        hf_repo_id = str(ecfg.get("hf_repo_id") or "")
        local_model_dir = str(ecfg.get("local_model_dir") or "")

        fit = _score_model_fit(
            model_text,
            hf_repo_id,
            local_model_dir,
            backend=str(ecfg.get("type") or ""),
            n_gpu_layers=int(ecfg.get("n_gpu_layers") or 0),
        )
        scored.append({
            "engine_name": engine_name,
            "backend": str(ecfg.get("type") or ""),
            "model": model_text,
            "hf_repo_id": hf_repo_id,
            "local_model_dir": local_model_dir,
            **fit,
        })

    best_vllm = [r for r in scored if r["fit"] in {"good", "borderline"} and r["runtime"] == "vllm"]
    blocked_now = [r for r in scored if r["fit"] == "blocked_now"]
    better_ollama = [r for r in scored if r["fit"] == "better_ollama"]
    risky = [r for r in scored if r["fit"] == "likely_oom"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Available now with vLLM**")
        if not best_vllm:
            st.caption("No strong vLLM candidates detected.")
        for r in best_vllm[:5]:
            st.write(f"**{r['engine_name']}**")
            st.caption(r["model"] or r["hf_repo_id"] or r["local_model_dir"])
            st.caption(r["rationale"])

    with col2:
        st.markdown("**Currently blocked by loaded model**")
        if not blocked_now:
            st.caption("No models are currently blocked by loaded GPU memory.")
        for r in blocked_now[:5]:
            st.write(f"**{r['engine_name']}**")
            st.caption(r["model"] or r["hf_repo_id"] or r["local_model_dir"])
            st.caption("Good candidate for this machine, but current GPU memory is already occupied by another runtime.")

    with col3:
        st.markdown("**Better with Ollama / Too large**")

        if better_ollama:
            st.caption("These models are better candidates for Ollama on this system.")
            for r in better_ollama[:3]:
                st.write(f"**{r['engine_name']}**")
                st.caption(r["model"] or r["hf_repo_id"] or r["local_model_dir"])
                st.caption(r["rationale"])

        if risky:
            if better_ollama:
                st.write("")
            st.caption("These models are likely too large for reliable local use on the detected system.")
            for r in risky[:5]:
                st.write(f"**{r['engine_name']}**")
                st.caption(r["model"] or r["hf_repo_id"] or r["local_model_dir"])
                st.caption("Likely too large for reliable local use on the detected system.")

        if not better_ollama and not risky:
            st.caption("No Ollama-oriented or high-risk models detected.")



def _render_running_models():
    """Show live endpoints plus configured engine inventory."""
    st.subheader("Active Runtime Status")

    cfg = _load_config()
    _render_system_recommendations(cfg)
    selected_profile = st.session_state.get("ui_cfg").profile if st.session_state.get("ui_cfg") else "default_local"
    rows = _build_engine_inventory(cfg, selected_profile)

    ollama_urls = sorted({(r["endpoint"] or "").replace("/v1", "").rstrip("/") for r in rows if r["backend"] == "ollama" and r["endpoint"]})
    vllm_urls = sorted({r["endpoint"].rstrip("/") for r in rows if r["backend"] in {"vllm", "openai", "openai-compatible", "openai_compatible"} and r["endpoint"]})
    llama_cpp_urls = sorted({r["endpoint"].rstrip("/") for r in rows if r["backend"] in {"llama_cpp", "llama-cpp", "llamacpp"} and r["endpoint"]})

    live_col, active_col = st.columns([1.3, 1])
    with live_col:
        st.markdown("**Live endpoints**")
        if not ollama_urls and not vllm_urls and not llama_cpp_urls:
            st.caption("No Ollama, vLLM, or llama.cpp endpoints are configured.")

        for url in ollama_urls:
            ok, models, err = _endpoint_status(url, "ollama")
            icon = _status_icon("reachable" if ok else "unreachable")
            st.write(f"{icon} **Ollama** `{url}` — {'reachable' if ok else 'unreachable'}")

            if ok:
                _maybe_render_model_list(models, "ollama")
            else:
                st.caption(_friendly_endpoint_error(url, err))
                with st.expander("Technical details", expanded=False):
                    st.code(err or "No technical details available.", language="text")

        for url in vllm_urls:
            backend_hint = "openai" if "api.openai.com" in url else "vllm"

            ok, models, err = _endpoint_status(url, "vllm")
            icon = _status_icon("reachable" if ok else "unreachable")
            label = "OpenAI" if "api.openai.com" in url else "vLLM"
            st.write(f"{icon} **{label}** `{url}` — {'reachable' if ok else 'unreachable'}")

            if ok:
                _maybe_render_model_list(models, backend_hint)
            else:
                st.caption(_friendly_endpoint_error(url, err))
                with st.expander("Technical details", expanded=False):
                    st.code(err or "No technical details available.", language="text")

        for url in llama_cpp_urls:
            ok, models, err = _endpoint_status(url, "vllm")  # same OpenAI protocol
            icon = _status_icon("reachable" if ok else "unreachable")
            st.write(f"{icon} **llama.cpp** `{url}` — {'reachable' if ok else 'unreachable'}")
            if ok:
                _maybe_render_model_list(models, "vllm")
            else:
                st.caption(_friendly_endpoint_error(url, err))
                with st.expander("Technical details", expanded=False):
                    st.code(err or "No technical details available.", language="text")
    with active_col:
        st.markdown("**Current session binding**")
        selected = [r for r in rows if r["selected"]]

        if not selected:
            st.info("No engine is selected in the active profile.")
        else:
            row = selected[0]
            st.write(f"**Primary: {row['engine_name']}**")
            st.caption(f"backend: {row['backend']} | endpoint: {row['endpoint'] or 'n/a'}")
            st.caption(
                f"served: {'yes' if row['served'] else 'no'} | "
                f"downloaded: {'yes' if row['downloaded'] else ('n/a' if row['downloaded'] is None else 'no')}"
            )
            if row["backend"] in {"vllm", "openai-compatible", "openai_compatible", "llama_cpp", "llama-cpp", "llamacpp"}:
                _render_vllm_launch_guidance(row)

            if len(selected) > 1:
                st.markdown("**Fallback engines**")
                for idx, fb in enumerate(selected[1:], start=1):
                    st.write(f"**Fallback {idx}: {fb['engine_name']}**")
                    st.caption(f"backend: {fb['backend']} | endpoint: {fb['endpoint'] or 'n/a'}")
                    st.caption(
                        f"served: {'yes' if fb['served'] else 'no'} | "
                        f"downloaded: {'yes' if fb['downloaded'] else ('n/a' if fb['downloaded'] is None else 'no')}"
                    )

    st.divider()
    st.subheader("Configured Engines")
    if not rows:
        st.info("No engines registered. Use Search or Import below to add models.")
    else:
        for row in rows:
            with st.expander(f"{row['engine_name']} — {row['backend']} — {row['artifact_family']}", expanded=row['selected']):
                st.write(f"**Model / ref:** `{row['model'] or '(none configured)'}`")
                if row['hf_repo_id']:
                    st.caption(f"HF repo: `{row['hf_repo_id']}`")
                if row['local_model_dir']:
                    exists = _safe_exists(row['local_model_dir'])
                    st.caption(f"Local model dir: `{row['local_model_dir']}` ({'present' if exists else 'missing'})")

                fit = _score_model_fit(
                    model_text=row.get("model", ""),
                    hf_repo_id=row.get("hf_repo_id", ""),
                    local_model_dir=row.get("local_model_dir", ""),
                    backend=row.get("backend", ""),
                    n_gpu_layers=int(row.get("n_gpu_layers") or 0),
                )

                st.caption(f"GPU free: {fit['free_vram_gb']} / {fit['total_vram_gb']} GB")
                if fit["fit"] == "good":
                    st.success("Good fit")
                elif fit["fit"] == "borderline":
                    st.warning("Borderline fit")
                elif fit["fit"] == "blocked_now":
                    st.info("Would fit, but current VRAM is occupied")
                elif fit["fit"] == "better_ollama":
                    st.info("Better candidate for Ollama")
                elif fit["fit"] == "likely_oom":
                    st.error("Likely too large")
                elif fit["fit"] == "remote":
                    st.info("Remote / cloud runtime")

                st.caption(
                    f"{_fit_badge_label(fit['fit'])} | "
                    f"{_runtime_badge_label(fit['runtime'])} | "
                    f"format: {fit['artifact_family'] or 'unknown'}"
                )
                st.caption(fit["rationale"])

                st.caption(_render_status_badge('Selected in profile', row['selected']))
                st.caption(_render_status_badge('Endpoint reachable', row['reachable']))
                st.caption(_render_status_badge('Serving this model', row['served']))
                st.caption(_render_status_badge('Downloaded locally', row['downloaded']))
                st.write(f"**Runtime status:** {_runtime_state_badge(row['runtime_state'])}")
                st.caption(row["runtime_message"])

                if row['endpoint']:
                    st.caption(f"Endpoint: `{row['endpoint']}`")

                if row['served_models']:
                    with st.expander("Discovered served models", expanded=False):
                        for m in row['served_models']:
                            st.write(f"`{m}`")

                if row["backend"] in {"vllm", "openai-compatible", "openai_compatible", "llama_cpp", "llama-cpp", "llamacpp"}:
                    _render_vllm_launch_guidance(row)

                if row['runtime_state'] == "wrong_model":
                    st.warning("The endpoint is reachable, but it is not serving the model Engram expects for this engine.")

                if row['endpoint_error']:
                    st.warning(f"Endpoint status: {row['endpoint_error']}")
                
    st.divider()
    st.subheader("Downloaded Local Models")
    local_rows = [r for r in rows if r['local_model_dir']]
    if not local_rows:
        st.caption("No local model directories are registered yet.")
    else:
        for row in local_rows:
            exists = _safe_exists(row['local_model_dir'])
            st.write(f"**{row['engine_name']}** — `{row['local_model_dir']}`")
            st.caption(f"format: {row['artifact_family']} | exists: {'yes' if exists else 'no'}")
            if row['hf_repo_id']:
                st.caption(f"HF repo: `{row['hf_repo_id']}`")

            fit = _score_model_fit(
                model_text=row["engine_name"],
                hf_repo_id=row.get("hf_repo_id", ""),
                local_model_dir=row["local_model_dir"],
                backend=row.get("backend", ""),
            )
    
            st.caption(
                f"{_fit_badge_label(fit['fit'])} | "
                f"{_runtime_badge_label(fit['runtime'])}"
            )
            st.caption(fit["rationale"])

# ------------------------------------------------------------------
# Profile editor
# ------------------------------------------------------------------

def _render_profile_editor():
    """Edit failover profiles — primary/fallback model selection."""
    st.subheader("Profile Editor")
    st.caption(
        "Profiles define the engine failover chain. The first engine is **primary**; "
        "subsequent engines are fallbacks tried in order."
    )

    cfg = _load_config()
    profiles = cfg.get("profiles", {})
    engines_cfg = cfg.get("engines", {})
    all_engine_names = sorted(engines_cfg.keys())

    if not profiles:
        st.warning("No profiles found in config. Create one below.")

    # Profile selector
    profile_names = sorted(profiles.keys())
    tab_edit, tab_new = st.tabs(["Edit existing", "Create new"])

    with tab_edit:
        if not profile_names:
            st.info("No profiles to edit.")
            return

        selected = st.selectbox("Profile", profile_names, key="mm_prof_sel")
        prof = profiles.get(selected, {})
        current_engines = list(prof.get("engines", []))

        st.write("**Current engine order** (first = primary):")
        if not current_engines:
            st.write("  (empty)")
        else:
            for idx, eng in enumerate(current_engines):
                ecfg = engines_cfg.get(eng, {})
                etype = ecfg.get("type", "?")
                model = ecfg.get("model", "?")
                role = "Primary" if idx == 0 else f"Fallback {idx}"
                st.write(f"  **{idx+1}. {eng}** ({etype}) — `{model}` [{role}]")

        # Add engine to profile
        st.write("---")
        available = [e for e in all_engine_names if e not in current_engines]
        if available:
            col_add, col_pos, col_btn = st.columns([2, 1, 1])
            with col_add:
                new_eng = st.selectbox("Add engine", available, key="mm_prof_add_eng")
            with col_pos:
                position = st.selectbox("As", ["Primary", "Fallback"], key="mm_prof_add_pos")
            with col_btn:
                st.write("")  # Spacer
                if st.button("Add", key="mm_prof_add_btn"):
                    pos = "prepend" if position == "Primary" else "append"
                    if add_engine_to_profile(selected, new_eng, pos, _config_path()):
                        st.success(f"Added **{new_eng}** as {position.lower()} to **{selected}**.")
                        st.rerun()
                    else:
                        st.warning(f"{new_eng} is already in {selected}.")
        else:
            st.caption("All registered engines are already in this profile.")

        # Remove engine from profile
        if current_engines:
            col_rm, col_rm_btn = st.columns([3, 1])
            with col_rm:
                rm_eng = st.selectbox("Remove engine", current_engines, key="mm_prof_rm_eng")
            with col_rm_btn:
                st.write("")
                if st.button("Remove", key="mm_prof_rm_btn"):
                    cfg = _load_config()
                    eng_list = cfg["profiles"][selected].get("engines", [])
                    if rm_eng in eng_list:
                        eng_list.remove(rm_eng)
                        _save_config(cfg)
                        st.success(f"Removed **{rm_eng}** from **{selected}**.")
                        st.rerun()

        # Cloud failover toggle
        allow_cloud = st.toggle(
            "Allow cloud failover",
            value=bool(prof.get("allow_cloud_failover", False)),
            key="mm_prof_cloud",
            help="If enabled, the router may fail over to cloud engines.",
        )
        if allow_cloud != bool(prof.get("allow_cloud_failover", False)):
            cfg = _load_config()
            cfg["profiles"][selected]["allow_cloud_failover"] = allow_cloud
            _save_config(cfg)

    with tab_new:
        new_name = st.text_input("New profile name", key="mm_prof_new_name",
                                  placeholder="e.g., my_local, fast_cpu")

        # Let user pick engines from the registered list
        if all_engine_names:
            primary = st.selectbox("Primary engine", all_engine_names, key="mm_prof_new_primary")
            fallbacks = st.multiselect(
                "Fallback engines (in order)",
                [e for e in all_engine_names if e != primary],
                key="mm_prof_new_fallbacks",
            )
        else:
            st.warning("No engines registered. Add a model first.")
            primary = None
            fallbacks = []

        if st.button("Create profile", key="mm_prof_new_btn", disabled=not new_name or not primary):
            cfg = _load_config()
            eng_list = [primary] + fallbacks
            cfg.setdefault("profiles", {})[new_name] = {
                "engines": eng_list,
                "allow_cloud_failover": False,
                "max_attempts": 4,
            }
            _save_config(cfg)
            st.success(f"Created profile **{new_name}** with engines: {eng_list}")
            st.rerun()


# ------------------------------------------------------------------
# Registered engines panel
# ------------------------------------------------------------------

def _render_registered_engines():
    """Show and manage registered engine entries."""
    st.subheader("Engine Registry")

    cfg = _load_config()
    engines_cfg = cfg.get("engines", {})

    if not engines_cfg:
        st.info("No engines registered. Use Search or Import above to add models.")
        return

    for name, ecfg in sorted(engines_cfg.items()):
        etype = ecfg.get("type", "?")
        model = ecfg.get("model", "?")
        base_url = ecfg.get("base_url", "")
        max_ctx = ecfg.get("max_context", "?")
        num_gpu = ecfg.get("num_gpu")

        detail = f"{etype} · `{model}` · ctx={max_ctx}"
        if num_gpu is not None:
            detail += f" · gpu_layers={num_gpu}"
        if base_url:
            detail += f" · {base_url}"

        col_info, col_del = st.columns([5, 1])
        with col_info:
            st.write(f"**{name}** — {detail}")
        with col_del:
            if st.button("Delete", key=f"mm_eng_del_{name}"):
                cfg = _load_config()
                cfg.get("engines", {}).pop(name, None)
                # Also remove from any profiles
                for pname, pcfg in cfg.get("profiles", {}).items():
                    eng_list = pcfg.get("engines", [])
                    if name in eng_list:
                        eng_list.remove(name)
                _save_config(cfg)
                st.success(f"Deleted engine **{name}**.")
                st.rerun()


# ------------------------------------------------------------------
# Main render entry point (called from app.py)
# ------------------------------------------------------------------

def render_model_management():
    """Render the full model management page."""
    st.header("Models")
    st.caption("Track what is configured, downloaded, running, and selected. This page is organized as an operations console, not just a picker.")

    _render_hardware()
    st.divider()

    _render_running_models()
    st.divider()

    st.subheader("Search & Add Models")
    _render_hf_search()
    st.divider()

    _render_local_import()
    st.divider()

    _render_registered_engines()
    st.divider()

    _render_profile_editor()
