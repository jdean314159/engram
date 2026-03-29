"""Model management for Engram.

Handles model discovery, sizing, format selection, and registration.
Works in three modes:

  1. Online:    Search HuggingFace, download models, auto-detect format.
  2. Offline:   Point at a local directory or GGUF file.
  3. Running:   Discover models already served by Ollama / vLLM.

Key decisions this module helps with:
  - vLLM vs Ollama:  model fits in VRAM entirely? → vLLM (faster, logprobs).
                      Doesn't fit? → Ollama (partial offload via num_gpu).
  - GGUF vs safetensors: GGUF for Ollama, safetensors/HF for vLLM.

Dependencies:
  - stdlib only for core functionality
  - huggingface_hub (optional) for richer HF search
  - torch (optional) for VRAM detection

Author: Jeffrey Dean
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


QUANTIZATION_TERMS = ("awq", "gptq", "gguf", "exl2", "int4", "4bit", "4-bit", "q4", "q5", "q8")


def _looks_quantized(text: str) -> bool:
    low = (text or "").lower()
    return any(term in low for term in QUANTIZATION_TERMS)


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _quantized_queries(query: str) -> List[str]:
    q = _normalize_query(query)
    if not q:
        return []
    variants = [q]
    if not _looks_quantized(q):
        variants.extend([f"{q} AWQ", f"{q} GPTQ", f"{q} GGUF"])
        if "a3b" not in q.lower():
            variants.append(f"{q} A3B AWQ")
    # preserve order, remove duplicates
    out = []
    seen = set()
    for item in variants:
        key = item.lower()
        if key not in seen:
            out.append(item)
            seen.add(key)
    return out


def _model_text_blob(info: "HFModelInfo") -> str:
    return " ".join([info.repo_id, info.name, info.author, info.pipeline_tag, " ".join(info.tags)]).lower()


def _populate_format_signals(info: "HFModelInfo") -> None:
    text = _model_text_blob(info)
    info.has_gguf = info.has_gguf or ("gguf" in text)
    info.has_awq = info.has_awq or ("awq" in text)
    info.has_gptq = info.has_gptq or ("gptq" in text)
    info.has_exl2 = info.has_exl2 or ("exl2" in text)
    if info.has_gguf and not info.has_safetensors and not (info.has_awq or info.has_gptq or info.has_exl2):
        info.has_safetensors = False


def classify_hf_model_format(info: "HFModelInfo") -> str:
    """Classify the dominant artifact family for a HF result.

    Returns one of: gguf, transformers_quantized, mixed, base.
    """
    _populate_format_signals(info)
    has_transformers_quant = info.has_awq or info.has_gptq or info.has_exl2
    if info.has_gguf and has_transformers_quant:
        return "mixed"
    if info.has_gguf:
        return "gguf"
    if has_transformers_quant:
        return "transformers_quantized"
    return "base"


def _preferred_transformers_quant(info: "HFModelInfo") -> str:
    _populate_format_signals(info)
    if info.has_gptq:
        return "gptq_int4"
    if info.has_awq:
        return "awq"
    if info.has_exl2:
        return "exl2"
    return "awq"


def _search_score(info: HFModelInfo, query: str, prefer_quantized: bool, vram_gb: Optional[float]) -> float:
    haystack = _model_text_blob(info)
    tokens = [t for t in re.split(r"[^a-z0-9\.]+", (query or "").lower()) if t]
    score = 0.0
    for tok in tokens:
        if info.repo_id.lower() == tok or info.name.lower() == tok:
            score += 60
        elif tok in info.repo_id.lower():
            score += 20
        elif tok in haystack:
            score += 8
    score += min(math.log10((info.downloads or 0) + 1) * 6, 30)
    score += min(math.log10((info.likes or 0) + 1) * 4, 16)
    _populate_format_signals(info)
    quantized = _looks_quantized(haystack) or info.has_gguf or info.has_awq or info.has_gptq or info.has_exl2
    if quantized:
        score += 16 if prefer_quantized else 8
    if vram_gb and info.estimated_size_gb_fp16:
        fp16 = info.estimated_size_gb_fp16
        if fp16 <= vram_gb * 0.85:
            score += 10
        elif quantized and fp16 > vram_gb * 0.85:
            score += 18
    if info.pipeline_tag in ("text-generation", "text2text-generation", "conversational"):
        score += 6
    return score


def _merge_rank_results(results: List[HFModelInfo], query: str, prefer_quantized: bool, vram_gb: Optional[float], limit: int) -> List[HFModelInfo]:
    by_repo: Dict[str, HFModelInfo] = {}
    for info in results:
        existing = by_repo.get(info.repo_id)
        if existing is None:
            by_repo[info.repo_id] = info
            continue
        # merge richer metadata
        existing.downloads = max(existing.downloads, info.downloads)
        existing.likes = max(existing.likes, info.likes)
        existing.pipeline_tag = existing.pipeline_tag or info.pipeline_tag
        existing.tags = list(dict.fromkeys([*existing.tags, *info.tags]))
        existing.has_gguf = existing.has_gguf or info.has_gguf
        existing.has_safetensors = existing.has_safetensors or info.has_safetensors
        existing.has_awq = existing.has_awq or info.has_awq
        existing.has_gptq = existing.has_gptq or info.has_gptq
        existing.has_exl2 = existing.has_exl2 or info.has_exl2
        existing.gguf_files = existing.gguf_files or info.gguf_files
        existing.estimated_params_b = existing.estimated_params_b or info.estimated_params_b
        existing.estimated_size_gb_fp16 = existing.estimated_size_gb_fp16 or info.estimated_size_gb_fp16
    ranked = sorted(by_repo.values(), key=lambda m: _search_score(m, query, prefer_quantized, vram_gb), reverse=True)
    return ranked[:limit]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Detected GPU information."""
    name: str
    vram_gb: float
    index: int = 0
    accelerator: str = "cuda"  # cuda | mps | cpu


@dataclass
class SystemInfo:
    """System hardware summary."""
    ram_gb: Optional[float] = None
    gpus: List[GPUInfo] = field(default_factory=list)
    total_vram_gb: float = 0.0
    accelerator: str = "cpu"
    warning: Optional[str] = None

    @property
    def primary_vram_gb(self) -> float:
        """VRAM of the primary (largest) GPU."""
        if not self.gpus:
            return 0.0
        return max(g.vram_gb for g in self.gpus)


@dataclass
class HFModelInfo:
    """Model info from HuggingFace."""
    repo_id: str
    name: str
    author: str = ""
    downloads: int = 0
    likes: int = 0
    pipeline_tag: str = ""
    tags: List[str] = field(default_factory=list)
    siblings: List[Dict[str, Any]] = field(default_factory=list)

    # Derived
    has_gguf: bool = False
    has_safetensors: bool = False
    has_awq: bool = False
    has_gptq: bool = False
    has_exl2: bool = False
    gguf_files: List[Dict[str, Any]] = field(default_factory=list)
    estimated_params_b: Optional[float] = None  # Billions
    estimated_size_gb_fp16: Optional[float] = None

    def size_str(self) -> str:
        if self.estimated_params_b:
            return f"{self.estimated_params_b:.1f}B"
        return "unknown"


@dataclass
class FormatRecommendation:
    """Recommended engine/format for a model given hardware constraints."""
    engine: str               # "vllm" | "ollama" | "llama_cpp"
    format: str               # "safetensors" | "gguf"
    quantization: str         # "fp16" | "awq" | "Q4_K_M" | "Q5_K_M" | "Q8_0" | ...
    fits_in_vram: bool        # Whether the model fits entirely in GPU memory
    estimated_vram_gb: float  # Estimated VRAM usage
    num_gpu_layers: Optional[int] = None  # For Ollama/llama.cpp partial offload
    gguf_path: Optional[str] = None       # For llama_cpp: local GGUF file path
    reason: str = ""


@dataclass
class LocalModelInfo:
    """A model found on the local filesystem."""
    path: Path
    name: str
    format: str               # "gguf" | "safetensors" | "pytorch" | "unknown"
    size_gb: float
    files: List[str] = field(default_factory=list)



@dataclass
class DownloadResult:
    """Result of downloading a Hugging Face model snapshot locally."""
    repo_id: str
    local_dir: Path
    method: str
    success: bool
    revision: Optional[str] = None
    error: str = ""


def default_model_root() -> Path:
    """Default Engram-managed local model directory."""
    root = Path("~/.engram/models").expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_local_model_dir(repo_id: str, root: Optional[Path] = None) -> Path:
    """Return the Engram-managed target directory for a repo id."""
    base = Path(root).expanduser() if root is not None else default_model_root()
    parts = [part for part in str(repo_id).split("/") if part]
    if not parts:
        raise ValueError("repo_id must not be empty")
    path = base
    for part in parts:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", part)
        path = path / safe
    return path


def download_hf_model(
    repo_id: str,
    local_dir: Optional[Path] = None,
    *,
    revision: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> DownloadResult:
    """Download a model snapshot into an Engram-managed local directory.

    Prefers huggingface_hub.snapshot_download when available. Falls back to
    the ``huggingface-cli download`` command if installed. The target
    directory defaults to ``~/.engram/models/<author>/<repo>``.
    """
    target = Path(local_dir).expanduser() if local_dir is not None else default_local_model_dir(repo_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(target),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            local_dir_use_symlinks=False,
        )
        return DownloadResult(repo_id=repo_id, local_dir=target, method="huggingface_hub", success=True, revision=revision)
    except ImportError:
        pass
    except Exception as e:
        logger.warning("huggingface_hub download failed for %s: %s", repo_id, e)

    cmd = ["huggingface-cli", "download", repo_id, "--local-dir", str(target)]
    if revision:
        cmd.extend(["--revision", revision])
    if allow_patterns:
        for pattern in allow_patterns:
            cmd.extend(["--include", pattern])
    if ignore_patterns:
        for pattern in ignore_patterns:
            cmd.extend(["--exclude", pattern])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
        if proc.returncode == 0:
            return DownloadResult(repo_id=repo_id, local_dir=target, method="huggingface-cli", success=True, revision=revision)
        error = (proc.stderr or proc.stdout or "download failed").strip()
        return DownloadResult(repo_id=repo_id, local_dir=target, method="huggingface-cli", success=False, revision=revision, error=error)
    except FileNotFoundError:
        return DownloadResult(
            repo_id=repo_id,
            local_dir=target,
            method="huggingface-cli",
            success=False,
            revision=revision,
            error="Neither huggingface_hub nor huggingface-cli is available.",
        )
    except Exception as e:
        return DownloadResult(repo_id=repo_id, local_dir=target, method="huggingface-cli", success=False, revision=revision, error=str(e))


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_system() -> SystemInfo:
    """Detect system hardware: RAM, GPUs, VRAM.

    Prefer torch when it can initialize an accelerator, but fall back to
    ``nvidia-smi`` whenever torch is missing or CUDA init fails. This keeps
    the UI useful after Linux/NVIDIA suspend-resume failures where the driver
    still sees the GPU but torch cannot create a CUDA context.
    """
    info = SystemInfo()

    # RAM
    info.ram_gb = _get_ram_gb()

    torch_gpu_found = False
    torch_cuda_error: Optional[str] = None

    try:
        import torch
        try:
            if torch.cuda.is_available():
                info.accelerator = "cuda"
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    vram = props.total_memory / (1024 ** 3)
                    info.gpus.append(GPUInfo(
                        name=props.name,
                        vram_gb=round(vram, 1),
                        index=i,
                        accelerator="cuda",
                    ))
                torch_gpu_found = len(info.gpus) > 0
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                info.accelerator = "mps"
                if info.ram_gb:
                    effective = max(0.0, min(info.ram_gb * 0.75, info.ram_gb - 4.0))
                    info.gpus.append(GPUInfo(
                        name="Apple Silicon (unified)",
                        vram_gb=round(effective, 1),
                        index=0,
                        accelerator="mps",
                    ))
                    torch_gpu_found = True
        except Exception as exc:
            torch_cuda_error = str(exc)
            logger.warning("Torch accelerator detection failed: %s", exc)
            torch_gpu_found = False
    except ImportError:
        torch_gpu_found = False

    if not torch_gpu_found:
        smi_gpus = _detect_gpus_nvidia_smi()
        if smi_gpus:
            info.gpus = smi_gpus
            info.accelerator = "cuda"
            if torch_cuda_error:
                info.warning = (
                    "GPU detected via nvidia-smi, but PyTorch could not initialize CUDA in this process. "
                    "On some Linux + NVIDIA systems this happens after suspend/resume. "
                    "Close GPU-using apps, reload nvidia_uvm, or reboot before retrying model startup."
                )

    info.total_vram_gb = sum(g.vram_gb for g in info.gpus)
    return info


def _get_ram_gb() -> Optional[float]:
    """Cross-platform RAM detection."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    import sys
    if sys.platform.startswith("linux"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((pages * page_size) / (1024 ** 3), 1)
        except Exception:
            return None
    return None


def _detect_gpus_nvidia_smi() -> List[GPUInfo]:
    """Fallback GPU detection via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
    except Exception:
        return []
    gpus = []
    for line in out.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append(GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_gb=round(float(parts[2]) / 1024, 1),
                    accelerator="cuda",
                ))
            except (ValueError, IndexError):
                continue
    return gpus


# ---------------------------------------------------------------------------
# HuggingFace model search
# ---------------------------------------------------------------------------

def search_hf_models(
    query: str,
    task: Optional[str] = None,
    limit: int = 20,
    sort: str = "downloads",
    include_quantized: bool = True,
    vram_gb: Optional[float] = None,
) -> List[HFModelInfo]:
    """Search HuggingFace for models.

    Uses huggingface_hub if available, falls back to REST API.
    Returns empty list if offline / no network. Search is intentionally
    broader than a strict pipeline-tag match so quantized repos are not lost.
    """
    queries = _quantized_queries(query) if include_quantized else [_normalize_query(query)]
    results: List[HFModelInfo] = []
    for q in queries:
        if not q:
            continue
        try:
            results.extend(_search_hf_hub(q, task, limit, sort))
        except ImportError:
            results.extend(_search_hf_api(q, task, limit, sort))
        except Exception:
            results.extend(_search_hf_api(q, task, limit, sort))
    return _merge_rank_results(results, query, include_quantized, vram_gb, limit)


def _search_hf_hub(query, task, limit, sort) -> List[HFModelInfo]:
    """Search via huggingface_hub library (richer metadata)."""
    from huggingface_hub import HfApi
    api = HfApi()
    kwargs = dict(search=query, sort=sort, direction=-1, limit=limit)
    if task:
        kwargs["pipeline_tag"] = task
    models = api.list_models(**kwargs)
    results = []
    for m in models:
        info = HFModelInfo(
            repo_id=m.id or "",
            name=(m.id or "").split("/")[-1],
            author=(m.id or "").split("/")[0] if "/" in (m.id or "") else "",
            downloads=getattr(m, "downloads", 0) or 0,
            likes=getattr(m, "likes", 0) or 0,
            pipeline_tag=getattr(m, "pipeline_tag", "") or "",
            tags=list(getattr(m, "tags", []) or []),
        )
        # Check for file formats in siblings
        siblings = getattr(m, "siblings", []) or []
        for s in siblings:
            fname = getattr(s, "rfilename", "") or ""
            if fname.endswith(".gguf"):
                info.has_gguf = True
                size = getattr(s, "size", None)
                info.gguf_files.append({"filename": fname, "size": size})
            elif fname.endswith(".safetensors"):
                info.has_safetensors = True

        _populate_format_signals(info)
        _estimate_params(info)
        results.append(info)
    return results


def _search_hf_api(query, task, limit, sort) -> List[HFModelInfo]:
    """Search via HuggingFace REST API (no dependencies)."""
    url = (
        f"https://huggingface.co/api/models?"
        f"search={urllib.request.quote(query)}"
        f"&sort={sort}"
        f"&direction=-1"
        f"&limit={limit}"
    )
    if task:
        url += f"&pipeline_tag={urllib.request.quote(task)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "engram"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning("HuggingFace API search failed: %s", e)
        return []

    results = []
    for item in data:
        info = HFModelInfo(
            repo_id=item.get("id", ""),
            name=item.get("id", "").split("/")[-1],
            author=item.get("id", "").split("/")[0] if "/" in item.get("id", "") else "",
            downloads=item.get("downloads", 0) or 0,
            likes=item.get("likes", 0) or 0,
            pipeline_tag=item.get("pipeline_tag", "") or "",
            tags=item.get("tags", []) or [],
        )
        # REST API doesn't include siblings by default; would need a
        # second call per model.  Flag GGUF from repo name heuristic.
        repo_lower = info.repo_id.lower()
        info.has_gguf = "gguf" in repo_lower
        info.has_safetensors = not info.has_gguf  # Heuristic

        _populate_format_signals(info)
        _estimate_params(info)
        results.append(info)
    return results


def _estimate_params(info: HFModelInfo):
    """Best-effort parameter count estimation from repo name/tags."""
    text = (info.repo_id + " " + " ".join(info.tags)).lower()

    # Look for patterns like "7b", "13b", "70b", "0.5b", "1.5b"
    match = re.search(r"(\d+\.?\d*)\s*b(?:illion)?(?:\b|[-_])", text)
    if match:
        info.estimated_params_b = float(match.group(1))
        # fp16: ~2 bytes/param
        info.estimated_size_gb_fp16 = info.estimated_params_b * 2.0
        return

    # Check for parameter count in tags
    for tag in info.tags:
        if "params" in tag.lower():
            m2 = re.search(r"(\d+\.?\d*)", tag)
            if m2:
                val = float(m2.group(1))
                if val > 100:  # Likely millions
                    info.estimated_params_b = val / 1000
                else:
                    info.estimated_params_b = val
                info.estimated_size_gb_fp16 = info.estimated_params_b * 2.0
                return


# ---------------------------------------------------------------------------
# Format recommendation
# ---------------------------------------------------------------------------

# GGUF quantization: approximate bytes per parameter
GGUF_QUANT_BPP = {
    "Q2_K":   0.31,
    "Q3_K_S": 0.38,
    "Q3_K_M": 0.42,
    "Q4_0":   0.50,
    "Q4_K_S": 0.53,
    "Q4_K_M": 0.56,
    "Q5_0":   0.63,
    "Q5_K_S": 0.65,
    "Q5_K_M": 0.68,
    "Q6_K":   0.78,
    "Q8_0":   1.00,
    "fp16":   2.00,
}

# KV cache overhead rule of thumb: ~10-15% on top of model weights
KV_OVERHEAD_FACTOR = 1.15


def recommend_format(
    params_b: float,
    vram_gb: float,
    prefer_quality: bool = False,
) -> FormatRecommendation:
    """Recommend engine/format/quantization given model size and VRAM.

    Args:
        params_b: Model parameter count in billions.
        vram_gb: Available VRAM in GB (primary GPU).
        prefer_quality: If True, prefer higher quantization even if slower.

    Decision tree:
        1. fp16 fits in VRAM? → vLLM (best speed + logprobs)
        2. AWQ/GPTQ 4-bit fits? → vLLM with quantized weights
        3. GGUF Q5_K_M fits? → Ollama (good quality/speed balance)
        4. GGUF Q4_K_M fits? → Ollama (reasonable quality)
        5. Nothing fits? → Ollama with partial GPU offload
    """
    if vram_gb <= 0:
        # CPU-only
        quant = "Q4_K_M"
        size = params_b * GGUF_QUANT_BPP[quant]
        return FormatRecommendation(
            engine="ollama", format="gguf", quantization=quant,
            fits_in_vram=False, estimated_vram_gb=0,
            num_gpu_layers=0,
            reason=f"No GPU detected. Using Ollama CPU-only with {quant}.",
        )

    # fp16 in VRAM? (vLLM)
    fp16_size = params_b * 2.0 * KV_OVERHEAD_FACTOR
    if fp16_size <= vram_gb * 0.9:  # 10% headroom
        return FormatRecommendation(
            engine="vllm", format="safetensors", quantization="fp16",
            fits_in_vram=True, estimated_vram_gb=round(fp16_size, 1),
            reason=f"Model fits in VRAM at fp16 ({fp16_size:.1f}GB / {vram_gb:.1f}GB). vLLM recommended.",
        )

    # AWQ 4-bit in VRAM? (vLLM)
    awq_size = params_b * 0.56 * KV_OVERHEAD_FACTOR  # AWQ ≈ 4-bit
    if awq_size <= vram_gb * 0.9:
        return FormatRecommendation(
            engine="vllm", format="safetensors", quantization="awq",
            fits_in_vram=True, estimated_vram_gb=round(awq_size, 1),
            reason=f"Model fits in VRAM with AWQ quantization ({awq_size:.1f}GB / {vram_gb:.1f}GB).",
        )

    # GGUF quantizations for Ollama
    quant_preference = (
        ["Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
        if not prefer_quality else
        ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
    )

    for quant in quant_preference:
        bpp = GGUF_QUANT_BPP[quant]
        size = params_b * bpp * KV_OVERHEAD_FACTOR
        if size <= vram_gb * 0.9:
            return FormatRecommendation(
                engine="ollama", format="gguf", quantization=quant,
                fits_in_vram=True, estimated_vram_gb=round(size, 1),
                reason=f"Model fits in VRAM as GGUF {quant} ({size:.1f}GB / {vram_gb:.1f}GB).",
            )

    # Partial offload: GGUF Q4_K_M, split across GPU and CPU
    # Use llama.cpp (llama-server) rather than Ollama so the user gets
    # precise --n-gpu-layers control and avoids Ollama's layer-count heuristic.
    quant = "Q4_K_M"
    full_size = params_b * GGUF_QUANT_BPP[quant] * KV_OVERHEAD_FACTOR
    total_layers = max(1, int(params_b * 2))
    frac = min(1.0, (vram_gb * 0.85) / full_size) if full_size > 0 else 0
    gpu_layers = max(0, int(total_layers * frac))

    return FormatRecommendation(
        engine="llama_cpp", format="gguf", quantization=quant,
        fits_in_vram=False, estimated_vram_gb=round(vram_gb * frac, 1),
        num_gpu_layers=gpu_layers,
        reason=(
            f"Model too large for full GPU offload ({full_size:.1f} GB > {vram_gb:.1f} GB). "
            f"llama.cpp split offload: {gpu_layers}/{total_layers} layers on GPU, "
            f"remainder on CPU. Use llama-server with --n-gpu-layers {gpu_layers}."
        ),
    )


def recommend_format_for_hf_model(
    model: HFModelInfo,
    vram_gb: float,
    prefer_quality: bool = False,
) -> FormatRecommendation:
    """Recommend backend/quantization for a specific HF result.

    Unlike ``recommend_format()``, this uses artifact-family signals from the
    exact Hugging Face result so GPTQ/AWQ models are not misrouted to Ollama
    and GGUF models are not misrouted to vLLM.
    """
    if model.estimated_params_b is None:
        raise ValueError("Model parameter estimate required for recommendation")

    family = classify_hf_model_format(model)
    if family == "gguf":
        quant_preference = (
            ["Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
            if not prefer_quality else
            ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
        )
        if vram_gb <= 0:
            quant = "Q4_K_M"
            return FormatRecommendation(
                engine="ollama", format="gguf", quantization=quant,
                fits_in_vram=False, estimated_vram_gb=0, num_gpu_layers=0,
                reason=(
                    "No GPU detected. CPU-only inference. "
                    "Ollama (simple) or llama.cpp/llama-server (more control) both work. "
                    "For llama-server: set n_gpu_layers=0 in the engine config."
                ),
            )
        for quant in quant_preference:
            size = model.estimated_params_b * GGUF_QUANT_BPP[quant] * KV_OVERHEAD_FACTOR
            if size <= vram_gb * 0.9:
                return FormatRecommendation(
                    engine="ollama", format="gguf", quantization=quant,
                    fits_in_vram=True, estimated_vram_gb=round(size,1),
                    reason=f"Repo is GGUF-oriented. {quant} fits in VRAM ({size:.1f}GB / {vram_gb:.1f}GB). Prefer Ollama.",
                )
        quant = "Q4_K_M"
        full_size = model.estimated_params_b * GGUF_QUANT_BPP[quant] * KV_OVERHEAD_FACTOR
        total_layers = max(1, int(model.estimated_params_b * 2))
        frac = min(1.0, (vram_gb * 0.85) / full_size) if full_size > 0 else 0
        gpu_layers = max(0, int(total_layers * frac))
        return FormatRecommendation(
            engine="llama_cpp", format="gguf", quantization=quant,
            fits_in_vram=False, estimated_vram_gb=round(vram_gb * frac, 1),
            num_gpu_layers=gpu_layers,
            reason=(
                f"Repo is GGUF-oriented. Full GPU offload will not fit "
                f"({full_size:.1f} GB > {vram_gb:.1f} GB). "
                f"llama.cpp split offload: {gpu_layers}/{total_layers} layers on GPU. "
                f"Download the GGUF file and use llama-server with --n-gpu-layers {gpu_layers}."
            ),
        )

    if family == "transformers_quantized":
        quant = _preferred_transformers_quant(model)
        size = model.estimated_params_b * 0.56 * KV_OVERHEAD_FACTOR
        fits = vram_gb > 0 and size <= vram_gb * 0.9
        reason = (
            f"Repo signals {quant.upper()} / Transformers-style weights. "
            f"Prefer vLLM over Ollama for this artifact family."
        )
        if fits:
            reason += f" Estimated VRAM {size:.1f}GB / {vram_gb:.1f}GB."
        else:
            reason += f" Estimated VRAM {size:.1f}GB exceeds {vram_gb:.1f}GB; full-GPU vLLM may not fit."
        return FormatRecommendation(
            engine="vllm",
            format="safetensors",
            quantization=quant,
            fits_in_vram=fits,
            estimated_vram_gb=round(size if fits else min(size, vram_gb), 1),
            reason=reason,
        )

    if family == "mixed":
        # Mixed repo naming/artifacts: prefer transformers-style runtime when AWQ/GPTQ/EXL2
        # is explicitly present, but tell the user the repo looks ambiguous.
        quant = _preferred_transformers_quant(model)
        size = model.estimated_params_b * 0.56 * KV_OVERHEAD_FACTOR
        fits = vram_gb > 0 and size <= vram_gb * 0.9
        return FormatRecommendation(
            engine="vllm",
            format="safetensors",
            quantization=quant,
            fits_in_vram=fits,
            estimated_vram_gb=round(size if fits else min(size, vram_gb), 1),
            reason=(
                f"Repo name/artifacts look mixed (GGUF plus {quant.upper()}). "
                f"Recommending vLLM because the selected result explicitly advertises a Transformers-style quantized format. "
                f"Verify the files in the repo before downloading."
            ),
        )

    return recommend_format(model.estimated_params_b, vram_gb, prefer_quality=prefer_quality)


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def download_ollama_model(model_tag: str, base_url: str = "http://localhost:11434") -> bool:
    """Pull a model via Ollama's API.

    Args:
        model_tag: e.g. "qwen3:8b", "llama3:70b-q4_K_M"
        base_url: Ollama server URL

    Returns True if successful.
    """
    url = f"{base_url.rstrip('/')}/api/pull"
    payload = json.dumps({"name": model_tag, "stream": False}).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode())
            status = result.get("status", "")
            logger.info("Ollama pull %s: %s", model_tag, status)
            return "success" in status.lower() or status == ""
    except Exception as e:
        logger.error("Ollama pull failed for %s: %s", model_tag, e)
        return False


def import_gguf_to_ollama(
    gguf_path: Path,
    model_name: str,
    base_url: str = "http://localhost:11434",
) -> bool:
    """Import a local GGUF file into Ollama.

    Creates a Modelfile and runs `ollama create`.
    """
    gguf_path = Path(gguf_path).resolve()
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    modelfile = f'FROM "{gguf_path}"\n'
    modelfile_path = gguf_path.parent / f"{model_name}.Modelfile"
    modelfile_path.write_text(modelfile)

    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            logger.info("Imported %s to Ollama as '%s'", gguf_path.name, model_name)
            return True
        else:
            logger.error("ollama create failed: %s", result.stderr)
            return False
    except FileNotFoundError:
        logger.error("ollama command not found. Is Ollama installed?")
        return False
    finally:
        modelfile_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Local model discovery
# ---------------------------------------------------------------------------

def scan_local_model(path: Path) -> Optional[LocalModelInfo]:
    """Examine a local path and determine model format.

    Args:
        path: Directory containing model files, or a single GGUF file.

    Returns LocalModelInfo or None if not a recognized model.
    """
    path = Path(path).resolve()

    if path.is_file() and path.suffix == ".gguf":
        return LocalModelInfo(
            path=path,
            name=path.stem,
            format="gguf",
            size_gb=round(path.stat().st_size / (1024**3), 2),
            files=[path.name],
        )

    if not path.is_dir():
        return None

    files = list(path.iterdir())
    fnames = [f.name for f in files if f.is_file()]

    # Check for GGUF files
    gguf_files = [f for f in fnames if f.endswith(".gguf")]
    if gguf_files:
        total_size = sum(
            (path / f).stat().st_size for f in gguf_files
        )
        return LocalModelInfo(
            path=path,
            name=path.name,
            format="gguf",
            size_gb=round(total_size / (1024**3), 2),
            files=gguf_files,
        )

    # Check for safetensors
    st_files = [f for f in fnames if f.endswith(".safetensors")]
    if st_files:
        total_size = sum(
            (path / f).stat().st_size for f in st_files
        )
        return LocalModelInfo(
            path=path,
            name=path.name,
            format="safetensors",
            size_gb=round(total_size / (1024**3), 2),
            files=st_files,
        )

    # Check for PyTorch bin files
    pt_files = [f for f in fnames if f.endswith(".bin") and "pytorch" in f.lower()]
    if pt_files:
        total_size = sum((path / f).stat().st_size for f in pt_files)
        return LocalModelInfo(
            path=path,
            name=path.name,
            format="pytorch",
            size_gb=round(total_size / (1024**3), 2),
            files=pt_files,
        )

    return None


# ---------------------------------------------------------------------------
# Engine YAML registration
# ---------------------------------------------------------------------------

def register_model_in_config(
    engine_name: str,
    engine_type: str,
    model_id: str,
    config_path: Optional[Path] = None,
    base_url: Optional[str] = None,
    max_context: int = 8192,
    num_gpu: Optional[int] = None,
    system_prompt: str = "You are a helpful AI assistant.",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Register a model as a new engine entry in llm_engines.yaml.

    Merges into existing config without overwriting other entries.
    Returns the path to the config file.
    """
    import yaml

    if config_path is None:
        config_path = Path("~/.engram/llm_engines.yaml").expanduser()

    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    config: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

    engines = config.setdefault("engines", {})

    entry: Dict[str, Any] = {
        "type": engine_type,
        "model": model_id,
        "timeout": 120 if engine_type == "ollama" else 60,
        "max_context": max_context,
        "system_prompt": system_prompt,
        "compression_strategy": "compress",
        "max_retries": 1,
    }

    if base_url:
        entry["base_url"] = base_url
    elif engine_type == "ollama":
        entry["base_url"] = "http://localhost:11434/v1"
    elif engine_type == "vllm":
        entry["base_url"] = "http://localhost:8000/v1"
    elif engine_type in ("llama_cpp", "llama-cpp", "llamacpp"):
        entry["base_url"] = "http://127.0.0.1:8080/v1"
        entry["timeout"] = 300   # CPU inference is slower

    if engine_type in ("llama_cpp", "llama-cpp", "llamacpp"):
        entry["n_gpu_layers"] = num_gpu if num_gpu is not None else 0
    elif num_gpu is not None:
        entry["num_gpu"] = num_gpu

    if extra:
        entry.update(extra)

    engines[engine_name] = entry

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Registered engine '%s' in %s", engine_name, config_path)
    return config_path


def add_engine_to_profile(
    profile_name: str,
    engine_name: str,
    position: str = "append",
    config_path: Optional[Path] = None,
) -> bool:
    """Add an engine to a failover profile.

    Args:
        profile_name: Profile to modify (e.g., "default_local")
        engine_name: Engine to add
        position: "prepend" (primary) or "append" (fallback)
        config_path: YAML config path

    Returns True if modified.
    """
    import yaml

    if config_path is None:
        config_path = Path("~/.engram/llm_engines.yaml").expanduser()

    if not config_path.exists():
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        profiles[profile_name] = {"engines": [], "allow_cloud_failover": False, "max_attempts": 4}

    engines_list = profiles[profile_name].setdefault("engines", [])
    if engine_name in engines_list:
        return False  # Already present

    if position == "prepend":
        engines_list.insert(0, engine_name)
    else:
        engines_list.append(engine_name)

    config["profiles"] = profiles

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Added '%s' to profile '%s' (%s)", engine_name, profile_name, position)
    return True


# ---------------------------------------------------------------------------
# High-level: recommend + register workflow
# ---------------------------------------------------------------------------

def recommend_and_register(
    repo_id: str,
    system: Optional[SystemInfo] = None,
    config_path: Optional[Path] = None,
    prefer_quality: bool = False,
) -> Dict[str, Any]:
    """End-to-end: search model → size → recommend → generate config entry.

    Does NOT download or pull. Returns a dict with the recommendation
    and a ready-to-register engine config.

    This is what the UI calls to get a recommendation before the user
    confirms the download.
    """
    if system is None:
        system = detect_system()

    # Try to get model info from HF
    models = search_hf_models(repo_id, limit=1)
    if not models:
        return {"error": f"Model '{repo_id}' not found on HuggingFace"}

    model = models[0]
    if model.estimated_params_b is None:
        return {
            "error": f"Cannot estimate size for '{repo_id}'. "
                     f"Specify parameter count manually.",
            "model": asdict(model),
        }

    rec = recommend_format_for_hf_model(
        model,
        system.primary_vram_gb,
        prefer_quality=prefer_quality,
    )

    # Generate a clean engine name
    safe_name = re.sub(r"[^a-z0-9_]", "_", model.name.lower())
    if rec.quantization != "fp16":
        safe_name += f"_{rec.quantization.lower()}"

    return {
        "model": asdict(model),
        "system": asdict(system),
        "recommendation": asdict(rec),
        "suggested_engine_name": safe_name,
        "suggested_model_id": (
            model.repo_id if rec.engine == "vllm"
            else f"{model.name.lower().replace('-', '_')}:{rec.quantization.lower()}"
        ),
    }


# ---------------------------------------------------------------------------
# Model fit scoring (used by apps/sandbox/model_management.py)
# ---------------------------------------------------------------------------

def get_gpu_memory_snapshot():
    if shutil.which("nvidia-smi") is None:
        return None, None, None, "nvidia-smi not found"
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            return None, None, None, msg or f"nvidia-smi exited with code {proc.returncode}"
        first = (proc.stdout or "").strip().splitlines()[0]
        used_mb_str, total_mb_str = [x.strip() for x in first.split(",")]
        used_gb = int(used_mb_str) / 1024.0
        total_gb = int(total_mb_str) / 1024.0
        return used_gb, total_gb - used_gb, total_gb, ""
    except Exception as exc:
        return None, None, None, str(exc)


def artifact_family_from_text(*parts, backend=""):
    if (backend or "").lower().strip() in {"llama_cpp", "llama-cpp", "llamacpp"}:
        return "gguf"
    text = " ".join(p for p in parts if p).lower()
    if "gguf" in text or "q3_k" in text or "q4_k" in text or "q5_k" in text:
        return "gguf"
    if "awq" in text:
        return "awq"
    if "gptq" in text:
        return "gptq"
    if "exl2" in text:
        return "exl2"
    return "unknown"


def extract_model_size_b(text):
    text = (text or "").lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def estimate_model_risk(artifact_family, model_size_b, total_vram_gb,
                        free_vram_gb, backend="", n_gpu_layers=0):
    backend_norm = (backend or "").lower().strip()
    size = model_size_b or 0.0
    if backend_norm in {"anthropic", "claude", "openai", "openai-compatible", "openai_compatible"}:
        return {"fit": "remote", "runtime": "remote",
                "rationale": "Remote/cloud runtime; local GPU memory is not the limiting factor."}
    if artifact_family == "gguf":
        if n_gpu_layers > 0:
            return {"fit": "good", "runtime": "ollama",
                    "rationale": f"Split-offload configured ({n_gpu_layers} GPU layers); fits on this system."}
        return {"fit": "good", "runtime": "ollama",
                "rationale": "GGUF models are a better fit for Ollama/llama.cpp-style runtimes."}
    if total_vram_gb is None:
        return {"fit": "unknown",
                "runtime": "vllm" if artifact_family in {"awq", "gptq", "exl2"} else "ollama",
                "rationale": "GPU capacity could not be detected."}
    if artifact_family in {"awq", "gptq", "exl2", "unknown"}:
        if size <= 10:
            fit, runtime, rationale = "good", "vllm", "Small enough to fit comfortably on this GPU class."
        elif size <= 16:
            fit, runtime, rationale = "good", "vllm", "Strong fit for vLLM on a 24 GB-class GPU."
        elif size <= 24:
            fit, runtime, rationale = "borderline", "vllm", "May require conservative context length and memory settings."
        elif size <= 32:
            fit, runtime, rationale = "better_ollama", "ollama", "Large for vLLM on this GPU; likely a better candidate for Ollama."
        else:
            fit, runtime, rationale = "likely_oom", "ollama", "This model is likely too large for reliable local use on the detected system."
    else:
        fit, runtime, rationale = "unknown", "ollama", "Runtime fit could not be determined confidently."
    if free_vram_gb is not None and fit in {"good", "borderline"}:
        if free_vram_gb < 8:
            fit = "blocked_now"
            rationale = "Good candidate for this machine, but current GPU memory is already occupied."
    if artifact_family == "gptq" and (model_size_b or 0) >= 30 and fit in {"good", "borderline"}:
        fit, runtime, rationale = "better_ollama", "ollama", "Large GPTQ models are often a better fit for Ollama on 24 GB-class GPUs."
    return {"fit": fit, "runtime": runtime, "rationale": rationale}


def score_model_fit(model_text, hf_repo_id="", local_model_dir="",
                    backend="", n_gpu_layers=0):
    _, free_vram_gb, total_vram_gb, _ = get_gpu_memory_snapshot()
    family = artifact_family_from_text(model_text, hf_repo_id, local_model_dir, backend=backend)
    size_b = extract_model_size_b(" ".join([model_text, hf_repo_id, local_model_dir]))
    result = estimate_model_risk(family, size_b, total_vram_gb, free_vram_gb,
                                 backend=backend, n_gpu_layers=n_gpu_layers)
    result["artifact_family"] = family
    result["model_size_b"] = f"{size_b}" if size_b is not None else ""
    result["free_vram_gb"] = f"{free_vram_gb:.1f}" if free_vram_gb is not None else ""
    result["total_vram_gb"] = f"{total_vram_gb:.1f}" if total_vram_gb is not None else ""
    return result
