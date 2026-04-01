"""
Hardware and Engine Discovery Utilities

Reusable functions for detecting local hardware capabilities and
checking availability of LLM backends (Ollama, API keys).

These utilities are backend-agnostic and intended for use by any
Engram-based application that needs to select or validate an engine
configuration at startup.

Public API:
    detect_hardware()            -> HardwareProfile
    check_ollama_running()       -> bool
    list_ollama_models()         -> list[OllamaModelInfo]
    pull_ollama_model(...)       -> bool
    check_disk_space(path, gb)   -> tuple[bool, float]
    check_api_key(engine_type)   -> bool
    available_engines()          -> EngineAvailability

Author: Jeffrey Dean
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    """Detected hardware capabilities.

    Applications use this to select an appropriate strategy rather than
    hard-coding GPU assumptions.
    """
    gpu_name:      Optional[str]   # e.g. "NVIDIA GeForce RTX 3090"
    vram_gb:       float           # total VRAM (0.0 if no GPU)
    ram_gb:        float           # total system RAM
    cpu_cores:     int             # logical CPU cores
    has_cuda:      bool            # CUDA available

    # Derived capability flags (set by detect_hardware())
    can_run_3b:    bool = False    # ≥ 3 GB VRAM
    can_run_8b:    bool = False    # ≥ 5 GB VRAM
    can_run_9b:    bool = False    # ≥ 7 GB VRAM
    can_run_27b:   bool = False    # ≥ 20 GB VRAM
    can_run_voice: bool = False    # ≥ 5 GB VRAM (Whisper small fits)

    # Human-readable tier label
    tier: str = "cpu_only"        # cpu_only | low | mid | high | ultra


@dataclass
class OllamaModelInfo:
    """A model currently present in the local Ollama store."""
    name:       str
    size_gb:    float
    model_id:   str = ""


@dataclass
class EngineAvailability:
    """Summary of what backends are available on this machine."""
    ollama_running:     bool
    ollama_models:      List[OllamaModelInfo] = field(default_factory=list)
    has_google_key:     bool = False
    has_anthropic_key:  bool = False
    has_openai_key:     bool = False

    def has_model(self, name: str) -> bool:
        """Return True if *name* is pulled in Ollama (exact or prefix match)."""
        for m in self.ollama_models:
            if m.name == name or m.name.startswith(name):
                return True
        return False


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def detect_hardware() -> HardwareProfile:
    """Detect GPU, VRAM, RAM, and CPU cores.

    Uses PyTorch for GPU detection if available; falls back to nvidia-smi,
    then to CPU-only profile.  Never raises — always returns a valid profile.
    """
    ram_gb    = _detect_ram()
    cpu_cores = os.cpu_count() or 1

    # Try PyTorch first (most accurate)
    try:
        import torch
        if torch.cuda.is_available():
            props   = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            return _build_profile(
                gpu_name=torch.cuda.get_device_name(0),
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                cpu_cores=cpu_cores,
                has_cuda=True,
            )
    except Exception:
        pass

    # Try nvidia-smi as fallback (no PyTorch needed)
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        ).strip().splitlines()[0]
        parts   = [p.strip() for p in out.split(",")]
        vram_mb = float(parts[1])
        return _build_profile(
            gpu_name=parts[0],
            vram_gb=vram_mb / 1024,
            ram_gb=ram_gb,
            cpu_cores=cpu_cores,
            has_cuda=True,
        )
    except Exception:
        pass

    # CPU-only
    return HardwareProfile(
        gpu_name=None, vram_gb=0.0,
        ram_gb=ram_gb, cpu_cores=cpu_cores,
        has_cuda=False, tier="cpu_only",
    )


def _detect_ram() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _build_profile(
    gpu_name: str, vram_gb: float, ram_gb: float,
    cpu_cores: int, has_cuda: bool,
) -> HardwareProfile:
    can_3b    = vram_gb >= 2.5
    can_8b    = vram_gb >= 5.0
    can_9b    = vram_gb >= 7.0
    can_27b   = vram_gb >= 20.0
    can_voice = vram_gb >= 5.0

    if vram_gb >= 20:
        tier = "ultra"
    elif vram_gb >= 10:
        tier = "high"
    elif vram_gb >= 5:
        tier = "mid"
    elif vram_gb >= 2:
        tier = "low"
    else:
        tier = "cpu_only"

    return HardwareProfile(
        gpu_name=gpu_name, vram_gb=vram_gb,
        ram_gb=ram_gb, cpu_cores=cpu_cores,
        has_cuda=has_cuda,
        can_run_3b=can_3b, can_run_8b=can_8b,
        can_run_9b=can_9b, can_run_27b=can_27b,
        can_run_voice=can_voice,
        tier=tier,
    )


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def check_ollama_running(base_url: str = _OLLAMA_BASE) -> bool:
    """Return True if Ollama server is reachable."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3):
            return True
    except Exception:
        return False


def list_ollama_models(base_url: str = _OLLAMA_BASE) -> List[OllamaModelInfo]:
    """Return models currently pulled in Ollama, sorted by size descending."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        models = []
        for m in data.get("models", []):
            size_gb = m.get("size", 0) / (1024 ** 3)
            models.append(OllamaModelInfo(
                name=m.get("name", ""),
                size_gb=round(size_gb, 1),
                model_id=m.get("digest", "")[:12],
            ))
        return sorted(models, key=lambda m: m.size_gb, reverse=True)
    except Exception:
        return []


def pull_ollama_model(
    model_name: str,
    progress_callback: Optional[Callable[[float, float, float], None]] = None,
    base_url: str = _OLLAMA_BASE,
) -> bool:
    """Pull a model from Ollama registry with optional progress reporting.

    Args:
        model_name:        e.g. "qwen3:8b"
        progress_callback: called with (pct_0_to_100, downloaded_gb, total_gb)
                           whenever progress is available.  May be None.
        base_url:          Ollama server URL.

    Returns:
        True on success, False on failure.
    """
    url     = f"{base_url}/api/pull"
    payload = json.dumps({"name": model_name, "stream": True}).encode()
    req     = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                status = event.get("status", "")

                if event.get("error"):
                    return False

                if progress_callback and "completed" in event and "total" in event:
                    total     = event["total"]
                    completed = event["completed"]
                    if total > 0:
                        pct          = completed / total * 100
                        downloaded   = completed / (1024 ** 3)
                        total_gb     = total / (1024 ** 3)
                        progress_callback(pct, downloaded, total_gb)

                if status == "success":
                    return True

        return True   # stream ended without explicit success — assume ok

    except urllib.error.HTTPError as e:
        return False
    except Exception:
        return False


def start_ollama() -> bool:
    """Attempt to start the Ollama server as a background process.

    Returns True if Ollama is reachable within 10 seconds of starting.
    """
    if check_ollama_running():
        return True
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False

    import time
    for _ in range(10):
        time.sleep(1)
        if check_ollama_running():
            return True
    return False


# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------

def check_disk_space(
    path: str = "/",
    required_gb: float = 0.0,
) -> tuple[bool, float]:
    """Check available disk space.

    Returns:
        (has_enough, free_gb)  — has_enough is True when free_gb >= required_gb
    """
    try:
        usage  = shutil.disk_usage(path)
        free   = usage.free / (1024 ** 3)
        return free >= required_gb, round(free, 1)
    except Exception:
        return True, 0.0   # cannot check — assume ok


# ---------------------------------------------------------------------------
# API key checks
# ---------------------------------------------------------------------------

_KEY_ENV_MAP = {
    "gemini":    "GOOGLE_API_KEY",
    "google":    "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "claude":    "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "chatgpt":   "OPENAI_API_KEY",
}


def check_api_key(engine_type: str) -> bool:
    """Return True if the required API key env var is set for engine_type."""
    env_var = _KEY_ENV_MAP.get(engine_type.lower())
    if env_var is None:
        return True   # local engine — no key needed
    return bool(os.getenv(env_var))


def get_api_key_env_name(engine_type: str) -> Optional[str]:
    """Return the env var name for an engine type, or None for local engines."""
    return _KEY_ENV_MAP.get(engine_type.lower())


# ---------------------------------------------------------------------------
# Composite availability check
# ---------------------------------------------------------------------------

def available_engines(base_url: str = _OLLAMA_BASE) -> EngineAvailability:
    """Return a full snapshot of what backends are available right now."""
    running = check_ollama_running(base_url)
    models  = list_ollama_models(base_url) if running else []
    return EngineAvailability(
        ollama_running=running,
        ollama_models=models,
        has_google_key=check_api_key("gemini"),
        has_anthropic_key=check_api_key("anthropic"),
        has_openai_key=check_api_key("openai"),
    )
