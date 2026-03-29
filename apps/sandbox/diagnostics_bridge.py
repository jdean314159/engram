"""Thin wrapper around Engram's doctor/recommend logic for use in the UI.

Why this exists
---------------
The CLI (`engram doctor`, `engram recommend`) is built around argparse and
prints to stdout. For the UI we want *programmatic* access to the same
diagnostics payloads (Python dicts) without shelling out.

Config resolution
-----------------
The CLI defaults to reading ~/.engram/llm_engines.yaml. In a fresh install,
that file may not exist yet. For good UX, the UI will:

1) Prefer the user config at ~/.engram/llm_engines.yaml if present.
2) Otherwise fall back to the package-bundled default config shipped at
   engram/engine/llm_engines.yaml.

This makes the UI work out-of-the-box on Linux/Windows/macOS without any
manual config copying.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import subprocess

from engram.cli import run_doctor, run_recommend

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _probe_nvidia_smi() -> Dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        if not lines:
            return {
                "gpu_detected": False,
                "gpu_name": None,
                "total_vram_gb": None,
                "free_vram_gb": None,
                "error": "nvidia-smi returned no GPU rows",
            }

        first = [part.strip() for part in lines[0].split(",")]
        gpu_name = first[0] if len(first) > 0 else None
        total_mb = _safe_float(first[1]) if len(first) > 1 else None
        free_mb = _safe_float(first[2]) if len(first) > 2 else None

        return {
            "gpu_detected": True,
            "gpu_name": gpu_name,
            "total_vram_gb": (total_mb / 1024.0) if total_mb is not None else None,
            "free_vram_gb": (free_mb / 1024.0) if free_mb is not None else None,
            "error": None,
        }
    except Exception as ex:
        return {
            "gpu_detected": False,
            "gpu_name": None,
            "total_vram_gb": None,
            "free_vram_gb": None,
            "error": str(ex),
        }


def _probe_torch_cuda() -> Dict[str, Any]:
    try:
        import torch

        cuda_built = bool(getattr(torch.version, "cuda", None))
        cuda_available = bool(torch.cuda.is_available())

        device_count = 0
        device_name = None
        init_error = None

        try:
            device_count = int(torch.cuda.device_count())
            if cuda_available and device_count > 0:
                device_name = torch.cuda.get_device_name(0)
        except Exception as ex:
            init_error = str(ex)

        return {
            "torch_version": getattr(torch, "__version__", None),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "torch_cuda_built": cuda_built,
            "torch_cuda_available": cuda_available,
            "torch_device_count": device_count,
            "torch_device_name": device_name,
            "torch_cuda_error": init_error,
        }
    except Exception as ex:
        return {
            "torch_version": None,
            "torch_cuda_version": None,
            "torch_cuda_built": False,
            "torch_cuda_available": False,
            "torch_device_count": 0,
            "torch_device_name": None,
            "torch_cuda_error": str(ex),
        }


def _build_hardware_status() -> Dict[str, Any]:
    smi = _probe_nvidia_smi()
    torch_info = _probe_torch_cuda()

    gpu_detected = bool(smi.get("gpu_detected"))
    torch_cuda_available = bool(torch_info.get("torch_cuda_available"))

    if gpu_detected and torch_cuda_available:
        accelerator_class = "gpu"
    elif gpu_detected and not torch_cuda_available:
        accelerator_class = "gpu_unhealthy"
    else:
        accelerator_class = "cpu"

    return {
        "accelerator_class": accelerator_class,
        "gpu_detected": gpu_detected,
        "gpu_name": smi.get("gpu_name") or torch_info.get("torch_device_name"),
        "total_vram_gb": smi.get("total_vram_gb"),
        "free_vram_gb": smi.get("free_vram_gb"),
        "torch_cuda_available": torch_cuda_available,
        "torch_cuda_built": torch_info.get("torch_cuda_built"),
        "torch_cuda_version": torch_info.get("torch_cuda_version"),
        "torch_version": torch_info.get("torch_version"),
        "torch_device_count": torch_info.get("torch_device_count"),
        "torch_cuda_error": torch_info.get("torch_cuda_error"),
        "nvidia_smi_error": smi.get("error"),
    }


def _augment_payload_with_hardware(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    hw = _build_hardware_status()

    existing_hw = dict(out.get("hardware") or {})
    existing_hw.update({
        "accelerator": hw["accelerator_class"],
        "gpu_detected": hw["gpu_detected"],
        "gpu_name": hw["gpu_name"],
        "vram_gb": hw["total_vram_gb"],
        "free_vram_gb": hw["free_vram_gb"],
        "torch_cuda_available": hw["torch_cuda_available"],
        "torch_cuda_built": hw["torch_cuda_built"],
        "torch_cuda_version": hw["torch_cuda_version"],
        "torch_device_count": hw["torch_device_count"],
        "torch_cuda_error": hw["torch_cuda_error"],
        "nvidia_smi_error": hw["nvidia_smi_error"],
    })
    out["hardware"] = existing_hw

    if hw["accelerator_class"] == "gpu_unhealthy":
        out["hardware_warning"] = (
            "GPU detected, but PyTorch CUDA initialization is currently failing. "
            "Recommendations may be pessimistic until CUDA is healthy."
        )

    return out


def _resolve_llm_engines_config() -> str:
    """Return a path to an llm_engines.yaml that exists.

    Returns a string path because the CLI helpers accept strings.
    """
    user_cfg = Path("~/.engram/llm_engines.yaml").expanduser()
    if user_cfg.exists():
        return str(user_cfg)

    # Fall back to the bundled config inside the installed package.
    import engram.engine  # local import to keep UI imports lightweight

    bundled = Path(engram.engine.__file__).resolve().parent / "llm_engines.yaml"
    if bundled.exists():
        return str(bundled)

    # As a last resort, raise a clear error.
    raise FileNotFoundError(
        "Could not find llm_engines.yaml. Expected either ~/.engram/llm_engines.yaml "
        "or the bundled engram/engine/llm_engines.yaml."
    )


def doctor(profile: str, pull_missing: bool = False) -> Dict[str, Any]:
    """Return the same dict payload as `engram doctor --json`, augmented with shared hardware status."""
    cfg = _resolve_llm_engines_config()
    payload = run_doctor(profile=profile, pull_missing=pull_missing, config=cfg)
    return _augment_payload_with_hardware(payload)


def recommend(profile: str) -> Dict[str, Any]:
    """Return the same dict payload as `engram recommend --json`, augmented with shared hardware status."""
    cfg = _resolve_llm_engines_config()
    payload = run_recommend(profile=profile, config=cfg)
    return _augment_payload_with_hardware(payload)
