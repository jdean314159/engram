"""Device selection utilities.

Goal: choose the best available local accelerator without forcing heavy
dependencies or platform-specific assumptions.
"""

from __future__ import annotations

import os
from typing import Literal, Optional


Device = Literal["cpu", "cuda", "mps"]


def resolve_device(requested: Optional[str] = None) -> Device:
    """Resolve a compute device.

    Args:
        requested:
            - None / "auto": pick best available (cuda → mps → cpu)
            - "cpu" | "cuda" | "mps": request explicitly

    Returns:
        "cpu" | "cuda" | "mps"
    """
    # Allow environment override for easy portability.
    env = os.getenv("ENGRAM_DEVICE") or os.getenv("ENGRAM_EMBED_DEVICE")
    if env and not requested:
        requested = env

    requested = (requested or "auto").lower()
    if requested in {"cpu", "cuda", "mps"}:
        return requested  # type: ignore[return-value]

    # Auto mode.
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        # torch missing or misconfigured → CPU
        pass

    return "cpu"
