"""UI config persistence for Engram.

Goals:
- Keep the *library* (engram/) clean: no global mutable config.
- Keep the UI portable: Linux + Windows + macOS.
- Store user selections (profile, budgets, toggles) in a per-user config file.

We use platformdirs so the location is correct for each OS:
  - Linux:   ~/.config/Engram/app_config.json
  - Windows: %APPDATA%\\Engram\\app_config.json
  - macOS:   ~/Library/Application Support/Engram/app_config.json

This file is deliberately small JSON so users can edit it by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from platformdirs import user_config_dir


def _default_config_path() -> Path:
    """Return the default per-user config path."""
    cfg_dir = Path(user_config_dir(appname="Engram", appauthor=False))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "app_config.json"


@dataclass
class UIConfig:
    """User-adjustable UI settings.

    Keep this stable and additive; prefer defaults when fields are missing.
    """

    profile: str = "default_local"
    allow_cloud_failover: bool = False
    cloud_policy: str = "query_plus_summary"  # none|query_only|query_plus_summary|full_context

    # Prompt / retrieval knobs
    total_prompt_tokens: int = 6000
    reserve_output_tokens: int = 512
    include_cold_fallback: bool = True
    store_overflow_summary: bool = False

    # Optional per-layer budgets (Engram has its own internal defaults; these are overrides)
    working_tokens: Optional[int] = None
    episodic_tokens: Optional[int] = None
    semantic_tokens: Optional[int] = None
    cold_tokens: Optional[int] = None

    # Engine selection (runtime-only override; comma-separated engine names)
    engine_order: Optional[str] = None
    endpoint_only_mode: bool = False

    # UI behavior
    show_debug_panels: bool = True


def load_ui_config(path: Optional[Path] = None) -> UIConfig:
    """Load config from disk (or return defaults if missing)."""
    path = path or _default_config_path()
    if not path.exists():
        return UIConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # If the file is corrupted, do not crash the UI; return defaults.
        return UIConfig()
    # Merge known fields only
    cfg = UIConfig()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def save_ui_config(cfg: UIConfig, path: Optional[Path] = None) -> Path:
    """Persist UI config to disk."""
    path = path or _default_config_path()
    payload: Dict[str, Any] = asdict(cfg)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
