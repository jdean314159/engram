"""
Configuration Loader for LLM Engines

Loads engines from YAML config files.

Author: Jeffrey Dean
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, List

import yaml

from .base import LLMEngine, CompressionStrategy
from .router import FailoverEngine, FailoverPolicy
from .model_discovery import resolve_vllm_model
from ..telemetry import Telemetry, LoggingSink, JsonlFileSink

try:
    from .openai_cloud_engine import OpenAICloudEngine
except Exception:  # pragma: no cover
    OpenAICloudEngine = None

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "llm_engines.yaml"


def _user_config_path() -> Path:
    """Return the default user config path (~/.engram/llm_engines.yaml)."""
    return Path("~/.engram/llm_engines.yaml").expanduser()


def ensure_user_config_exists() -> Path:
    """Ensure the user config exists by copying the packaged default if missing.

    This makes a fresh install usable without requiring the user to discover where the
    default YAML lives inside site-packages.
    """
    user_path = _user_config_path()
    if user_path.exists():
        return user_path
    user_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil as _shutil
    _shutil.copyfile(_DEFAULT_CONFIG, user_path)
    return user_path


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load YAML config file.

    Resolution order:
      1) explicit `config_path` (if provided)
      2) user config at ~/.engram/llm_engines.yaml (auto-created from packaged default)
      3) packaged default config (engram/engine/llm_engines.yaml)
    """
    if config_path:
        path = Path(config_path).expanduser()
    else:
        path = ensure_user_config_exists()

    if not path.exists():
        path = _DEFAULT_CONFIG

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def create_engine(engine_name: str, config_path: Optional[str] = None, engine_config_override: Optional[Dict] = None) -> LLMEngine:
    """Create an LLM engine from YAML config."""
    config = load_config(config_path)
    engines_cfg = config.get("engines") or {}
    if engine_name not in engines_cfg:
        available = ", ".join(engines_cfg.keys()) or "(none)"
        raise ValueError(f"Engine '{engine_name}' not found. Available: {available}")

    engine_config = dict(engines_cfg[engine_name] or {})
    if engine_config_override:
        engine_config.update(engine_config_override)
    engine_type = engine_config.get("type")
    if not engine_type:
        raise ValueError(f"Engine '{engine_name}' must set type")

    strategy_str = engine_config.get("compression_strategy", "compress")
    compression_strategy = CompressionStrategy(strategy_str)

    common_params = {
        "model_name": engine_config["model"],
        "system_prompt": engine_config.get("system_prompt"),
        "compression_strategy": compression_strategy,
        "max_retries": int(engine_config.get("max_retries", 1)),
        "timeout": int(engine_config.get("timeout", 120)),
        "max_context": int(engine_config.get("max_context", 8192)),
    }

    if engine_type == "vllm":
        from .vllm_engine import VLLMEngine
        configured_model = str(engine_config["model"])
        resolution = str(engine_config.get("model_resolution", "prefer_discovered"))
        match_strategy = str(engine_config.get("model_match", "suffix_or_exact"))
        discovery_timeout_s = float(engine_config.get("discovery_timeout_s", 2.0))
        discovery_resolution = None
        resolved_model = configured_model
        try:
            discovery_resolution = resolve_vllm_model(
                base_url=engine_config["base_url"],
                configured_model=configured_model,
                resolution=resolution,
                match_strategy=match_strategy,
                timeout_s=discovery_timeout_s,
            )
            resolved_model = discovery_resolution.resolved_model
        except Exception as ex:
            logger.info("vLLM model discovery skipped for %s: %s", engine_name, ex)

        eng = VLLMEngine(
            base_url=engine_config["base_url"],
            api_key=engine_config.get("api_key", "dummy"),
            model_name=resolved_model,
            configured_model_name=configured_model,
            discovery_resolution=(discovery_resolution.__dict__ if discovery_resolution else None),
            reasoning_visibility=str(engine_config.get("reasoning_visibility", "auto")),
            **{k: v for k, v in common_params.items() if k != "model_name"},
        )        
        return eng

    if engine_type == "anthropic":
        from .claude_engine import ClaudeEngine
        return ClaudeEngine(
            api_key=engine_config.get("api_key"),  # None = env var
            **common_params,
        )

    if engine_type == "ollama":
        from .ollama_engine import OllamaEngine
        eng = OllamaEngine(
            base_url=engine_config.get("base_url", "http://localhost:11434/v1"),
            api_key=engine_config.get("api_key", "ollama"),
            num_gpu=engine_config.get("num_gpu"),
            **common_params,
        )
        if bool(engine_config.get("auto_pull", False)):
            eng.ensure_model_pulled()
        return eng

    if engine_type in ("llama_cpp", "llama-cpp", "llamacpp"):
        from .llama_cpp_engine import LlamaCppEngine
        return LlamaCppEngine(
            base_url=engine_config.get("base_url", "http://127.0.0.1:8080/v1"),
            api_key=engine_config.get("api_key", "dummy"),
            gguf_path=engine_config.get("gguf_path"),
            n_gpu_layers=int(engine_config.get("n_gpu_layers", 0)),
            reasoning_visibility=str(engine_config.get("reasoning_visibility", "auto")),
            **common_params,
        )

    if engine_type in ("openai", "openai_compat", "cloud"):
        if OpenAICloudEngine is None:
            raise ImportError(
                "OpenAICloudEngine requires optional dependencies. Install with: pip install engram[engines]"
            )
        eng = OpenAICloudEngine(
            model_name=engine_config["model"],
            api_key=engine_config.get("api_key"),
            api_key_env=engine_config.get("api_key_env", "OPENAI_API_KEY"),
            base_url=engine_config.get("base_url", "https://api.openai.com/v1"),
            system_prompt=engine_config.get("system_prompt"),
            compression_strategy=compression_strategy,
            max_retries=int(engine_config.get("max_retries", 1)),
            timeout=int(engine_config.get("timeout", 120)),
            max_context=int(engine_config.get("max_context", 8192)),
        )
        return eng

    raise ValueError(f"Unknown engine type: {engine_type}")


def create_failover_engine(
    profile_name: str = "default_local",
    config_path: Optional[str] = None,
    override_engines: Optional[list[str]] = None,
    override_allow_cloud_failover: Optional[bool] = None,
    override_cloud_policy: Optional[str] = None,
) -> LLMEngine:
    """Create a FailoverEngine profile from YAML.

    Example YAML:

    profiles:
      default_local:
        engines: [qwen3_32b, qwen3_8b]
        ensure_models: true
        allow_cloud_failover: false
        max_attempts: 4
    """
    config = load_config(config_path)
    profiles = config.get("profiles") or {}
    if profile_name not in profiles:
        available = ", ".join(profiles.keys()) or "(none)"
        raise ValueError(f"Profile '{profile_name}' not found. Available: {available}")

    profile = profiles[profile_name] or {}
    # Telemetry (optional): can be configured per-profile, or enabled globally via env vars.
    #
    # Env vars:
    #   ENGRAM_TELEMETRY=1
    #   ENGRAM_TELEMETRY_SINK=log|jsonl
    #   ENGRAM_TELEMETRY_PATH=~/.engram/telemetry.jsonl
    telemetry_cfg = profile.get("telemetry") or {}
    telemetry: Optional[Telemetry] = None

    env_enabled = str(os.getenv("ENGRAM_TELEMETRY", "")).strip().lower() in {"1", "true", "yes", "on"}
    if env_enabled:
        sink = str(os.getenv("ENGRAM_TELEMETRY_SINK", "log")).lower().strip() or "log"
        if sink == "jsonl":
            raw = os.getenv("ENGRAM_TELEMETRY_PATH", str(Path.home() / ".engram" / "telemetry.jsonl"))
            path = Path(str(raw)).expanduser()
            telemetry = Telemetry(sink=JsonlFileSink(path), enabled=True)
        else:
            telemetry = Telemetry(sink=LoggingSink(), enabled=True)
    elif bool(telemetry_cfg.get("enabled", False)):
        sink = str(telemetry_cfg.get("sink", "log")).lower()
        if sink == "jsonl":
            path = Path(str(telemetry_cfg.get("jsonl_path", Path.home() / ".engram" / "telemetry.jsonl"))).expanduser()
            telemetry = Telemetry(sink=JsonlFileSink(path), enabled=True)
        else:
            telemetry = Telemetry(sink=LoggingSink(), enabled=True)

    engine_names: List[str] = list(override_engines or profile.get("engines") or [])
    if not engine_names:
        raise ValueError(f"Profile '{profile_name}' must list engines")

    engines: List[LLMEngine] = []
    for name in engine_names:
        engines.append(create_engine(name, config_path=config_path))

    # Optionally ensure Ollama models for the whole profile (useful if auto_pull is not set per engine)
    if bool(profile.get("ensure_models", False)):
        for eng in engines:
            if eng.__class__.__name__ == "OllamaEngine":
                eng.ensure_model_pulled()

    policy = FailoverPolicy(
        max_attempts=int(profile.get("max_attempts", 4)),
        allow_cloud_failover=bool(profile.get("allow_cloud_failover", False)) if override_allow_cloud_failover is None else bool(override_allow_cloud_failover),
        cloud_policy=str(profile.get("cloud_policy", "query_plus_summary")) if override_cloud_policy is None else str(override_cloud_policy),
        circuit_breaker_failures=int(profile.get("circuit_breaker_failures", 3)),
        circuit_breaker_cooldown_s=float(profile.get("circuit_breaker_cooldown_s", 30.0)),
    )
    return FailoverEngine(engines=engines, policy=policy, name=profile_name, telemetry=telemetry)
