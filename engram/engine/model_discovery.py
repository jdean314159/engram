"""Lightweight model discovery helpers for local engine backends.

This module is intentionally dependency-light (std-lib + requests).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional

import requests


@dataclass
class DiscoveredModel:
    id: str
    raw: Dict[str, Any]


@dataclass
class DiscoveryResolution:
    requested_model: str
    resolved_model: str
    source: str
    match_strategy: str
    discovered_ids: List[str]


def list_vllm_models(base_url: str, timeout_s: float = 2.0) -> List[DiscoveredModel]:
    """List models from an OpenAI-compatible server (e.g., vLLM).

    Expects `GET {base_url}/models` where base_url typically ends with /v1.
    """
    url = base_url.rstrip("/") + "/models"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    data = r.json() or {}
    items = data.get("data") or []
    out: List[DiscoveredModel] = []
    for it in items:
        mid = it.get("id")
        if mid:
            out.append(DiscoveredModel(id=str(mid), raw=dict(it)))
    return out


def list_ollama_models(base_url: str, timeout_s: float = 2.0) -> List[DiscoveredModel]:
    """List models from Ollama.

    For Ollama, `base_url` should be the server root (e.g., http://localhost:11434).
    """
    root = base_url.rstrip("/")
    url = root + "/api/tags"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    data = r.json() or {}
    items = data.get("models") or []
    out: List[DiscoveredModel] = []
    for it in items:
        name = it.get("name")
        if name:
            out.append(DiscoveredModel(id=str(name), raw=dict(it)))
    return out


def _normalize_model_id(value: str) -> str:
    value = (value or "").strip().lower()
    value = value.replace("\\", "/")
    return value



def _model_aliases(value: str) -> set[str]:
    raw = _normalize_model_id(value)
    if not raw:
        return set()
    aliases = {raw}
    # HF-style basename (org/model -> model)
    aliases.add(PurePosixPath(raw).name)
    aliases.add(raw.split(":")[-1])
    aliases.add(raw.replace("-awq", ""))
    aliases.add(raw.replace("-instruct", ""))
    aliases.add(raw.replace("_", "-"))
    aliases = {a.strip("/") for a in aliases if a}
    return aliases



def match_discovered_model(configured_model: str, discovered_id: str, strategy: str = "suffix_or_exact") -> bool:
    """Best-effort match between configured repo/model ids and served model ids.

    Strategies:
      - exact: normalized exact equality only
      - basename: HF basename equality only
      - suffix_or_exact: exact, basename, and suffix matching on common aliases
    """
    cfg_aliases = _model_aliases(configured_model)
    disc_aliases = _model_aliases(discovered_id)
    if not cfg_aliases or not disc_aliases:
        return False

    if strategy == "exact":
        return _normalize_model_id(configured_model) == _normalize_model_id(discovered_id)

    if strategy == "basename":
        return PurePosixPath(_normalize_model_id(configured_model)).name == PurePosixPath(_normalize_model_id(discovered_id)).name

    # default: suffix_or_exact
    if cfg_aliases & disc_aliases:
        return True
    for ca in cfg_aliases:
        for da in disc_aliases:
            if ca.endswith(da) or da.endswith(ca):
                return True
    return False



def resolve_vllm_model(
    *,
    base_url: str,
    configured_model: str,
    resolution: str = "prefer_discovered",
    match_strategy: str = "suffix_or_exact",
    timeout_s: float = 2.0,
) -> DiscoveryResolution:
    """Resolve the model id to send to a vLLM/OpenAI-compatible server.

    vLLM often serves a model under a different ID than the original HF repo id.
    This helper queries `/v1/models` and, when appropriate, swaps the configured
    model name for the discovered served name.
    """
    requested = str(configured_model)
    if resolution == "configured_only":
        return DiscoveryResolution(
            requested_model=requested,
            resolved_model=requested,
            source="configured",
            match_strategy=match_strategy,
            discovered_ids=[],
        )

    discovered = list_vllm_models(base_url, timeout_s=timeout_s)
    discovered_ids = [m.id for m in discovered]

    match: Optional[str] = None
    for model in discovered:
        if match_discovered_model(requested, model.id, strategy=match_strategy):
            match = model.id
            break

    if match and resolution in {"prefer_discovered", "discover"}:
        return DiscoveryResolution(
            requested_model=requested,
            resolved_model=match,
            source="discovered",
            match_strategy=match_strategy,
            discovered_ids=discovered_ids,
        )

    return DiscoveryResolution(
        requested_model=requested,
        resolved_model=requested,
        source="configured",
        match_strategy=match_strategy,
        discovered_ids=discovered_ids,
    )
