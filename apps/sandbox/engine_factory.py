"""Factory for creating a configured ProjectMemory from UI selections.

This is the glue between:
  - the UI config (profile/budgets/toggles)
  - Engram's engine router (primary/fallback/cloud)
  - ProjectMemory (prompt building + memory layers)

Keep this file readable; most users will copy/modify it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from engram.engine import create_failover_engine
from engram.project_memory import ProjectMemory, TokenBudget


@dataclass
class RuntimeOverrides:
    """Overrides coming from the UI."""
    total_prompt_tokens: int
    reserve_output_tokens: int
    include_cold_fallback: bool
    store_overflow_summary: bool

    working_tokens: Optional[int] = None
    episodic_tokens: Optional[int] = None
    semantic_tokens: Optional[int] = None
    cold_tokens: Optional[int] = None

    engine_order: Optional[list[str]] = None
    allow_cloud_failover: Optional[bool] = None
    cloud_policy: Optional[str] = None


def create_project_memory(
    *,
    profile: str,
    project_dir: Path,
    overrides: RuntimeOverrides,
) -> ProjectMemory:
    """Create ProjectMemory with the selected engine profile and budgets."""
    engine = create_failover_engine(
        profile,
        override_engines=overrides.engine_order,
        override_allow_cloud_failover=overrides.allow_cloud_failover,
        override_cloud_policy=overrides.cloud_policy,
    )

    # Start from Engram's defaults, then apply UI overrides where provided.
    budget = TokenBudget(
        working=overrides.working_tokens or 800,
        episodic=overrides.episodic_tokens or 1800,
        semantic=overrides.semantic_tokens or 1800,
        cold=overrides.cold_tokens or 600,
        )

    pm = ProjectMemory(
        project_id="default",
        project_type="general_assistant",
        base_dir=project_dir,
        llm_engine=engine,
        token_budget=budget,
        session_id="ui",
    )
    return pm
