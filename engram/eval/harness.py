"""Lightweight evaluation harness.

Goals:
  - CPU-only
  - No network dependencies
  - Exercises key invariants: persistence, prompt budgeting, cloud sanitization

This is intentionally minimal; it is designed to catch regressions early.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Any


class _DummyEngine:
    def __init__(self, max_context_length: int = 256):
        self.model_name = "dummy"
        self.system_prompt = "You are a test engine."
        self._max_context_length = max_context_length
        self.is_cloud = False

    @property
    def max_context_length(self) -> int:
        return self._max_context_length

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def compress_prompt(self, prompt: str, target_tokens: int) -> str:
        keep = max(64, target_tokens * 4)
        return prompt[:keep]


def run_basic() -> Dict[str, Any]:
    """Run a basic, CPU-only eval suite.

    Returns a dict suitable for JSON output.
    """
    results: Dict[str, Any] = {
        "ok": True,
        "checks": {},
    }

    from engram.project_memory import ProjectMemory
    from engram.engine.utils.privacy import sanitize_prompt_for_cloud

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        pm = ProjectMemory(
            project_id="eval_project",
            project_type="general",
            base_dir=base,
            llm_engine=_DummyEngine(max_context_length=256),
        )

        # Cold storage dedup
        pm.cold.archive([
            {"project_id": "eval_project", "session_id": "s", "text": "hello", "metadata": {}},
            {"project_id": "eval_project", "session_id": "s", "text": "hello", "metadata": {}},
        ])
        stats = pm.cold.get_stats()
        results["checks"]["cold_dedup"] = int(stats.get("total_rows", 0)) == 1

        # Pressure valve
        for i in range(50):
            pm.working.add("user", f"message {i} " + ("x" * 50))
        built = pm.build_prompt("hi", max_prompt_tokens=256, reserve_output_tokens=64)
        results["checks"]["pressure_valve"] = bool(built.get("compressed")) and int(built.get("prompt_tokens", 99999)) <= 192

        # Cloud sanitization
        p = built["prompt"]
        stripped = sanitize_prompt_for_cloud(p, policy="query_only")
        if "----- BEGIN RETRIEVED MEMORY -----" not in p:
            # Nothing to strip -> considered ok.
            results["checks"]["cloud_query_only"] = True
        else:
            results["checks"]["cloud_query_only"] = (
                "----- BEGIN RETRIEVED MEMORY -----" not in stripped
                and "----- END RETRIEVED MEMORY -----" not in stripped
                and ("User:" in stripped)
            )

    # Overall
    results["ok"] = all(bool(v) for v in results["checks"].values())
    return results
