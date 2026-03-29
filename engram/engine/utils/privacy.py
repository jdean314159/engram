"""Privacy utilities for cloud failover.

We treat retrieved memory as potentially sensitive. When routing a request to a
cloud engine, we apply a configurable *cloud_policy* to minimize data egress.

The prompt assembled by ProjectMemory.build_prompt() wraps the retrieved memory
portion between these markers:

  ----- BEGIN RETRIEVED MEMORY -----
  ...
  ----- END RETRIEVED MEMORY -----

This module provides a best-effort sanitizer that can:
  - strip the entire memory block (query_only)
  - replace it with a compact heuristic summary (query_plus_summary)
  - leave it intact (full_context)
  - disallow cloud usage (none) (handled by router)
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_BEGIN = "----- BEGIN RETRIEVED MEMORY -----"
_END = "----- END RETRIEVED MEMORY -----"


def _extract_memory_block(prompt: str) -> tuple[str, str, str]:
    """Return (prefix, memory_block, suffix). memory_block includes markers."""
    if _BEGIN not in prompt:
        return prompt, "", ""
    start = prompt.find(_BEGIN)
    end = prompt.find(_END)
    if start < 0:
        return prompt, "", ""
    if end < 0:
        # Truncated prompt: treat everything after BEGIN as memory block.
        return prompt[:start], prompt[start:], ""
    if end < start:
        return prompt, "", ""
    end = end + len(_END)
    return prompt[:start], prompt[start:end], prompt[end:]


def _heuristic_summary(memory_block: str, max_chars: int = 2000) -> str:
    """Cheap summarization without needing another model call."""
    # Strip markers and compress whitespace.
    core = memory_block
    core = core.replace(_BEGIN, "").replace(_END, "")
    core = re.sub(r"\s+", " ", core).strip()
    if len(core) <= max_chars:
        return core
    return core[: max_chars - 3] + "..."


def sanitize_prompt_for_cloud(prompt: str, policy: str) -> str:
    """Apply cloud policy to prompt.

    Policies:
      - query_only: remove retrieved memory block
      - query_plus_summary: replace retrieved memory block with compact summary
      - full_context: no changes
    """
    p = (policy or "").lower().strip()
    if p in ("", "full_context"):
        return prompt

    prefix, mem, suffix = _extract_memory_block(prompt)
    if not mem:
        return prompt

    if p == "query_only":
        logger.info("Cloud policy=query_only: stripping retrieved memory block")
        return (prefix + suffix).strip()

    if p == "query_plus_summary":
        summary = _heuristic_summary(mem)
        logger.info("Cloud policy=query_plus_summary: replacing retrieved memory block with compact summary")
        replacement = "Retrieved memory summary (minimized for cloud).\n" + summary + "\n"
        return (prefix + replacement + suffix).strip()

    # Unknown -> conservative
    logger.warning("Unknown cloud policy '%s'; defaulting to query_only", policy)
    return (prefix + suffix).strip()
