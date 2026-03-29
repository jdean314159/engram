from __future__ import annotations

from typing import Any, Dict

from .base import ReasoningStrategy


class DirectAnswerStrategy(ReasoningStrategy):
    """Minimal baseline strategy using ProjectMemory.respond()."""

    name = "direct_answer"

    def run(self, memory, user_message: str, **kwargs) -> Dict[str, Any]:
        result = memory.respond(
            user_message,
            strategy=kwargs.pop("strategy", self.name),
            **kwargs,
        )
        result["strategy"] = self.name
        return result
