from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class ReasoningStrategy(ABC):
    """Base class for reusable reasoning strategies."""

    name: str = "base"

    @abstractmethod
    def run(self, memory, user_message: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
