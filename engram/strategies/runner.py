from __future__ import annotations

from typing import Any, Dict


class StrategyRunner:
    """Registry and dispatcher for reasoning strategies."""

    def __init__(self):
        self._strategies: dict[str, Any] = {}

    def register(self, strategy) -> None:
        self._strategies[strategy.name] = strategy

    def has_strategy(self, name: str) -> bool:
        return name in self._strategies

    def available_strategies(self) -> list[str]:
        return sorted(self._strategies.keys())

    def get(self, name: str):
        try:
            return self._strategies[name]
        except KeyError as e:
            available = ", ".join(self.available_strategies()) or "<none>"
            raise ValueError(f"Unknown strategy '{name}'. Available: {available}") from e

    def run(self, memory, strategy_name: str, user_message: str, **kwargs) -> Dict[str, Any]:
        strategy = self.get(strategy_name)
        return strategy.run(memory, user_message, **kwargs)
