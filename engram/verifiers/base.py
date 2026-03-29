from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class VerificationResult:
    passed: bool
    score: float = 0.0
    reason: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CandidateVerifier(ABC):
    """Base class for general-purpose candidate verifiers."""

    name: str = "base_verifier"

    @abstractmethod
    def verify(self, candidate: str, **kwargs) -> VerificationResult:
        raise NotImplementedError
