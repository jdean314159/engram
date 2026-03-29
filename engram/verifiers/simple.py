from __future__ import annotations

from .base import CandidateVerifier, VerificationResult


class PassThroughVerifier(CandidateVerifier):
    """Default verifier that only checks for non-empty output."""

    name = "pass_through"

    def verify(self, candidate: str, **kwargs) -> VerificationResult:
        text = candidate or ""
        passed = bool(text.strip())
        return VerificationResult(
            passed=passed,
            score=float(len(text.strip())),
            reason="non_empty" if passed else "empty_candidate",
            metadata={},
        )


class ExactMatchVerifier(CandidateVerifier):
    """Simple reusable verifier for exact-match tasks and tests."""

    name = "exact_match"

    def __init__(self, expected: str):
        self.expected = expected

    def verify(self, candidate: str, **kwargs) -> VerificationResult:
        text = (candidate or "").strip()
        expected = self.expected.strip()
        passed = text == expected
        return VerificationResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="exact_match" if passed else "not_exact_match",
            metadata={"expected": expected},
        )
