"""
Engram Test Harness - Runner Core

Lightweight runner: no pytest required, but tests are also importable
as pytest-compatible functions (return None on pass, raise on fail).

Result types: PASS / FAIL / SKIP / ERROR
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

# ── ANSI colours ─────────────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

PASS_CLR  = lambda s: _c("32;1", s)
FAIL_CLR  = lambda s: _c("31;1", s)
SKIP_CLR  = lambda s: _c("33;1", s)
ERR_CLR   = lambda s: _c("35;1", s)
HDR_CLR   = lambda s: _c("36;1", s)


# ── Result ────────────────────────────────────────────────────────────────────

class Status(Enum):
    PASS  = "PASS"
    FAIL  = "FAIL"
    SKIP  = "SKIP"
    ERROR = "ERROR"


@dataclass
class TestResult:
    name: str
    group: str
    status: Status
    duration_ms: float = 0.0
    message: str = ""
    tb: str = ""


# ── Skip exception ────────────────────────────────────────────────────────────

class SkipTest(Exception):
    """Raised by require() and test functions to signal a skipped test.

    Inherits from unittest.SkipTest so pytest recognises it natively
    and marks the test as skipped rather than failed.
    """

try:
    import unittest as _unittest
    # Re-base on unittest.SkipTest for pytest compatibility
    class SkipTest(_unittest.SkipTest):  # noqa: F811
        pass
except Exception:
    pass  # Fall back to plain Exception — harness runner handles it directly


# ── Dependency probe ──────────────────────────────────────────────────────────

def _has(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


DEPS: Dict[str, bool] = {
    "chromadb":             _has("chromadb"),
    "sentence_transformers": _has("sentence_transformers"),
    "diskcache":            _has("diskcache"),
    "kuzu":                 _has("kuzu"),
    "torch":                _has("torch"),
    "numpy":                _has("numpy"),
    "sklearn":              _has("sklearn"),
    "openai":               _has("openai"),
    "anthropic":            _has("anthropic"),
    "tiktoken":             _has("tiktoken"),
    "pydantic":             _has("pydantic"),
}


def require(*pkgs: str) -> None:
    """Raise SkipTest if any pkg is unavailable."""
    missing = [p for p in pkgs if not DEPS.get(p, _has(p))]
    if missing:
        raise SkipTest(f"Missing optional deps: {', '.join(missing)}")


# ── Runner ────────────────────────────────────────────────────────────────────

@dataclass
class HarnessRunner:
    """Collects and executes test groups."""

    _groups: Dict[str, List[Callable]] = field(default_factory=dict)
    results: List[TestResult] = field(default_factory=list)

    def register(self, group: str, fn: Callable) -> None:
        self._groups.setdefault(group, []).append(fn)

    def run_all(
        self,
        groups: Optional[List[str]] = None,
        filter_pattern: Optional[str] = None,
        verbose: bool = False,
    ) -> int:
        """Run tests; return non-zero exit code if any FAIL/ERROR."""
        target_groups = groups or list(self._groups.keys())

        print(HDR_CLR("\n═══ Engram Test Harness ═══"))
        _print_dep_table()

        for grp in target_groups:
            fns = self._groups.get(grp, [])
            if not fns:
                continue
            print(f"\n{HDR_CLR('─── ' + grp + ' ───')}")

            for fn in fns:
                name = fn.__name__
                if filter_pattern and filter_pattern not in name:
                    continue
                self._run_one(grp, name, fn, verbose)

        return self._print_summary()

    def _run_one(self, group: str, name: str, fn: Callable, verbose: bool) -> None:
        t0 = time.perf_counter()
        try:
            fn()
            ms = (time.perf_counter() - t0) * 1000
            r = TestResult(name, group, Status.PASS, ms)
            label = PASS_CLR("PASS")
        except SkipTest as e:
            ms = (time.perf_counter() - t0) * 1000
            r = TestResult(name, group, Status.SKIP, ms, str(e))
            label = SKIP_CLR("SKIP")
        except AssertionError as e:
            ms = (time.perf_counter() - t0) * 1000
            tb = traceback.format_exc()
            r = TestResult(name, group, Status.FAIL, ms, str(e), tb)
            label = FAIL_CLR("FAIL")
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            tb = traceback.format_exc()
            r = TestResult(name, group, Status.ERROR, ms, str(e), tb)
            label = ERR_CLR("ERRO")

        self.results.append(r)
        duration_str = f"{r.duration_ms:6.1f}ms"
        print(f"  [{label}] {duration_str}  {name}")
        if r.message and (r.status in (Status.FAIL, Status.ERROR, Status.SKIP) or verbose):
            # Print first line of message
            first = r.message.splitlines()[0] if r.message else ""
            if first:
                print(f"           {first}")
        if verbose and r.tb:
            for line in r.tb.splitlines():
                print(f"           {line}")

    def _print_summary(self) -> int:
        counts = {s: 0 for s in Status}
        for r in self.results:
            counts[r.status] += 1

        total = len(self.results)
        print(f"\n{HDR_CLR('═══ Summary ═══')}")
        print(
            f"  Total: {total}  "
            f"{PASS_CLR(str(counts[Status.PASS]) + ' pass')}  "
            f"{FAIL_CLR(str(counts[Status.FAIL]) + ' fail')}  "
            f"{ERR_CLR(str(counts[Status.ERROR]) + ' error')}  "
            f"{SKIP_CLR(str(counts[Status.SKIP]) + ' skip')}"
        )

        failures = [r for r in self.results if r.status in (Status.FAIL, Status.ERROR)]
        if failures:
            print(f"\n{FAIL_CLR('Failed tests:')}")
            for r in failures:
                print(f"  {r.group} :: {r.name}")
                if r.message:
                    print(f"    {r.message.splitlines()[0]}")
            return 1
        return 0


def _print_dep_table() -> None:
    print(HDR_CLR("\nDependency Status:"))
    for pkg, available in DEPS.items():
        state = PASS_CLR("✓ available") if available else SKIP_CLR("✗ missing  ")
        print(f"  {state}  {pkg}")


# ── Global singleton ──────────────────────────────────────────────────────────

RUNNER = HarnessRunner()


def test_group(name: str):
    """Decorator: @test_group('Working Memory')"""
    def decorator(fn: Callable) -> Callable:
        RUNNER.register(name, fn)
        return fn
    return decorator
