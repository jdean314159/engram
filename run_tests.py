#!/usr/bin/env python3
"""
Engram Test Harness — main entry point.

Usage:
    # Run all tests
    python run_tests.py

    # Run specific groups (comma-separated, case-insensitive substring match)
    python run_tests.py --groups "Working Memory,Cold Storage"

    # Run only tests whose name contains a pattern
    python run_tests.py --filter test_wm_thread

    # Show full tracebacks on failures
    python run_tests.py --verbose

    # Skip slow/integration tests
    python run_tests.py --fast

    # Performance benchmarks only
    python run_tests.py --groups Performance

    # As pytest (install pytest, then):
    pytest tests/harness/ -v

Available groups:
    Working Memory          Layer 1 — SQLite, always runs
    Cold Storage            Layer 4 — SQLite FTS5, always runs
    Embedding Cache         In-memory + optional diskcache
    Embedding Service       Lazy model loader
    Memory Ingestion        MemoryIngestor, IngestionPolicy
    Unified Retrieval       UnifiedRetriever, multi-layer merge
    Forgetting Policy       Episodic → cold lifecycle
    Surprise Filter         TITANS perplexity gating
    Neural Memory           RTRL/TITANS (requires torch)
    Graph Extraction        TF-IDF co-occurrence (requires sklearn)
    Episodic Memory         Layer 2 (requires chromadb, sentence_transformers)
    Semantic Memory         Layer 3 (requires kuzu)
    Engine: Base            LLMEngine interface, LogprobResult
    Engine: Router          Failover, circuit breaker
    Engine: Privacy         Cloud sanitisation
    Engine: Structured Output  JSON parsing
    Engine: Ollama          OllamaEngine (live server optional)
    ProjectMemory: Integration  End-to-end pipeline
    ProjectMemory: Lifecycle    MemoryLifecycleManager
    ProjectMemory: Telemetry    Events and sinks
    Strategies              StrategyRunner, DirectAnswer
    Verifiers               SimpleVerifier
    Reporting               RunReporter
    Performance             Latency benchmarks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Raise the open-file-descriptor limit before loading any test code.
# Kuzu opens multiple file handles per database instance; with many tests
# running sequentially the default limit (1024 on most Linux systems) is
# easily exhausted.  We request 4096 soft / leave hard limit untouched.
try:
    import resource as _resource
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    _target = min(max(_soft, 4096), _hard) if _hard > 0 else max(_soft, 4096)
    if _target > _soft:
        _resource.setrlimit(_resource.RLIMIT_NOFILE, (_target, _hard))
except Exception:
    pass  # Windows or permission denied — silently continue

# Ensure project root is importable
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Engram Test Harness")
    p.add_argument(
        "--groups", "-g",
        help="Comma-separated group names (substring match, case-insensitive). "
             "Default: all groups.",
        default=None,
    )
    p.add_argument(
        "--filter", "-f",
        help="Only run test functions whose name contains this string.",
        default=None,
        dest="filter_pattern",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full tracebacks for failures.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Skip Performance benchmarks and live integration tests.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all registered test groups and exit.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Import harness — this registers all tests
    from tests.harness import RUNNER

    if args.list:
        print("Registered test groups:")
        for grp in sorted(RUNNER._groups.keys()):
            count = len(RUNNER._groups[grp])
            print(f"  {grp:40s}  ({count} tests)")
        return 0

    # Resolve groups
    groups = None
    if args.groups:
        requested = [g.strip() for g in args.groups.split(",")]
        all_groups = list(RUNNER._groups.keys())
        groups = []
        for req in requested:
            matches = [g for g in all_groups if req.lower() in g.lower()]
            if not matches:
                print(f"Warning: no group matching '{req}'. Available: {all_groups}")
            groups.extend(matches)

    if args.fast and groups is None:
        groups = [g for g in RUNNER._groups.keys() if "Performance" not in g]

    return RUNNER.run_all(
        groups=groups,
        filter_pattern=args.filter_pattern,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
