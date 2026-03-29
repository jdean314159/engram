"""Engram test harness package.

Import order matters — each test module registers its tests
into RUNNER via the @test_group decorator on import.
"""

from .runner import RUNNER, test_group, SkipTest, require, DEPS

# Register all test modules by importing them
from . import test_working_memory       # Layer 1 — always available
from . import test_cold_storage         # Layer 4 — always available
from . import test_embedding            # Embedding cache/service
from . import test_memory_pipeline      # Ingestion, retrieval, forgetting
from . import test_neural_and_extraction # RTRL neural memory, graph extraction
from . import test_episodic_memory      # Layer 2 — requires chromadb
from . import test_semantic_memory      # Layer 3 — requires kuzu
from . import test_engine               # LLM engines and router
from . import test_integration          # ProjectMemory end-to-end
from . import test_strategies_reporting # Strategies, verifiers, reporting
from . import test_respond             # P0-1: ProjectMemory.respond()
from . import test_forgetting_decoupled  # P1-2: ForgettingPolicy decoupled
from . import test_surprise_filter       # P1-3: SurpriseFilter explicit failures
from . import test_index_text            # P1-4: index_text graceful no-op
from . import test_working_memory_isolation  # P1-5: per-session in-memory isolation
from . import test_experiment_wiring        # P2-7: ExperimentMemory auto-wired
from . import test_formatted_prompt         # P3-10: ContextResult.to_formatted_prompt()
from . import test_typed_relations          # P2-6: typed relation extraction
from . import test_rtrl_signal             # RTRL surprise signal validation
from . import test_performance          # Latency benchmarks

__all__ = ["RUNNER", "test_group", "SkipTest", "require", "DEPS"]
