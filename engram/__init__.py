"""Public API for Engram.

Engram is a local-first memory runtime for LLM applications. The main entry
point for most applications is :class:`ProjectMemory`.

Public surface:
- ProjectMemory: orchestrates the memory layers for one project
- TokenBudget / ContextResult: prompt-assembly helpers
- WorkingMemory / ColdStorage: lightweight explicit memory layers
- Lazy exports for episodic, semantic, filter, and neural components
"""

__version__ = "0.1.19"

from .project_memory import (
    ContextResult,
    ProjectMemory,
    TokenBudget,
    wrap_memory_block,
    assemble_prompt,
    truncate_to_tokens,
)
from .memory.cold_storage import ColdStorage
from .memory.working_memory import Message, WorkingMemory
from .memory.ingestion import IngestionDecision, IngestionPolicy, MemoryIngestor
from .memory.retrieval import RetrievalCandidate, RetrievalPolicy, UnifiedRetriever
from .memory.lifecycle import LifecycleConfig, LifecycleReport, MemoryLifecycleManager
# ProjectType, Node, Relationship: kuzu-free — safe to import in base environments
from .memory.types import Node, ProjectType, Relationship

__all__ = [
    "ProjectMemory",
    "TokenBudget",
    "ContextResult",
    # Prompt assembly helpers (independently testable)
    "wrap_memory_block",
    "assemble_prompt",
    "truncate_to_tokens",
    "WorkingMemory",
    "Message",
    "ColdStorage",
    "MemoryIngestor",
    "IngestionDecision",
    "IngestionPolicy",
    "UnifiedRetriever",
    "RetrievalCandidate",
    "RetrievalPolicy",
    "MemoryLifecycleManager",
    "LifecycleConfig",
    "LifecycleReport",
    # Kuzu-free types — always available without semantic optional dependency
    "ProjectType",
    "Node",
    "Relationship",
    # Graph extraction
    "GraphExtractor",
    "ExtractionConfig",
    "ExtractionStats",
    # Forgetting policy
    "ForgettingPolicy",
    "ForgettingConfig",
    # Embedding cache
    "EmbeddingCache",
    # New architecture components (public for custom-app authors)
    "EmbeddingService",
    "NeuralCoordinator",
    "MemoryContext",
    # Semantic layer contract (for custom backend authors)
    "SemanticLayerProtocol",
    # Fine-tuning export
    "export_to_file",
    "export_stats",
    "ExportConfig",
    "__version__",
]


def __getattr__(name):
    if name in ("EpisodicMemory", "Episode"):
        from .memory.episodic_memory import Episode, EpisodicMemory
        return {"EpisodicMemory": EpisodicMemory, "Episode": Episode}[name]
    if name == "SemanticMemory":
        from .memory.semantic_memory import SemanticMemory
        return SemanticMemory
    if name == "SurpriseFilter":
        from .filters.surprise_filter import SurpriseFilter
        return SurpriseFilter
    if name == "TITANSMemory":
        from .rtrl.core import TITANSMemory
        return TITANSMemory
    if name in ("NeuralMemory", "NeuralMemoryConfig"):
        from .rtrl.neural_memory import NeuralMemory, NeuralMemoryConfig
        return {"NeuralMemory": NeuralMemory, "NeuralMemoryConfig": NeuralMemoryConfig}[name]
    if name in ("GraphExtractor", "ExtractionConfig", "ExtractionStats"):
        from .memory.extraction import ExtractionConfig, ExtractionStats, GraphExtractor
        return {"GraphExtractor": GraphExtractor, "ExtractionConfig": ExtractionConfig, "ExtractionStats": ExtractionStats}[name]
    if name in ("ForgettingPolicy", "ForgettingConfig"):
        from .memory.forgetting import ForgettingConfig, ForgettingPolicy
        return {"ForgettingPolicy": ForgettingPolicy, "ForgettingConfig": ForgettingConfig}[name]
    if name == "EmbeddingCache":
        from .memory.embedding_cache import EmbeddingCache
        return EmbeddingCache
    if name == "EmbeddingService":
        from .memory.embedding_service import EmbeddingService
        return EmbeddingService
    if name == "NeuralCoordinator":
        from .memory.neural_coordinator import NeuralCoordinator
        return NeuralCoordinator
    if name == "MemoryContext":
        from .memory.memory_context import MemoryContext
        return MemoryContext
    if name == "SemanticLayerProtocol":
        from .memory.retrieval import SemanticLayerProtocol
        return SemanticLayerProtocol
    if name in ("export_to_file", "export_stats", "ExportConfig"):
        from .finetune.export import ExportConfig, export_stats, export_to_file
        return {"export_to_file": export_to_file, "export_stats": export_stats, "ExportConfig": ExportConfig}[name]
    raise AttributeError(f"module 'engram' has no attribute {name!r}")
