"""
Engram Memory — Five-layer memory architecture.

Layer 1: Working Memory   (SQLite, 2-5ms)     — session context
Layer 2: Episodic Memory  (ChromaDB, 100ms)    — cross-session search
Layer 3: Semantic Memory  (Kuzu graph, 5-50ms) — structured knowledge
Layer 4: Cold Storage     (stub)               — compressed archive
Layer 5: Neural Memory    (RTRL, 0.3ms)        — sub-symbolic adaptation

Author: Jeffrey Dean
"""

from .working_memory import WorkingMemory, Message
from .cold_storage import ColdStorage
from .ingestion import IngestionDecision, IngestionPolicy, MemoryIngestor
from .retrieval import RetrievalCandidate, RetrievalPolicy, UnifiedRetriever
from .lifecycle import LifecycleConfig, LifecycleReport, MemoryLifecycleManager

__all__ = [
    'WorkingMemory',
    'Message',
    'ColdStorage',
    'MemoryIngestor',
    'IngestionDecision',
    'IngestionPolicy',
    'UnifiedRetriever',
    'RetrievalCandidate',
    'RetrievalPolicy',
]


# Lazy imports for heavy dependencies
def __getattr__(name):
    if name in ('EpisodicMemory', 'Episode'):
        from .episodic_memory import EpisodicMemory, Episode
        if name == 'EpisodicMemory':
            return EpisodicMemory
        return Episode
    elif name in ('SemanticMemory', 'ProjectType', 'Node', 'Relationship'):
        from .semantic_memory import SemanticMemory, ProjectType, Node, Relationship
        return {'SemanticMemory': SemanticMemory, 'ProjectType': ProjectType,
                'Node': Node, 'Relationship': Relationship}[name]
    elif name in ('ProgrammingAssistantHelpers', 'FileOrganizerHelpers',
                  'LanguageTutorHelpers', 'VoiceInterfaceHelpers'):
        from . import semantic_helpers as sh
        return getattr(sh, name)
    raise AttributeError(f"module 'engram.memory' has no attribute {name!r}")
