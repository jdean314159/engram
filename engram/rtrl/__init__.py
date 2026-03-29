"""
Engram RTRL — Neural memory via subgrouped Real-Time Recurrent Learning.

Core: ModernSubgroupedRTRL, RTRLConfig — the RTRL engine
TITANS: TITANSMemory, TITANSConfig — surprise-gated memory interface
Neural: NeuralMemory, NeuralMemoryConfig — Engram layer 5 wrapper

Author: Jeffrey Dean
"""

from .neural_memory import NeuralMemory, NeuralMemoryConfig, EmbeddingProjector

__all__ = [
    'NeuralMemory',
    'NeuralMemoryConfig',
    'EmbeddingProjector',
]


# Lazy imports for core RTRL classes (rarely needed directly)
def __getattr__(name):
    if name in ('TITANSMemory', 'TITANSConfig', 'ModernSubgroupedRTRL', 'RTRLConfig'):
        from . import core
        return getattr(core, name)
    raise AttributeError(f"module 'engram.rtrl' has no attribute {name!r}")
