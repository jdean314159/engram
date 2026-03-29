"""
Engram Filters — Surprise-gated memory storage.

Perplexity filter: text-level surprise (which TEXT to store)
Neural filter:     behavioral surprise (when to UPDATE WEIGHTS)

Author: Jeffrey Dean
"""

from .surprise_filter import SurpriseFilter, SurpriseMetrics, SurpriseBaseline, FilterStats

__all__ = [
    'SurpriseFilter',
    'SurpriseMetrics',
    'SurpriseBaseline',
    'FilterStats',
]
