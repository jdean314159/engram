"""Lightweight telemetry/event sink.

This module provides structured events that can be emitted during runtime
(failover decisions, prompt compression, retrieval choices, etc.).

Design goals:
  - Dependency-light (stdlib only)
  - Opt-in (no event emission overhead unless enabled)
  - Supports console/logging and JSONL file sinks
"""

from .core import Telemetry, TelemetryEvent
from .sinks import LoggingSink, JsonlFileSink

__all__ = ["Telemetry", "TelemetryEvent", "LoggingSink", "JsonlFileSink"]
