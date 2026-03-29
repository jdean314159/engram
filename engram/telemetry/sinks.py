from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .core import TelemetryEvent


class LoggingSink:
    """Emit telemetry events via Python logging."""

    def __init__(self, logger_name: str = "engram.telemetry", level: int = logging.INFO):
        self.logger = logging.getLogger(logger_name)
        self.level = level

    def emit(self, event: TelemetryEvent) -> None:
        # Keep it human-readable but structured.
        self.logger.log(
            self.level,
            "telemetry kind=%s msg=%s fields=%s",
            event.kind,
            event.message,
            event.fields,
        )


class JsonlFileSink:
    """Append telemetry events to a JSONL file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: TelemetryEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.to_json())
            f.write("\n")
