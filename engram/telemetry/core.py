from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Protocol


class TelemetrySink(Protocol):
    def emit(self, event: "TelemetryEvent") -> None: ...


@dataclass
class TelemetryEvent:
    ts: float
    kind: str
    message: str
    fields: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "kind": self.kind,
            "message": self.message,
            "fields": self.fields,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class Telemetry:
    """A small event emitter.

    If sink is None, emits nothing.
    """

    def __init__(self, sink: Optional[TelemetrySink] = None, enabled: bool = False):
        self.sink = sink
        self.enabled = bool(enabled and sink is not None)

    def emit(self, kind: str, message: str, **fields: Any) -> None:
        if not self.enabled or self.sink is None:
            return
        ev = TelemetryEvent(ts=time.time(), kind=kind, message=message, fields=fields)
        try:
            self.sink.emit(ev)
        except Exception:
            # Telemetry must never break the main path.
            return
