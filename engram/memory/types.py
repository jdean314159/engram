"""
Shared types for Engram memory layers.

These types are intentionally kuzu-free so they can be imported in any
environment, whether or not the semantic (kuzu) optional dependency is
installed. Importing this module never triggers a kuzu import.

Author: Jeffrey Dean
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class ProjectType(Enum):
    """Supported project types with generic + optional domain overlays."""
    GENERAL_ASSISTANT = "general_assistant"
    PROGRAMMING_ASSISTANT = "programming_assistant"
    FILE_ORGANIZER = "file_organizer"
    LANGUAGE_TUTOR = "language_tutor"
    VOICE_INTERFACE = "voice_interface"


@dataclass
class Node:
    """Graph node with properties."""
    table: str
    id: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "id": self.id,
            "properties": self.properties,
        }


@dataclass
class Relationship:
    """Graph relationship/edge."""
    rel_type: str
    from_id: str
    to_id: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rel_type": self.rel_type,
            "from_id": self.from_id,
            "to_id": self.to_id,
            "properties": self.properties,
        }
