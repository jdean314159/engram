"""Project-type schema definitions for Engram semantic memory.

Each function receives a ``SemanticMemory`` instance and calls its
``_create_node_table_safe`` / ``_create_rel_table_safe`` helpers to
set up the Kuzu graph schema for a specific project type.

The general-assistant schema (User, Fact, Preference, Event) is part of
the core schema in ``SemanticMemory._init_core_schema()`` and is always
created.  The functions here are *additive* — they layer domain-specific
node and relationship types on top of the core schema.

Author: Jeffrey Dean
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .semantic_memory import SemanticMemory, ProjectType

logger = logging.getLogger(__name__)


def init_project_schema(mem: "SemanticMemory", project_type: "ProjectType") -> None:
    """Dispatch to the correct schema initialiser for ``project_type``."""
    from .semantic_memory import ProjectType as PT

    dispatch = {
        PT.PROGRAMMING_ASSISTANT: init_programming_schema,
        PT.FILE_ORGANIZER: init_file_organizer_schema,
        PT.LANGUAGE_TUTOR: init_language_tutor_schema,
        PT.VOICE_INTERFACE: init_voice_interface_schema,
        PT.GENERAL_ASSISTANT: _noop,
    }
    fn = dispatch.get(project_type, _noop)
    if fn is _noop and project_type not in dispatch:
        logger.warning(
            "No domain schema defined for project_type=%r; using general schema only.",
            project_type,
        )
    fn(mem)


def _noop(mem: "SemanticMemory") -> None:
    """No-op for project types that have no additional schema beyond core."""


def init_programming_schema(mem: "SemanticMemory") -> None:
    """Additional Kuzu schema for programming-assistant projects.

    Adds: Concept, CodeSnippet, Bug, APIKnowledge node types and their
    relationships (REQUIRES, DEPENDS_ON, CAUSED_BY, ALSO_CAUSES,
    ALTERNATIVE_TO, IMPLEMENTS, USES_API).
    """
    # Concept node (programming concepts)
    mem._create_node_table_safe("Concept", [
        ("id", "STRING"),
        ("name", "STRING"),
        ("difficulty", "STRING"),   # beginner, intermediate, advanced
        ("category", "STRING"),     # language, framework, pattern, etc.
        ("documentation", "STRING"),
        ("examples", "STRING"),     # JSON array
    ], "id")

    # CodeSnippet node
    mem._create_node_table_safe("CodeSnippet", [
        ("id", "STRING"),
        ("language", "STRING"),
        ("code", "STRING"),
        ("description", "STRING"),
        ("complexity", "INT64"),
        ("tokens", "INT64"),
        ("tags", "STRING"),         # JSON array
        ("created", "DOUBLE"),
    ], "id")

    # Bug node
    mem._create_node_table_safe("Bug", [
        ("id", "STRING"),
        ("error_type", "STRING"),
        ("message", "STRING"),
        ("frequency", "INT64"),
        ("solution", "STRING"),
        ("context", "STRING"),      # JSON
    ], "id")

    # APIKnowledge node
    mem._create_node_table_safe("APIKnowledge", [
        ("id", "STRING"),
        ("name", "STRING"),
        ("version", "STRING"),
        ("endpoint", "STRING"),
        ("documentation", "STRING"),
        ("parameters", "STRING"),   # JSON
    ], "id")

    # Relationships
    mem._create_rel_table_safe("REQUIRES", "Concept", "Concept", [
        ("importance", "STRING"),   # critical, recommended, optional
    ])
    mem._create_rel_table_safe("DEPENDS_ON", "CodeSnippet", "CodeSnippet", [
        ("dependency_type", "STRING"),
        ("strength", "DOUBLE"),
    ])
    mem._create_rel_table_safe("CAUSED_BY", "Bug", "CodeSnippet", [
        ("likelihood", "DOUBLE"),
    ])
    mem._create_rel_table_safe("ALSO_CAUSES", "Bug", "Bug", [
        ("correlation", "DOUBLE"),
    ])
    mem._create_rel_table_safe("ALTERNATIVE_TO", "CodeSnippet", "CodeSnippet", [
        ("performance_diff", "DOUBLE"),
        ("notes", "STRING"),
    ])
    mem._create_rel_table_safe("IMPLEMENTS", "CodeSnippet", "Concept", [])
    mem._create_rel_table_safe("USES_API", "CodeSnippet", "APIKnowledge", [])


def init_file_organizer_schema(mem: "SemanticMemory") -> None:
    """Additional Kuzu schema for file-organizer projects (90K+ photo collections).

    Adds: Photo, Location, Person node types and their relationships
    (TAKEN_AT, CONTAINS, PART_OF, SIMILAR_TO, IN_COUNTRY, IN_CITY).
    """
    mem._create_node_table_safe("Photo", [
        ("id", "STRING"),
        ("path", "STRING"),
        ("filename", "STRING"),
        ("timestamp", "DOUBLE"),
        ("size", "INT64"),
        ("width", "INT64"),
        ("height", "INT64"),
        ("hash", "STRING"),         # For duplicate detection
        ("camera", "STRING"),
        ("metadata", "STRING"),     # JSON (EXIF etc.)
    ], "id")

    mem._create_node_table_safe("Location", [
        ("id", "STRING"),
        ("name", "STRING"),
        ("latitude", "DOUBLE"),
        ("longitude", "DOUBLE"),
        ("city", "STRING"),
        ("country", "STRING"),
    ], "id")

    mem._create_node_table_safe("Person", [
        ("id", "STRING"),
        ("name", "STRING"),
        ("face_encoding", "STRING"),  # JSON array of face encoding
    ], "id")

    mem._create_rel_table_safe("TAKEN_AT", "Photo", "Location", [
        ("confidence", "DOUBLE"),
    ])
    mem._create_rel_table_safe("CONTAINS", "Photo", "Person", [
        ("bounding_box", "STRING"),  # JSON
        ("confidence", "DOUBLE"),
    ])
    mem._create_rel_table_safe("PART_OF", "Photo", "Event", [])
    mem._create_rel_table_safe("SIMILAR_TO", "Photo", "Photo", [
        ("similarity", "DOUBLE"),
        ("algorithm", "STRING"),
    ])
    mem._create_rel_table_safe("IN_COUNTRY", "Location", "Location", [])
    mem._create_rel_table_safe("IN_CITY", "Location", "Location", [])


def init_language_tutor_schema(mem: "SemanticMemory") -> None:
    """Additional Kuzu schema for language-tutor projects.

    Adds: VocabularyWord, GrammarRule, LanguageConcept, Mistake node
    types and their relationships (REQUIRES_CONCEPT, MASTERED,
    CONFUSED_WITH, MADE_MISTAKE, APPLIES_RULE).
    """
    mem._create_node_table_safe("VocabularyWord", [
        ("id", "STRING"),
        ("word", "STRING"),
        ("language", "STRING"),
        ("translation", "STRING"),
        ("difficulty", "STRING"),
        ("part_of_speech", "STRING"),
        ("examples", "STRING"),     # JSON array
    ], "id")

    mem._create_node_table_safe("GrammarRule", [
        ("id", "STRING"),
        ("name", "STRING"),
        ("language", "STRING"),
        ("difficulty", "STRING"),
        ("explanation", "STRING"),
        ("examples", "STRING"),     # JSON
    ], "id")

    mem._create_node_table_safe("LanguageConcept", [
        ("id", "STRING"),
        ("name", "STRING"),
        ("language", "STRING"),
        ("difficulty", "STRING"),
        ("description", "STRING"),
    ], "id")

    mem._create_node_table_safe("Mistake", [
        ("id", "STRING"),
        ("incorrect", "STRING"),
        ("correct", "STRING"),
        ("error_type", "STRING"),
        ("frequency", "INT64"),
        ("last_occurred", "DOUBLE"),
    ], "id")

    mem._create_rel_table_safe("REQUIRES_CONCEPT", "LanguageConcept", "LanguageConcept", [
        ("importance", "STRING"),
    ])
    mem._create_rel_table_safe("MASTERED", "User", "VocabularyWord", [
        ("mastery_level", "DOUBLE"),
        ("last_practiced", "DOUBLE"),
    ])
    mem._create_rel_table_safe("CONFUSED_WITH", "VocabularyWord", "VocabularyWord", [
        ("frequency", "INT64"),
    ])
    mem._create_rel_table_safe("MADE_MISTAKE", "User", "Mistake", [
        ("timestamp", "DOUBLE"),
    ])
    mem._create_rel_table_safe("APPLIES_RULE", "LanguageConcept", "GrammarRule", [])


def init_voice_interface_schema(mem: "SemanticMemory") -> None:
    """Additional Kuzu schema for voice-interface projects.

    Adds: Command, Pattern node types and their relationships
    (FOLLOWS, TRIGGERS, MATCHES_PATTERN).
    """
    mem._create_node_table_safe("Command", [
        ("id", "STRING"),
        ("text", "STRING"),
        ("intent", "STRING"),
        ("entities", "STRING"),     # JSON
        ("frequency", "INT64"),
        ("last_used", "DOUBLE"),
    ], "id")

    mem._create_node_table_safe("Pattern", [
        ("id", "STRING"),
        ("pattern", "STRING"),
        ("intent", "STRING"),
        ("confidence", "DOUBLE"),
    ], "id")

    mem._create_rel_table_safe("FOLLOWS", "Command", "Command", [
        ("frequency", "INT64"),
        ("avg_delay", "DOUBLE"),
    ])
    mem._create_rel_table_safe("TRIGGERS", "Command", "Command", [
        ("condition", "STRING"),
    ])
    mem._create_rel_table_safe("MATCHES_PATTERN", "Command", "Pattern", [
        ("confidence", "DOUBLE"),
    ])
