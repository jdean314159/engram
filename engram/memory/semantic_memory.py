"""
Semantic Memory Implementation

Graph database for structured knowledge using Kuzu.
Supports complex relationships, pattern matching, and traversal queries.

Key features:
- Property graph model (nodes + relationships with properties)
- Cypher query language
- 5-50ms query latency
- Project-specific schemas
- Multi-hop traversal

Author: Jeffrey Dean
"""

import kuzu
import json
import time
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from .semantic_search import SemanticSearchMixin
from .semantic_schema import init_project_schema
# ProjectType, Node, Relationship live in types.py (no kuzu dependency)
# so they can be imported in base environments without kuzu installed.
from .types import Node, ProjectType, Relationship

logger = logging.getLogger(__name__)


class SemanticMemory(SemanticSearchMixin):
    """
    Kuzu graph database for semantic knowledge.
    
    Stores structured facts, preferences, and relationships.
    Supports Cypher queries for pattern matching and traversal.
    
    Usage:
        memory = SemanticMemory(
            db_path="./data/semantic",
            project_type=ProjectType.PROGRAMMING_ASSISTANT
        )
        
        # Add nodes
        memory.add_node("Concept", "async_python", {
            "name": "Async Python",
            "difficulty": "intermediate",
            "documentation": "https://..."
        })
        
        # Add relationships
        memory.add_relationship(
            "Concept", "async_python",
            "Concept", "event_loops",
            "REQUIRES"
        )
        
        # Query
        results = memory.query('''
            MATCH (c:Concept)-[:REQUIRES*1..3]->(prereq)
            WHERE c.name = 'Async Python'
            RETURN prereq.name, prereq.difficulty
        ''')
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        project_type: Optional[ProjectType] = None,
        buffer_pool_size: int = 256 * 1024 * 1024,  # 256MB
    ):
        """
        Initialize semantic memory.
        
        Args:
            db_path: Path to Kuzu database directory. None = in-memory (testing)
            project_type: Project type for schema initialization
            buffer_pool_size: Memory buffer for Kuzu (bytes)
        """
        self.db_path = db_path
        self.project_type = project_type

        # Initialize Kuzu database.
        # Newer Kuzu versions (0.6+) default to mmap-ing very large virtual
        # address regions on 64-bit Linux.  We pass an explicit buffer_pool_size
        # and fall back to progressively smaller values if the OS cannot
        # satisfy the mmap request (seen as "Buffer manager exception: Mmap
        # for size ... failed").
        _pool_sizes = [buffer_pool_size, 64 * 1024 * 1024, 16 * 1024 * 1024]
        _last_err = None
        if db_path:
            db_path = Path(db_path)
            for _pool in _pool_sizes:
                try:
                    self.db = kuzu.Database(str(db_path), buffer_pool_size=_pool)
                    break
                except Exception as e:
                    _last_err = e
                    logger.debug("kuzu.Database(pool=%d) failed: %s", _pool, e)
            else:
                raise RuntimeError(
                    f"Kuzu failed to open database at {db_path} "
                    f"with all buffer_pool_size fallbacks: {_last_err}"
                )
        else:
            for _pool in _pool_sizes:
                try:
                    self.db = kuzu.Database(buffer_pool_size=_pool)
                    break
                except Exception as e:
                    _last_err = e
            else:
                raise RuntimeError(
                    f"Kuzu in-memory database failed: {_last_err}"
                )
        
        self.conn = kuzu.Connection(self.db)
        self._lock = threading.RLock()
        
        # Track created tables
        self.node_tables = set()
        self.rel_tables = set()
        
        # Initialize core schema
        self._init_core_schema()
        
        # Initialize project-specific schema if provided
        if project_type:
            self._init_project_schema(project_type)
    
    def _init_core_schema(self):
        """Initialize core schema used by all projects."""
        
        # User node
        self._create_node_table_safe("User", [
            ("id", "STRING"),
            ("name", "STRING"),
            ("created", "DOUBLE"),
            ("metadata", "STRING"),  # JSON
        ], "id")
        
        # Fact node (general knowledge)
        self._create_node_table_safe("Fact", [
            ("id", "STRING"),
            ("content", "STRING"),
            ("timestamp", "DOUBLE"),
            ("confidence", "DOUBLE"),  # 0.0-1.0
            ("source", "STRING"),
            ("metadata", "STRING"),  # JSON
        ], "id")
        
        # Preference node
        self._create_node_table_safe("Preference", [
            ("id", "STRING"),
            ("category", "STRING"),
            ("value", "STRING"),
            ("strength", "DOUBLE"),  # 0.0-1.0
            ("timestamp", "DOUBLE"),
        ], "id")

        # Generic event node used by the memory runtime itself.
        # This intentionally carries fields needed by both generic assistant
        # memory and the file-organizer overlay so retrieval does not depend on
        # project-specific table variants.
        self._create_node_table_safe("Event", [
            ("id", "STRING"),
            ("summary", "STRING"),
            ("detail", "STRING"),
            ("timestamp", "DOUBLE"),
            ("importance", "DOUBLE"),
            ("source", "STRING"),
            ("metadata", "STRING"),
            ("name", "STRING"),
            ("start_time", "DOUBLE"),
            ("end_time", "DOUBLE"),
            ("location_id", "STRING"),
            ("description", "STRING"),
        ], "id")
        
        # Core relationships
        self._create_rel_table_safe("HAS_FACT", "User", "Fact", [
            ("created", "DOUBLE"),
        ])
        
        self._create_rel_table_safe("HAS_PREFERENCE", "User", "Preference", [
            ("created", "DOUBLE"),
        ])
        
        self._create_rel_table_safe("RELATES_TO", "Fact", "Fact", [
            ("relation_type", "STRING"),
            ("strength", "DOUBLE"),
        ])

        self._create_rel_table_safe("HAS_EVENT", "User", "Event", [
            ("created", "DOUBLE"),
        ])

        self._create_rel_table_safe("DERIVED_FROM", "Event", "Fact", [
            ("created", "DOUBLE"),
        ])

        # --- Typed relation edges (populated by TypedRelationExtractor) ---
        # These connect User context nodes to Entity/Fact/Preference nodes
        # extracted from conversation text. Unlike CO_OCCURS (entity↔entity
        # statistical), these carry semantic meaning usable at query time.

        # User PREFERS something (e.g. "I prefer pytest over unittest")
        self._create_rel_table_safe("PREFERS", "Preference", "Entity", [
            ("confidence",   "DOUBLE"),
            ("source_text",  "STRING"),
            ("timestamp",    "DOUBLE"),
        ])

        # User USES a tool/technology (e.g. "I use PyCharm")
        self._create_rel_table_safe("USES", "Fact", "Entity", [
            ("confidence",   "DOUBLE"),
            ("source_text",  "STRING"),
            ("timestamp",    "DOUBLE"),
        ])

        # Fact KNOWS_ABOUT an entity (e.g. "User works at Anthropic")
        self._create_rel_table_safe("KNOWS_ABOUT", "Fact", "Entity", [
            ("confidence",   "DOUBLE"),
            ("source_text",  "STRING"),
            ("timestamp",    "DOUBLE"),
        ])
    
    def _init_project_schema(self, project_type: "ProjectType"):
        """Initialize project-specific schema (delegates to semantic_schema)."""
        init_project_schema(self, project_type)

    def _create_node_table_safe(
        self,
        table_name: str,
        properties: List[Tuple[str, str]],
        primary_key: str
    ):
        """Create node table if it doesn't exist."""
        if table_name in self.node_tables:
            return
        
        props_sql = ", ".join([f"{name} {type_}" for name, type_ in properties])
        sql = f"CREATE NODE TABLE IF NOT EXISTS {table_name}({props_sql}, PRIMARY KEY({primary_key}))"
        
        try:
            self.conn.execute(sql)
            self.node_tables.add(table_name)
        except Exception as e:
            logger.debug("Node table %s: %s", table_name, e)
            self.node_tables.add(table_name)  # Assume exists
    
    def _create_rel_table_safe(
        self,
        rel_name: str,
        from_table: str,
        to_table: str,
        properties: List[Tuple[str, str]] = None
    ):
        """Create relationship table if it doesn't exist.

        Does NOT cache failures caused by missing node tables — those can be
        retried once the node tables exist.  Only caches success or definitive
        "already exists" responses so GraphExtractor can force a retry.
        """
        rel_key = f"{rel_name}_{from_table}_{to_table}"
        if rel_key in self.rel_tables:
            return

        props_sql = ""
        if properties:
            props_sql = ", " + ", ".join([f"{name} {type_}" for name, type_ in properties])

        sql = f"CREATE REL TABLE IF NOT EXISTS {rel_name}(FROM {from_table} TO {to_table}{props_sql})"

        try:
            self.conn.execute(sql)
            self.rel_tables.add(rel_key)
        except Exception as e:
            err_str = str(e).lower()
            if "already exist" in err_str or "already defined" in err_str:
                # Table already exists — safe to cache
                self.rel_tables.add(rel_key)
            else:
                # Real failure (e.g. node table doesn't exist yet) — do NOT cache.
                # GraphExtractor._ensure_schema will retry after creating node tables.
                logger.debug("Rel table %s creation deferred: %s", rel_key, e)
    
    def add_node(
        self,
        table: str,
        node_id: str,
        properties: Dict[str, Any]
    ) -> Node:
        """
        Add node to graph.
        
        Args:
            table: Node table name
            node_id: Unique node ID
            properties: Node properties dict
            
        Returns:
            Node object
        """
        with self._lock:
            # Build property map
            props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
            sql = f"CREATE (n:{table} {{id: $id, {props_str}}})"
            
            params = {"id": node_id}
            params.update(properties)
            
            try:
                self.conn.execute(sql, params)
            except Exception as e:
                # Node might already exist, try update
                set_str = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
                update_sql = f"MATCH (n:{table}) WHERE n.id = $id SET {set_str}"
                self.conn.execute(update_sql, params)
        
        return Node(table=table, id=node_id, properties=properties)
    
    def add_relationship(
        self,
        from_table: str,
        from_id: str,
        to_table: str,
        to_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Relationship:
        """
        Add relationship between nodes.
        
        Args:
            from_table: Source node table
            from_id: Source node ID
            to_table: Target node table
            to_id: Target node ID
            rel_type: Relationship type
            properties: Optional relationship properties
            
        Returns:
            Relationship object
        """
        props = properties or {}
        
        with self._lock:
            if props:
                prop_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
                sql = f"""
                    MATCH (a:{from_table}), (b:{to_table})
                    WHERE a.id = $from_id AND b.id = $to_id
                    CREATE (a)-[:{rel_type} {{{prop_str}}}]->(b)
                """
                params = {"from_id": from_id, "to_id": to_id}
                params.update(props)
            else:
                sql = f"""
                    MATCH (a:{from_table}), (b:{to_table})
                    WHERE a.id = $from_id AND b.id = $to_id
                    CREATE (a)-[:{rel_type}]->(b)
                """
                params = {"from_id": from_id, "to_id": to_id}
            
            self.conn.execute(sql, params)
        
        return Relationship(
            rel_type=rel_type,
            from_id=from_id,
            to_id=to_id,
            properties=props
        )
    
    def query(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute Cypher query.
        
        Args:
            cypher: Cypher query string
            parameters: Optional query parameters
            
        Returns:
            List of result dicts
        """
        params = parameters or {}
        
        with self._lock:
            result = self.conn.execute(cypher, params)
            
            # Convert to list of dicts
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append(dict(zip(result.get_column_names(), row)))
        
        return rows
    

    def add_fact(
        self,
        content: str,
        confidence: float = 0.7,
        source: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
        user_id: str = "primary_user",
        fact_id: Optional[str] = None,
    ) -> Node:
        """Add a generic fact and link it to the default user node."""
        if not fact_id:
            import hashlib
            fact_id = "fact_" + hashlib.sha256(content.encode()).hexdigest()[:16]
        self.add_node("User", user_id, {
            "name": user_id,
            "created": time.time(),
            "metadata": json.dumps({}),
        })
        node = self.add_node("Fact", fact_id, {
            "content": content,
            "timestamp": time.time(),
            "confidence": confidence,
            "source": source,
            "metadata": json.dumps(metadata or {}),
        })
        try:
            self.add_relationship("User", user_id, "Fact", fact_id, "HAS_FACT", {"created": time.time()})
        except Exception:
            pass
        return node

    def add_preference(
        self,
        category: str,
        value: str,
        strength: float = 0.7,
        source: str = "conversation",
        user_id: str = "primary_user",
        preference_id: Optional[str] = None,
    ) -> Node:
        """Add a generic preference and link it to the default user node."""
        if not preference_id:
            import hashlib
            preference_id = "pref_" + hashlib.sha256(f"{category}:{value}".encode()).hexdigest()[:16]
        self.add_node("User", user_id, {
            "name": user_id,
            "created": time.time(),
            "metadata": json.dumps({"source": source}),
        })
        node = self.add_node("Preference", preference_id, {
            "category": category,
            "value": value,
            "strength": strength,
            "timestamp": time.time(),
        })
        try:
            self.add_relationship("User", user_id, "Preference", preference_id, "HAS_PREFERENCE", {"created": time.time()})
        except Exception:
            pass
        return node

    def _ensure_default_user(self, user_id: str = "primary_user") -> None:
        """Upsert the User node so relationship targets always exist."""
        self.add_node("User", user_id, {
            "name": user_id,
            "created": time.time(),
            "metadata": json.dumps({}),
        })

    def add_event(
        self,
        summary: str,
        *,
        detail: str = "",
        importance: float = 0.6,
        source: str = "lifecycle",
        metadata: Optional[Dict[str, Any]] = None,
        user_id: str = "default_user",
        event_id: Optional[str] = None,
    ) -> Node:
        """Add a generic event node.

        Events are the generic bridge between raw episodes and durable semantic
        memory. They avoid forcing every project into a domain-specific schema.
        """
        event_id = event_id or f"event_{int(time.time() * 1000000)}"
        self._ensure_default_user(user_id)
        payload = {
            "summary": summary,
            "detail": detail,
            "timestamp": time.time(),
            "importance": importance,
            "source": source,
            "metadata": json.dumps(metadata or {}),
        }
        node = self.add_node("Event", event_id, payload)
        try:
            self.add_relationship("User", user_id, "Event", event_id, "HAS_EVENT", {"created": time.time()})
        except Exception:
            pass
        return node

    def list_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return generic events for retrieval and lifecycle inspection."""
        try:
            return self.query(
                f"MATCH (e:Event) RETURN e.id as id, e.summary as summary, e.detail as detail, e.timestamp as timestamp, e.importance as importance, e.source as source, e.metadata as metadata LIMIT {int(limit)}"
            )
        except Exception:
            return []

    def list_facts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return generic facts for application-side scoring."""
        try:
            return self.query(f"MATCH (f:Fact) RETURN f.id as id, f.content as content, f.timestamp as timestamp, f.confidence as confidence, f.source as source, f.metadata as metadata LIMIT {int(limit)}")
        except Exception:
            return []

    def list_preferences(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return generic preferences for application-side scoring."""
        try:
            return self.query(f"MATCH (p:Preference) RETURN p.id as id, p.category as category, p.value as value, p.strength as strength, p.timestamp as timestamp LIMIT {int(limit)}")
        except Exception:
            return []

    def generic_memory_rows(self, limit_per_type: int = 100) -> List[Dict[str, Any]]:
        """Return generic semantic rows independent of domain-specific schema."""
        rows: List[Dict[str, Any]] = []
        rows.extend({"type": "fact", **row} for row in self.list_facts(limit=limit_per_type))
        rows.extend({"type": "preference", **row} for row in self.list_preferences(limit=limit_per_type))
        rows.extend({"type": "event", **row} for row in self.list_events(limit=limit_per_type))
        return rows


    def get_node(self, table: str, node_id: str) -> Optional[Node]:
        """Retrieve node by ID."""
        sql = f"MATCH (n:{table}) WHERE n.id = $id RETURN n"
        results = self.query(sql, {"id": node_id})
        
        if not results:
            return None
        
        node_data = results[0]["n"]
        return Node(table=table, id=node_id, properties=node_data)
    
    def delete_node(self, table: str, node_id: str):
        """Delete node and its relationships."""
        with self._lock:
            sql = f"MATCH (n:{table}) WHERE n.id = $id DETACH DELETE n"
            self.conn.execute(sql, {"id": node_id})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            "node_tables": list(self.node_tables),
            "rel_tables": list(self.rel_tables),
            "node_counts": {},
            "rel_counts": {},
        }
        
        # Count nodes per table
        for table in self.node_tables:
            try:
                result = self.query(f"MATCH (n:{table}) RETURN count(n) as count")
                stats["node_counts"][table] = result[0]["count"] if result else 0
            except Exception as e:
                logger.debug("Stats query failed for %s: %s", table, e)
                stats["node_counts"][table] = 0
        
        return stats
    
    def close(self):
        """Close the Kuzu database connection to release file handles."""
        try:
            if hasattr(self, "conn") and self.conn is not None:
                del self.conn
                self.conn = None
        except Exception:
            pass
        try:
            if hasattr(self, "db") and self.db is not None:
                del self.db
                self.db = None
        except Exception:
            pass

    def __del__(self):
        """Ensure connection is closed when object is garbage collected."""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
