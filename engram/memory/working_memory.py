"""
Working Memory Implementation

Session-scoped short-term memory using SQLite.
Provides fast (2-5ms) access to recent conversation context.

Key features:
- Thread-safe SQLite operations
- Token budget management with FIFO eviction
- Keyword search and recent retrieval
- Automatic context window trimming

Used by: Voice Interface (pronoun resolution), all projects (conversation context)

Author: Jeffrey Dean
"""

import sqlite3
import json
import time
import threading
from typing import Callable, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager


@dataclass
class Message:
    """Single message in working memory."""
    id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    role: str = "user"  # 'user', 'assistant', 'system'
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    session_id: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "session_id": self.session_id,
        }
    
    @classmethod
    def from_row(cls, row: Tuple) -> "Message":
        """Create Message from SQLite row."""
        return cls(
            id=row[0],
            timestamp=row[1],
            role=row[2],
            content=row[3],
            metadata=json.loads(row[4]) if row[4] else {},
            token_count=row[5],
            session_id=row[6],
        )


class WorkingMemory:
    """
    SQLite-backed working memory with token budget management.
    
    Thread-safe for concurrent access (critical for voice interface).
    Automatically evicts oldest messages when token budget exceeded.
    
    Usage:
        memory = WorkingMemory(max_tokens=1000)
        memory.add("user", "What's 2+2?")
        memory.add("assistant", "4")
        recent = memory.get_recent(n=5)
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        session_id: str = "default",
        max_tokens: int = 1000,
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        """
        Initialize working memory.
        
        Args:
            db_path: Path to SQLite database. None = in-memory (shared cache for thread safety)
            session_id: Session identifier (isolates different conversations)
            max_tokens: Token budget for this session
            token_counter: Function(text: str) -> int. Defaults to len(text)//4
        """
        # Use a named per-session in-memory database to prevent cross-instance
        # contamination.  The shared-cache variant (file::memory:?cache=shared)
        # puts ALL in-memory WorkingMemory instances in the same process into the
        # same SQLite file.  Session-scoped isolation via SQL WHERE clauses is
        # sufficient for normal use, but concurrent tests (or multiple ProjectMemory
        # instances in one process) accumulate rows across sessions, produce wrong
        # counts, and make thread-safety tests unreliable.
        #
        # Named URIs (file:wm_<id>?mode=memory&cache=shared) give each instance
        # its own in-memory database while retaining the shared-cache thread safety
        # that thread-local connections require.
        if db_path is None:
            # Sanitise session_id: URIs disallow spaces, slashes, and most punctuation
            import re as _re
            safe_id = _re.sub(r"[^A-Za-z0-9_-]", "_", str(session_id))
            self.db_path = f"file:wm_{safe_id}?mode=memory&cache=shared"
            self._uri = True
        else:
            self.db_path = db_path
            self._uri = False
            
        self.session_id = session_id
        self.max_tokens = max_tokens
        self.token_counter = token_counter or self._default_token_counter
        
        # Thread-safety
        self._lock = threading.RLock()
        self._local = threading.local()
        
        self._init_db()
    
    def _default_token_counter(self, text: str) -> int:
        """Default token estimation: chars / 4."""
        return len(text) // 4
    
    @contextmanager
    def _get_connection(self):
        """Thread-local connection context manager."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0,
                uri=self._uri  # Enable URI mode for shared in-memory cache
            )
            self._local.conn.row_factory = sqlite3.Row
            # WAL mode: allows concurrent reads during writes (critical when
            # the forgetting daemon thread reads while the chat path writes).
            # synchronous=NORMAL is safe with WAL and ~2x faster than FULL.
            # cache_size=-8000 reserves 8MB of page cache per connection.
            try:
                self._local.conn.execute("PRAGMA journal_mode=WAL;")
                self._local.conn.execute("PRAGMA synchronous=NORMAL;")
                self._local.conn.execute("PRAGMA cache_size=-8000;")
            except Exception:
                pass  # In-memory or read-only databases ignore WAL silently

        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    token_count INTEGER NOT NULL,
                    session_id TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_timestamp 
                ON messages(session_id, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_role
                ON messages(session_id, role)
            """)
            
            conn.commit()
    
    def add(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add message to working memory.
        
        Automatically evicts oldest messages if token budget exceeded.
        
        Args:
            role: 'user', 'assistant', or 'system'
            content: Message text
            metadata: Optional metadata dict (e.g., {'importance': 0.8})
            
        Returns:
            Message: Added message with assigned ID
        """
        with self._lock:
            token_count = self.token_counter(content)
            
            message = Message(
                timestamp=time.time(),
                role=role,
                content=content,
                metadata=metadata or {},
                token_count=token_count,
                session_id=self.session_id,
            )
            
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO messages (timestamp, role, content, metadata, token_count, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.timestamp,
                        message.role,
                        message.content,
                        json.dumps(message.metadata),
                        message.token_count,
                        message.session_id,
                    )
                )
                message.id = cursor.lastrowid
                conn.commit()
            
            # Evict old messages if over budget
            self._enforce_token_budget()
            
            return message
    
    def _enforce_token_budget(self):
        """Evict oldest messages until under token budget."""
        with self._lock:
            current_tokens = self.get_token_count()
            
            if current_tokens <= self.max_tokens:
                return
            
            # Calculate how many tokens to remove
            tokens_to_remove = current_tokens - self.max_tokens
            
            with self._get_connection() as conn:
                # Get oldest messages until we've removed enough tokens
                cursor = conn.execute(
                    """
                    SELECT id, token_count FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (self.session_id,)
                )
                
                ids_to_delete = []
                tokens_removed = 0
                
                for row in cursor:
                    ids_to_delete.append(row[0])
                    tokens_removed += row[1]
                    if tokens_removed >= tokens_to_remove:
                        break
                
                if ids_to_delete:
                    placeholders = ','.join('?' * len(ids_to_delete))
                    conn.execute(
                        f"DELETE FROM messages WHERE id IN ({placeholders})",
                        ids_to_delete
                    )
                    conn.commit()
    
    def get_recent(self, n: int = 10) -> List[Message]:
        """
        Get N most recent messages.
        
        Args:
            n: Number of messages to retrieve
            
        Returns:
            List of Messages, newest first
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, timestamp, role, content, metadata, token_count, session_id
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (self.session_id, n)
                )
                
                return [Message.from_row(row) for row in cursor]
    
    def search(
        self,
        query: str,
        limit: int = 10,
        role_filter: Optional[str] = None,
    ) -> List[Message]:
        """
        Keyword search in message content.
        
        Simple substring matching - adequate for working memory.
        
        Args:
            query: Search string
            limit: Max results
            role_filter: Optional role filter ('user', 'assistant', 'system')
            
        Returns:
            List of matching Messages, newest first
        """
        with self._lock:
            with self._get_connection() as conn:
                if role_filter:
                    cursor = conn.execute(
                        """
                        SELECT id, timestamp, role, content, metadata, token_count, session_id
                        FROM messages
                        WHERE session_id = ? AND role = ? AND content LIKE ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (self.session_id, role_filter, f"%{query}%", limit)
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT id, timestamp, role, content, metadata, token_count, session_id
                        FROM messages
                        WHERE session_id = ? AND content LIKE ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (self.session_id, f"%{query}%", limit)
                    )
                
                return [Message.from_row(row) for row in cursor]
    
    def get_context_window(
        self,
        max_tokens: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Message]:
        """Get recent messages that fit within token budget.

        Critical for LLM context window management.
        Returns messages newest-first, automatically trimmed to budget.

        The SQL query caps at _CONTEXT_SCAN_LIMIT rows before Python applies
        the token budget. This prevents long-running sessions (voice interface,
        1000+ turns) from building a full unbounded result set in SQLite.  The
        limit is generous enough that token budgets are always satisfied —
        a 400-token budget with 50-token average messages needs at most 8 rows.

        Args:
            max_tokens: Token budget. None = use self.max_tokens
            include_system: Include system messages in budget calculation

        Returns:
            List of Messages that fit in budget, newest first
        """
        budget = max_tokens or self.max_tokens
        # Upper bound on rows to scan: budget / min_token_per_msg, with headroom.
        # min realistic token count is ~3 (very short turn); 4× headroom for safety.
        _CONTEXT_SCAN_LIMIT = max(200, (budget // 3) * 4)

        with self._lock:
            with self._get_connection() as conn:
                if include_system:
                    cursor = conn.execute(
                        """
                        SELECT id, timestamp, role, content, metadata, token_count, session_id
                        FROM messages
                        WHERE session_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (self.session_id, _CONTEXT_SCAN_LIMIT)
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT id, timestamp, role, content, metadata, token_count, session_id
                        FROM messages
                        WHERE session_id = ? AND role != 'system'
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (self.session_id, _CONTEXT_SCAN_LIMIT)
                    )

                messages = []
                total_tokens = 0

                for row in cursor:
                    msg = Message.from_row(row)
                    if total_tokens + msg.token_count <= budget:
                        messages.append(msg)
                        total_tokens += msg.token_count
                    else:
                        break
                
                return messages
    
    def get_token_count(self) -> int:
        """Get current token usage for this session."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT SUM(token_count) FROM messages
                    WHERE session_id = ?
                    """,
                    (self.session_id,)
                )
                result = cursor.fetchone()[0]
                return result if result is not None else 0
    
    def get_message_count(self) -> int:
        """Get number of messages in this session."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM messages
                    WHERE session_id = ?
                    """,
                    (self.session_id,)
                )
                return cursor.fetchone()[0]
    
    def clear_session(self):
        """Delete all messages in current session."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    "DELETE FROM messages WHERE session_id = ?",
                    (self.session_id,)
                )
                conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dict with message_count, token_count, utilization, etc.
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT 
                        COUNT(*) as message_count,
                        SUM(token_count) as token_count,
                        MIN(timestamp) as oldest_timestamp,
                        MAX(timestamp) as newest_timestamp,
                        COUNT(DISTINCT role) as unique_roles
                    FROM messages
                    WHERE session_id = ?
                    """,
                    (self.session_id,)
                )
                
                row = cursor.fetchone()
                
                message_count = row[0]
                token_count = row[1] or 0
                oldest_ts = row[2]
                newest_ts = row[3]
                unique_roles = row[4]
                
                return {
                    "session_id": self.session_id,
                    "message_count": message_count,
                    "token_count": token_count,
                    "token_budget": self.max_tokens,
                    "utilization": token_count / self.max_tokens if self.max_tokens > 0 else 0,
                    "oldest_timestamp": oldest_ts,
                    "newest_timestamp": newest_ts,
                    "session_duration": newest_ts - oldest_ts if oldest_ts and newest_ts else 0,
                    "unique_roles": unique_roles,
                }
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            delattr(self._local, 'conn')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
