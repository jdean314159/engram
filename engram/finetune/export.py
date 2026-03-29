"""Fine-tuning data export for Engram.

Extracts conversation pairs from Engram's memory layers and writes them
in standard training formats (OpenAI JSONL, Alpaca JSONL, raw text).

Source priority:
  1. Episodic memory — ChromaDB episodes, grouped by session_id
  2. Cold storage — archived episodes, grouped by session_id
  3. Working memory DB — direct SQLite scan across all sessions

Session grouping reconstructs user/assistant alternation from stored turns.
Episodes that lack a partner (lone user or lone assistant turn) are skipped
unless include_incomplete=True, in which case they emit single-turn records.

Output formats
--------------
openai   {"messages": [{"role": "system", "content": ...},
                        {"role": "user", "content": ...},
                        {"role": "assistant", "content": ...}]}
alpaca   {"instruction": <user turn>, "input": "", "output": <assistant turn>}
raw      Plain text alternating turns, sessions separated by blank lines.

Usage
-----
    from engram.finetune.export import ExportConfig, export_to_file

    cfg = ExportConfig(min_importance=0.6, min_chars=20)
    n = export_to_file(project_memory, Path("train.jsonl"), format="openai", config=cfg)
    print(f"Exported {n} records")

Author: Jeffrey Dean
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExportConfig:
    """Configuration for fine-tuning data export.

    Args:
        min_importance: Skip episodes with importance below this value.
                        0.0 = include everything.
        min_chars: Skip turns shorter than this many characters.
        max_chars: Truncate turns longer than this (0 = no limit).
        include_system: Include system prompt in openai-format records.
        system_prompt: System prompt text for openai format.
                       None = use the engine's system prompt if available.
        include_incomplete: If True, emit single-turn records when a
                            user/assistant pair cannot be completed.
        dedup: Remove duplicate pairs (matched by SHA-256 of content).
        days_back: Export only episodes from the last N days.
                   0 = no time filter.
        session_ids: If set, only export these session IDs.
        min_pair_chars: Skip pairs whose combined char count is below this.
    """
    min_importance: float = 0.0
    min_chars: int = 10
    max_chars: int = 0
    include_system: bool = True
    system_prompt: Optional[str] = None
    include_incomplete: bool = False
    dedup: bool = True
    days_back: int = 0
    session_ids: Optional[List[str]] = None
    min_pair_chars: int = 20


# ---------------------------------------------------------------------------
# Internal turn representation
# ---------------------------------------------------------------------------

@dataclass
class _Turn:
    role: str       # "user" | "assistant" | "system"
    content: str
    timestamp: float
    session_id: str
    importance: float = 0.5
    source: str = ""   # "episodic" | "cold" | "working"


@dataclass
class _Session:
    session_id: str
    turns: List[_Turn] = field(default_factory=list)

    def sorted_turns(self) -> List[_Turn]:
        return sorted(self.turns, key=lambda t: t.timestamp)


# ---------------------------------------------------------------------------
# Source readers
# ---------------------------------------------------------------------------

def _cutoff_ts(days_back: int) -> float:
    if days_back <= 0:
        return 0.0
    return time.time() - days_back * 86400.0


def _read_episodic(project_memory, config: ExportConfig) -> List[_Turn]:
    """Read turns from episodic (ChromaDB) memory."""
    episodic = getattr(project_memory, "episodic", None)
    if episodic is None:
        return []

    cutoff = _cutoff_ts(config.days_back)

    try:
        where: Dict[str, Any] = {"project_id": project_memory.project_id}
        if cutoff > 0:
            where = {"$and": [where, {"timestamp": {"$gte": cutoff}}]}

        results = episodic.collection.get(where=where)
    except Exception as e:
        logger.warning("Episodic export failed: %s", e)
        return []

    turns: List[_Turn] = []
    for i, doc_id in enumerate(results.get("ids", [])):
        meta = results["metadatas"][i]
        text = results["documents"][i]
        importance = float(meta.get("importance", 0.5))
        if importance < config.min_importance:
            continue

        # Episodes store their role in metadata; default to "user"
        inner = meta.get("metadata", "{}")
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except Exception:
                inner = {}
        role = inner.get("role", meta.get("role", "user"))

        ts = float(meta.get("timestamp", 0))
        sid = meta.get("session_id", "unknown")

        turns.append(_Turn(
            role=role,
            content=text,
            timestamp=ts,
            session_id=sid,
            importance=importance,
            source="episodic",
        ))

    return turns


def _read_cold(project_memory, config: ExportConfig) -> List[_Turn]:
    """Read turns from cold storage (SQLite FTS)."""
    cold = getattr(project_memory, "cold", None)
    if cold is None:
        return []

    cutoff = _cutoff_ts(config.days_back)

    try:
        rows = cold.retrieve("", n=10_000)
    except Exception as e:
        logger.warning("Cold export failed: %s", e)
        return []

    turns: List[_Turn] = []
    for row in rows:
        ts = float(row.get("timestamp", 0))
        if cutoff > 0 and ts < cutoff:
            continue
        if row.get("project_id") != project_memory.project_id:
            continue

        meta = row.get("metadata", {}) or {}
        importance = float(meta.get("importance", 0.5))
        if importance < config.min_importance:
            continue

        orig = meta.get("original_metadata", {})
        if isinstance(orig, str):
            try:
                orig = json.loads(orig)
            except Exception:
                orig = {}
        role = orig.get("role", meta.get("role", "user"))

        sid = row.get("session_id", "unknown")
        turns.append(_Turn(
            role=role,
            content=row.get("text", ""),
            timestamp=ts,
            session_id=sid,
            importance=importance,
            source="cold",
        ))

    return turns


def _read_working_db(project_memory, config: ExportConfig) -> List[_Turn]:
    """Read turns directly from working.db SQLite across all sessions.

    Working memory SQL is scoped to session_id; we bypass the WorkingMemory
    object and query the db directly to get all sessions.
    """
    db_path = getattr(project_memory, "_project_dir", None)
    if db_path is None:
        return []
    db_file = Path(db_path) / "working.db"
    if not db_file.exists():
        return []

    cutoff = _cutoff_ts(config.days_back)

    turns: List[_Turn] = []
    try:
        conn = sqlite3.connect(str(db_file))
        conn.row_factory = sqlite3.Row
        sql = "SELECT timestamp, role, content, session_id FROM messages"
        params: List[Any] = []
        clauses: List[str] = []
        if config.session_ids:
            placeholders = ",".join("?" * len(config.session_ids))
            clauses.append(f"session_id IN ({placeholders})")
            params.extend(config.session_ids)
        if cutoff > 0:
            clauses.append("timestamp >= ?")
            params.append(cutoff)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY session_id, timestamp"

        for row in conn.execute(sql, params):
            turns.append(_Turn(
                role=row["role"],
                content=row["content"],
                timestamp=float(row["timestamp"]),
                session_id=row["session_id"],
                source="working",
            ))
        conn.close()
    except Exception as e:
        logger.warning("Working DB export failed: %s", e)

    return turns


# ---------------------------------------------------------------------------
# Session grouping and pair extraction
# ---------------------------------------------------------------------------

def _group_by_session(turns: List[_Turn]) -> Dict[str, _Session]:
    sessions: Dict[str, _Session] = {}
    for turn in turns:
        sid = turn.session_id
        if sid not in sessions:
            sessions[sid] = _Session(session_id=sid)
        sessions[sid].turns.append(turn)
    return sessions


def _extract_pairs(
    session: _Session,
    config: ExportConfig,
) -> List[Tuple[Optional[_Turn], Optional[_Turn]]]:
    """Extract (user, assistant) pairs from a session's sorted turns.

    Consecutive same-role turns are concatenated with a newline.
    """
    turns = [t for t in session.sorted_turns() if t.role in ("user", "assistant")]
    if not turns:
        return []

    # Merge consecutive same-role turns
    merged: List[_Turn] = []
    for t in turns:
        if merged and merged[-1].role == t.role:
            merged[-1].content += "\n" + t.content
            merged[-1].timestamp = t.timestamp
        else:
            merged.append(_Turn(
                role=t.role,
                content=t.content,
                timestamp=t.timestamp,
                session_id=t.session_id,
                importance=t.importance,
                source=t.source,
            ))

    pairs: List[Tuple[Optional[_Turn], Optional[_Turn]]] = []
    i = 0
    while i < len(merged):
        cur = merged[i]
        if cur.role == "user":
            if i + 1 < len(merged) and merged[i + 1].role == "assistant":
                pairs.append((cur, merged[i + 1]))
                i += 2
            else:
                if config.include_incomplete:
                    pairs.append((cur, None))
                i += 1
        elif cur.role == "assistant":
            if config.include_incomplete:
                pairs.append((None, cur))
            i += 1
        else:
            i += 1

    return pairs


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def _apply_char_limits(text: str, config: ExportConfig) -> Optional[str]:
    """Return text after length filtering and optional truncation."""
    if len(text) < config.min_chars:
        return None
    if config.max_chars > 0 and len(text) > config.max_chars:
        text = text[: config.max_chars]
    return text


def _build_openai_record(
    user: Optional[_Turn],
    assistant: Optional[_Turn],
    config: ExportConfig,
    system_text: Optional[str],
) -> Optional[Dict[str, Any]]:
    messages = []
    if config.include_system and system_text:
        messages.append({"role": "system", "content": system_text})
    if user is not None:
        content = _apply_char_limits(user.content.strip(), config)
        if content is None:
            return None
        messages.append({"role": "user", "content": content})
    if assistant is not None:
        content = _apply_char_limits(assistant.content.strip(), config)
        if content is None:
            return None
        messages.append({"role": "assistant", "content": content})
    if len(messages) < (2 if not (config.include_system and system_text) else 3) and not config.include_incomplete:
        return None
    return {"messages": messages}


def _build_alpaca_record(
    user: Optional[_Turn],
    assistant: Optional[_Turn],
    config: ExportConfig,
) -> Optional[Dict[str, Any]]:
    if user is None or assistant is None:
        if not config.include_incomplete:
            return None
    instruction = _apply_char_limits((user.content.strip() if user else ""), config)
    output = _apply_char_limits((assistant.content.strip() if assistant else ""), config)
    if instruction is None or output is None:
        return None
    combined = (instruction or "") + (output or "")
    if len(combined) < config.min_pair_chars:
        return None
    return {"instruction": instruction, "input": "", "output": output}


def _build_raw_record(
    user: Optional[_Turn],
    assistant: Optional[_Turn],
    config: ExportConfig,
) -> Optional[str]:
    parts = []
    if user is not None:
        content = _apply_char_limits(user.content.strip(), config)
        if content:
            parts.append(f"User: {content}")
    if assistant is not None:
        content = _apply_char_limits(assistant.content.strip(), config)
        if content:
            parts.append(f"Assistant: {content}")
    if not parts:
        return None
    return "\n".join(parts)


def _pair_hash(user: Optional[_Turn], assistant: Optional[_Turn]) -> str:
    u = user.content if user else ""
    a = assistant.content if assistant else ""
    return hashlib.sha256(f"{u}\x00{a}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_to_file(
    project_memory,
    path: Path,
    format: str = "openai",
    config: Optional[ExportConfig] = None,
) -> int:
    """Export fine-tuning data from a ProjectMemory instance to a file.

    Args:
        project_memory: ProjectMemory instance.
        path: Output file path. Created/overwritten.
        format: "openai" | "alpaca" | "raw"
        config: ExportConfig. None = defaults.

    Returns:
        Number of records written.

    Raises:
        ValueError: Unknown format.
    """
    if format not in ("openai", "alpaca", "raw"):
        raise ValueError(f"Unknown format {format!r}. Choose: openai, alpaca, raw")

    config = config or ExportConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve system prompt
    system_text: Optional[str] = config.system_prompt
    if system_text is None and format == "openai":
        engine = getattr(project_memory, "llm_engine", None)
        if engine is not None:
            system_text = getattr(engine, "system_prompt", None)

    # Collect turns from all available sources
    all_turns: List[_Turn] = []
    all_turns.extend(_read_episodic(project_memory, config))
    all_turns.extend(_read_cold(project_memory, config))
    all_turns.extend(_read_working_db(project_memory, config))

    if not all_turns:
        logger.info("No turns found for project %s", project_memory.project_id)
        path.write_text("")
        return 0

    # Filter by session_ids
    if config.session_ids:
        sid_set = set(config.session_ids)
        all_turns = [t for t in all_turns if t.session_id in sid_set]

    # Deduplicate turns by (session_id, role, content) before grouping
    seen_turns: set = set()
    deduped_turns: List[_Turn] = []
    for t in all_turns:
        key = hashlib.sha256(f"{t.session_id}\x00{t.role}\x00{t.content}".encode()).hexdigest()
        if key not in seen_turns:
            seen_turns.add(key)
            deduped_turns.append(t)

    sessions = _group_by_session(deduped_turns)
    logger.info("Exporting from %d sessions, %d turns", len(sessions), len(deduped_turns))

    seen_pairs: set = set()
    written = 0

    with open(path, "w", encoding="utf-8") as f:
        for sid, session in sorted(sessions.items()):
            pairs = _extract_pairs(session, config)
            session_records: List[str] = []

            for user, assistant in pairs:
                if config.dedup:
                    ph = _pair_hash(user, assistant)
                    if ph in seen_pairs:
                        continue
                    seen_pairs.add(ph)

                if format == "openai":
                    rec = _build_openai_record(user, assistant, config, system_text)
                    if rec is not None:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
                elif format == "alpaca":
                    rec = _build_alpaca_record(user, assistant, config)
                    if rec is not None:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
                else:  # raw
                    rec_str = _build_raw_record(user, assistant, config)
                    if rec_str is not None:
                        session_records.append(rec_str)

            if format == "raw" and session_records:
                f.write("\n\n".join(session_records) + "\n\n")
                written += len(session_records)

    logger.info("Exported %d records to %s (format=%s)", written, path, format)
    return written


def export_stats(project_memory, config: Optional[ExportConfig] = None) -> Dict[str, Any]:
    """Dry-run: return counts without writing any file.

    Useful for estimating dataset size before committing to export.
    """
    config = config or ExportConfig()

    all_turns: List[_Turn] = []
    all_turns.extend(_read_episodic(project_memory, config))
    all_turns.extend(_read_cold(project_memory, config))
    all_turns.extend(_read_working_db(project_memory, config))

    sessions = _group_by_session(all_turns)
    total_pairs = 0
    complete_pairs = 0
    for session in sessions.values():
        pairs = _extract_pairs(session, config)
        total_pairs += len(pairs)
        complete_pairs += sum(1 for u, a in pairs if u is not None and a is not None)

    return {
        "total_turns": len(all_turns),
        "sessions": len(sessions),
        "total_pairs": total_pairs,
        "complete_pairs": complete_pairs,
        "incomplete_pairs": total_pairs - complete_pairs,
    }
