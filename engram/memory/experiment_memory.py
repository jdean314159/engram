from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExperimentRecord:
    run_id: str
    project_id: str
    session_id: str
    parent_run_id: Optional[str]
    created_at: float
    finished_at: Optional[float]
    status: str
    task_type: str
    problem_family: Optional[str]
    goal: str
    strategy: Optional[str]
    backend_label: Optional[str]
    model_name: Optional[str]
    prompt_strategy: Optional[str]
    tool_strategy: Optional[str]
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    retrieval: dict[str, Any]
    outcome_summary: Optional[str]
    failure_mode: Optional[str]
    lessons_learned: list[str]
    artifacts: list[str]


class ExperimentMemory:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                parent_run_id TEXT,
                created_at REAL NOT NULL,
                finished_at REAL,
                status TEXT NOT NULL,
                task_type TEXT NOT NULL,
                problem_family TEXT,
                goal TEXT NOT NULL,
                strategy TEXT,
                backend_label TEXT,
                model_name TEXT,
                prompt_strategy TEXT,
                tool_strategy TEXT,
                parameters_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                retrieval_json TEXT NOT NULL,
                outcome_summary TEXT,
                failure_mode TEXT,
                lessons_json TEXT NOT NULL,
                artifacts_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_runs_project_created
            ON runs(project_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_runs_task_type
            ON runs(task_type);

            CREATE INDEX IF NOT EXISTS idx_runs_strategy
            ON runs(strategy);

            CREATE INDEX IF NOT EXISTS idx_runs_failure_mode
            ON runs(failure_mode);
            """
        )
        self.conn.commit()

    def start_run(
        self,
        *,
        project_id: str,
        session_id: str,
        goal: str,
        task_type: str = "chat",
        problem_family: Optional[str] = None,
        strategy: Optional[str] = None,
        prompt_strategy: Optional[str] = None,
        tool_strategy: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        created_at = time.time()
        self.conn.execute(
            """
            INSERT INTO runs (
                run_id, project_id, session_id, parent_run_id,
                created_at, finished_at, status,
                task_type, problem_family, goal, strategy,
                backend_label, model_name, prompt_strategy, tool_strategy,
                parameters_json, metrics_json, retrieval_json,
                outcome_summary, failure_mode, lessons_json, artifacts_json
            ) VALUES (?, ?, ?, ?, ?, NULL, 'started', ?, ?, ?, ?, NULL, NULL, ?, ?, ?, '{}', '{}', NULL, NULL, '[]', '[]')
            """,
            (
                run_id,
                project_id,
                session_id,
                parent_run_id,
                created_at,
                task_type,
                problem_family,
                goal,
                strategy,
                prompt_strategy,
                tool_strategy,
                json.dumps(parameters or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        backend_label: Optional[str] = None,
        model_name: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
        retrieval: Optional[dict[str, Any]] = None,
        outcome_summary: Optional[str] = None,
        failure_mode: Optional[str] = None,
        lessons_learned: Optional[list[str]] = None,
        artifacts: Optional[list[str]] = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE runs
            SET finished_at = ?,
                status = ?,
                backend_label = ?,
                model_name = ?,
                metrics_json = ?,
                retrieval_json = ?,
                outcome_summary = ?,
                failure_mode = ?,
                lessons_json = ?,
                artifacts_json = ?
            WHERE run_id = ?
            """,
            (
                time.time(),
                status,
                backend_label,
                model_name,
                json.dumps(metrics or {}, ensure_ascii=False),
                json.dumps(retrieval or {}, ensure_ascii=False),
                outcome_summary,
                failure_mode,
                json.dumps(lessons_learned or [], ensure_ascii=False),
                json.dumps(artifacts or [], ensure_ascii=False),
                run_id,
            ),
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def search_runs(
        self,
        *,
        task_type: Optional[str] = None,
        strategy: Optional[str] = None,
        failure_mode: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []

        if task_type:
            clauses.append("task_type = ?")
            params.append(task_type)
        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)
        if failure_mode:
            clauses.append("failure_mode = ?")
            params.append(failure_mode)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT * FROM runs
            {where}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
        
    def recent_failures(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM runs
            WHERE failure_mode IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
