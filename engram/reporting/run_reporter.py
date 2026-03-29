from __future__ import annotations

import json
from typing import Any, Dict, List


class RunReporter:
    """Small reporting helper over ExperimentMemory records."""

    def __init__(self, experiment_memory):
        self.experiments = experiment_memory

    @staticmethod
    def _loads_json(value, default):
        if not value:
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return default

    def recent_run_summaries(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self.experiments.recent_runs(limit=limit)
        summaries: List[Dict[str, Any]] = []

        for row in rows:
            metrics = self._loads_json(row.get("metrics_json"), {})
            summaries.append(
                {
                    "run_id": row.get("run_id"),
                    "created_at": row.get("created_at"),
                    "finished_at": row.get("finished_at"),
                    "status": row.get("status"),
                    "task_type": row.get("task_type"),
                    "problem_family": row.get("problem_family"),
                    "strategy": row.get("strategy"),
                    "backend_label": row.get("backend_label"),
                    "model_name": row.get("model_name"),
                    "failure_mode": row.get("failure_mode"),
                    "duration_ms": metrics.get("duration_ms"),
                    "candidate_count": metrics.get("candidate_count"),
                    "selected_passed": metrics.get("selected_passed"),
                    "outcome_summary": row.get("outcome_summary"),
                }
            )

        return summaries

    def recent_failure_summaries(self, limit: int = 10) -> List[Dict[str, Any]]:
        rows = self.experiments.recent_failures(limit=limit)
        summaries: List[Dict[str, Any]] = []

        for row in rows:
            metrics = self._loads_json(row.get("metrics_json"), {})
            summaries.append(
                {
                    "run_id": row.get("run_id"),
                    "created_at": row.get("created_at"),
                    "task_type": row.get("task_type"),
                    "strategy": row.get("strategy"),
                    "backend_label": row.get("backend_label"),
                    "model_name": row.get("model_name"),
                    "failure_mode": row.get("failure_mode"),
                    "duration_ms": metrics.get("duration_ms"),
                    "outcome_summary": row.get("outcome_summary"),
                }
            )

        return summaries

    def strategy_summary(self, *, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self.experiments.recent_runs(limit=limit)
        grouped: Dict[str, Dict[str, Any]] = {}

        for row in rows:
            strategy = row.get("strategy") or "<unknown>"
            metrics = self._loads_json(row.get("metrics_json"), {})

            entry = grouped.setdefault(
                strategy,
                {
                    "strategy": strategy,
                    "run_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "avg_duration_ms": 0.0,
                    "avg_candidate_count": 0.0,
                    "verified_selected_count": 0,
                },
            )

            entry["run_count"] += 1
            if row.get("status") == "succeeded":
                entry["success_count"] += 1
            if row.get("failure_mode") is not None or row.get("status") == "failed":
                entry["failure_count"] += 1

            duration = metrics.get("duration_ms")
            if isinstance(duration, (int, float)):
                entry["avg_duration_ms"] += float(duration)

            candidate_count = metrics.get("candidate_count")
            if isinstance(candidate_count, (int, float)):
                entry["avg_candidate_count"] += float(candidate_count)

            if metrics.get("selected_passed") is True:
                entry["verified_selected_count"] += 1

        results: List[Dict[str, Any]] = []
        for entry in grouped.values():
            run_count = max(1, entry["run_count"])
            entry["avg_duration_ms"] = round(entry["avg_duration_ms"] / run_count, 3)
            entry["avg_candidate_count"] = round(entry["avg_candidate_count"] / run_count, 3)
            results.append(entry)

        results.sort(key=lambda x: (-x["run_count"], x["strategy"]))
        return results
