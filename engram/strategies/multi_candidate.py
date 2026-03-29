from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from .base import ReasoningStrategy


class MultiCandidateStrategy(ReasoningStrategy):
    """Generate multiple candidate answers and pick one deterministically.

    V1 keeps selection deliberately simple and general-purpose. It does not yet
    use a verifier or benchmark-specific scorer.
    """

    name = "multi_candidate"

    @staticmethod
    def _select_candidate(candidates: List[str], selection_policy: str) -> Tuple[int, str]:
        if not candidates:
            return 0, ""

        normalized = [c if isinstance(c, str) else "" for c in candidates]

        if selection_policy == "first_non_empty":
            for idx, candidate in enumerate(normalized):
                if candidate.strip():
                    return idx, candidate
            return 0, normalized[0]

        if selection_policy == "longest_non_empty":
            best_idx = 0
            best_len = -1
            for idx, candidate in enumerate(normalized):
                cand_len = len(candidate.strip())
                if cand_len > best_len:
                    best_len = cand_len
                    best_idx = idx
            return best_idx, normalized[best_idx]

        raise ValueError(
            f"Unknown selection_policy '{selection_policy}'. "
            f"Expected 'first_non_empty' or 'longest_non_empty'."
        )

    def run(self, memory, user_message: str, **kwargs) -> Dict[str, Any]:
        query = kwargs.pop("query", None)
        semantic_query = kwargs.pop("semantic_query", None)
        max_prompt_tokens = kwargs.pop("max_prompt_tokens", None)
        reserve_output_tokens = kwargs.pop("reserve_output_tokens", 512)
        include_cold_fallback = kwargs.pop("include_cold_fallback", True)
        store_overflow_summary = kwargs.pop("store_overflow_summary", False)

        task_type = kwargs.pop("task_type", "chat")
        problem_family = kwargs.pop("problem_family", None)
        prompt_strategy = kwargs.pop("prompt_strategy", None)
        tool_strategy = kwargs.pop("tool_strategy", None)
        parent_run_id = kwargs.pop("parent_run_id", None)

        n_candidates = int(kwargs.pop("n_candidates", 3))
        selection_policy = kwargs.pop("selection_policy", "first_non_empty")

        parameters = dict(kwargs.pop("parameters", {}) or {})
        parameters.update(
            {
                "n_candidates": n_candidates,
                "selection_policy": selection_policy,
            }
        )

        started = time.perf_counter()

        run_id = memory.experiments.start_run(
            project_id=memory.project_id,
            session_id=memory.session_id,
            goal=user_message,
            task_type=task_type,
            problem_family=problem_family,
            strategy=self.name,
            prompt_strategy=prompt_strategy,
            tool_strategy=tool_strategy,
            parent_run_id=parent_run_id,
            parameters=parameters,
        )

        try:
            memory.add_turn("user", user_message)

            built = memory.build_prompt(
                user_message=user_message,
                query=query or user_message,
                semantic_query=semantic_query,
                max_prompt_tokens=max_prompt_tokens,
                reserve_output_tokens=reserve_output_tokens,
                include_cold_fallback=include_cold_fallback,
                store_overflow_summary=store_overflow_summary,
            )

            candidates: List[str] = []
            for _ in range(max(1, n_candidates)):
                candidate = memory.llm_engine.generate(built["prompt"])
                candidates.append(candidate)

            selected_index, reply = self._select_candidate(candidates, selection_policy)
            memory.add_turn("assistant", reply)

            backend_label = getattr(memory.llm_engine, "backend_label", None)
            model_name = getattr(memory.llm_engine, "model_name", None)

            context_obj = built.get("context")
            token_counts = {}
            if context_obj is not None:
                token_counts = {
                    "working_tokens": getattr(context_obj, "working_tokens", 0),
                    "episodic_tokens": getattr(context_obj, "episodic_tokens", 0),
                    "semantic_tokens": getattr(context_obj, "semantic_tokens", 0),
                    "cold_tokens": getattr(context_obj, "cold_tokens", 0),
                    "total_tokens": getattr(context_obj, "total_tokens", 0),
                }

            metrics = {
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "prompt_tokens": built.get("prompt_tokens"),
                "memory_tokens": built.get("memory_tokens"),
                "compressed": built.get("compressed", False),
                "candidate_count": len(candidates),
                "selected_index": selected_index,
            }

            retrieval = {
                "query": query or user_message,
                "semantic_query": semantic_query,
                "token_counts": token_counts,
            }

            memory.experiments.finish_run(
                run_id,
                status="succeeded",
                backend_label=backend_label,
                model_name=model_name,
                metrics=metrics,
                retrieval=retrieval,
                outcome_summary=(reply[:400] if reply else None),
                failure_mode=None,
                lessons_learned=[],
                artifacts=[],
            )

            memory._store_experiment_episode_summary(
                run_id=run_id,
                user_message=user_message,
                reply=reply,
                strategy=self.name,
                task_type=task_type,
                problem_family=problem_family,
                backend_label=backend_label,
                model_name=model_name,
                metrics=metrics,
            )

            return {
                "run_id": run_id,
                "reply": reply,
                "prompt": built["prompt"],
                "context": built["context"],
                "compressed": built.get("compressed", False),
                "prompt_tokens": built.get("prompt_tokens"),
                "memory_tokens": built.get("memory_tokens"),
                "metrics": metrics,
                "strategy": self.name,
                "candidates": candidates,
                "selected_index": selected_index,
                "selection_policy": selection_policy,
            }

        except Exception as e:
            backend_label = getattr(memory.llm_engine, "backend_label", None) if memory.llm_engine else None
            model_name = getattr(memory.llm_engine, "model_name", None) if memory.llm_engine else None

            memory.experiments.finish_run(
                run_id,
                status="failed",
                backend_label=backend_label,
                model_name=model_name,
                metrics={
                    "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "candidate_count": max(1, n_candidates),
                },
                retrieval={
                    "query": query or user_message,
                    "semantic_query": semantic_query,
                },
                outcome_summary=None,
                failure_mode=type(e).__name__,
                lessons_learned=[],
                artifacts=[],
            )
            raise
