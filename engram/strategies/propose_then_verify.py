from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from .base import ReasoningStrategy
from ..verifiers import PassThroughVerifier


class ProposeThenVerifyStrategy(ReasoningStrategy):
    """Generate multiple candidates, verify them, then select the best one."""

    name = "propose_then_verify"

    @staticmethod
    def _select_best(
        candidates: List[str],
        verification_results: List[dict[str, Any]],
    ) -> Tuple[int, str, dict[str, Any]]:
        if not candidates:
            empty = {"passed": False, "score": 0.0, "reason": "no_candidates", "metadata": {}}
            return 0, "", empty

        best_idx = 0
        best_key = None

        for idx, (candidate, vr) in enumerate(zip(candidates, verification_results)):
            passed = bool(vr.get("passed", False))
            score = float(vr.get("score", 0.0))
            length = len((candidate or "").strip())

            key = (
                1 if passed else 0,
                score,
                length,
                -idx,  # deterministic tie-break toward earlier candidates
            )
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx

        return best_idx, candidates[best_idx], verification_results[best_idx]

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
        verifier = kwargs.pop("verifier", None) or PassThroughVerifier()
        verifier_name = getattr(verifier, "name", verifier.__class__.__name__)

        parameters = dict(kwargs.pop("parameters", {}) or {})
        parameters.update(
            {
                "n_candidates": n_candidates,
                "verifier_name": verifier_name,
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
            verification_results: List[dict[str, Any]] = []

            for _ in range(max(1, n_candidates)):
                candidate = memory.llm_engine.generate(built["prompt"])
                candidates.append(candidate)
                vr = verifier.verify(
                    candidate,
                    user_message=user_message,
                    query=query or user_message,
                )
                verification_results.append(vr.to_dict())

            selected_index, reply, selected_verification = self._select_best(
                candidates,
                verification_results,
            )
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

            pass_count = sum(1 for vr in verification_results if vr.get("passed"))
            metrics = {
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "prompt_tokens": built.get("prompt_tokens"),
                "memory_tokens": built.get("memory_tokens"),
                "compressed": built.get("compressed", False),
                "candidate_count": len(candidates),
                "verification_pass_count": pass_count,
                "selected_index": selected_index,
                "selected_score": float(selected_verification.get("score", 0.0)),
                "selected_passed": bool(selected_verification.get("passed", False)),
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
                "verification_results": verification_results,
                "selected_index": selected_index,
                "selected_verification": selected_verification,
                "verifier_name": verifier_name,
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
