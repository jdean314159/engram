#!/usr/bin/env python3
"""Small headless chat loop using Engram's strategy layer.

This keeps the example aligned with the Phase 2 direction: strategies become
the user-facing execution surface, while ProjectMemory handles persistence,
retrieval, and experiment logging underneath.
"""

from pathlib import Path

from engram import ProjectMemory, ProjectType
from engram.engine import create_failover_engine


def _print_backend_hints() -> None:
    print("\nBackend check suggestions:")
    print("  - Run: ollama list")
    print("  - Run: curl -s http://localhost:11434/api/tags")
    print("  - Run: curl -s http://localhost:8000/v1/models")
    print("  - Verify the configured Ollama model exists locally")
    print("  - Verify the configured vLLM server is running\n")


def main() -> None:
    engine = create_failover_engine("default_local")
    memory = ProjectMemory(
        project_id="headless-chat",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=Path("./data/memory"),
        llm_engine=engine,
    )

    print("Type 'quit' to exit. Memory is persisted per project.")
    print(f"Available strategies: {', '.join(memory.available_strategies())}")
    try:
        while True:
            user_message = input("you> ").strip()
            if not user_message:
                continue
            if user_message.lower() in {"quit", "exit"}:
                break

            try:
                result = memory.run_strategy(
                    "direct_answer",
                    user_message,
                    query=user_message,
                    max_prompt_tokens=4000,
                    reserve_output_tokens=400,
                )
                print(f"assistant> {result['reply']}")
            except Exception as e:
                print(f"\nassistant> request failed: {e}")
                failures = memory.recent_failures(limit=1)
                if failures:
                    latest = failures[0]
                    print(
                        f"latest failed run: run_id={latest.get('run_id')} "
                        f"failure_mode={latest.get('failure_mode')}"
                    )
                _print_backend_hints()

            if hasattr(memory, "run_lifecycle_maintenance"):
                memory.run_lifecycle_maintenance()
    finally:
        memory.close()


if __name__ == "__main__":
    main()
