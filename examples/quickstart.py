#!/usr/bin/env python3
"""Engram quickstart.

This example demonstrates the core library directly. For a visual walkthrough,
run the sandbox reference app instead:

    streamlit run apps/sandbox/app.py
"""

from pathlib import Path

from engram import NeuralMemoryConfig, ProjectMemory, ProjectType
from engram.engine import create_failover_engine


def main() -> None:
    engine = create_failover_engine("default_local")
    memory = ProjectMemory(
        project_id="quickstart-demo",
        project_type=ProjectType.PROGRAMMING_ASSISTANT,
        base_dir=Path("./data/memory"),
        llm_engine=engine,
        neural_config=NeuralMemoryConfig(),
        calibration_required=False,
    )

    try:
        memory.add_turn("user", "How do I use asyncio.gather()?")
        memory.add_turn(
            "assistant",
            "asyncio.gather runs awaitables concurrently and returns results in order.",
        )
        memory.store_episode(
            "User is learning async Python and asked about asyncio.gather().",
            importance=0.8,
        )

        context = memory.get_context(query="asyncio error handling", max_tokens=2000)
        built = memory.build_prompt(
            user_message="Show a safe error-handling pattern for asyncio.gather().",
            query="asyncio gather error handling",
            max_prompt_tokens=6000,
            reserve_output_tokens=600,
        )

        print("Context sections:")
        for section, content in context.to_prompt_sections().items():
            if content:
                print(f"\n[{section}]\n{content[:300]}")

        print("\nPrompt preview:\n")
        print(built["prompt"][:1200])

        if hasattr(memory, "run_lifecycle_maintenance"):
            report = memory.run_lifecycle_maintenance()
            print("\nLifecycle summary:")
            print(report)
    finally:
        memory.close()


if __name__ == "__main__":
    main()
