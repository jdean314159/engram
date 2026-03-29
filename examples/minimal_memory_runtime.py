#!/usr/bin/env python3
"""Minimal Engram integration example.

This is the smallest practical example of using Engram as a memory runtime in a
plain Python application.
"""

from pathlib import Path

from engram import ProjectMemory, ProjectType
from engram.engine import create_failover_engine


def main() -> None:
    # 1. Pick an engine profile from your YAML config.
    engine = create_failover_engine("default_local")

    # 2. Create one memory runtime per assistant, project, or workspace.
    memory = ProjectMemory(
        project_id="minimal-demo",
        project_type=ProjectType.GENERAL_ASSISTANT,
        base_dir=Path("./data/memory"),
        llm_engine=engine,
    )

    try:
        # 3. Feed turns into memory as they happen.
        memory.add_turn("user", "I prefer terminal-based tools and concise answers.")
        memory.add_turn(
            "assistant",
            "Understood. I will bias recommendations toward terminal workflows and brevity.",
        )

        # 4. Ask Engram for retrieved context.
        context = memory.get_context(query="tooling preferences", max_tokens=1200)

        # 5. Or ask it to build a prompt under a token budget.
        built = memory.build_prompt(
            user_message="Suggest a terminal-friendly code review workflow.",
            query="tooling preferences code review workflow",
            max_prompt_tokens=4000,
            reserve_output_tokens=400,
        )

        print("Retrieved sections:")
        for section, content in context.to_prompt_sections().items():
            if content:
                preview = content.replace("\n", " ")[:140]
                print(f"- {section}: {preview}")

        print("\nPrompt preview:\n")
        print(built["prompt"][:900])
    finally:
        memory.close()


if __name__ == "__main__":
    main()
